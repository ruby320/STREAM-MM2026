"""
GPT2-XL baseline benchmark: AdamW / 8bit-AdamW / Adam-mini / MDQ / GaLore.

记录 loss、grad_norm、quant_error 随 step 变化；checkpoint 与 CSV 输出至 baseline 目录。
"""
from __future__ import annotations

import argparse
import math
import os

import pandas as pd
import torch
import torch.distributed as dist
import bitsandbytes as bnb
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Config, GPT2LMHeadModel

from mdqbock_newhook import MDQAdamW

try:
    from adam_mini import Adam_mini
except ImportError:
    Adam_mini = None

try:
    from galore_torch import GaLoreAdamW
except ImportError:
    GaLoreAdamW = None

_BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= 配置区 =================
MODEL_TYPE = "gpt2-xl"
DATA_PATH = "/workspace/data/openwebtext_processed"
CKPT_DIR = os.path.join(_BASELINE_DIR, "ckpts")
RESULT_DIR = os.path.join(_BASELINE_DIR, "results")

MAX_STEPS = 1500
BATCH_SIZE_PER_GPU = 4
GRAD_ACCUM_STEPS = 16
SEQ_LEN = 1024
LR = 2e-4

LOG_INTERVAL = 10
CKPT_INTERVAL = 100

GALORE_RANK = 128
GALORE_UPDATE_PROJ_GAP = 200
GALORE_SCALE = 0.25
GALORE_PROJ_TYPE = "std"

ALL_OPTIMIZERS = ["AdamW", "8bit-AdamW", "Adam-mini", "MDQ", "GaLore"]
# ==========================================


def parse_args():
    p = argparse.ArgumentParser(description="GPT2-XL baseline optimizer benchmark")
    p.add_argument(
        "--optimizer",
        choices=ALL_OPTIMIZERS + ["all"],
        default="all",
        help="运行单个优化器或 all（默认全部跑）",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="若存在 ckpts/{opt}_latest.pt 则从 checkpoint 恢复",
    )
    return p.parse_args()


def actual_model(model):
    return model.module if hasattr(model, "module") else model


def compute_grad_norm(model) -> float:
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().float().norm(2).item() ** 2
    return math.sqrt(total_sq)


def compute_mdq_quant_error(optimizer: MDQAdamW) -> float:
    num = 0.0
    den = 0.0
    for group in optimizer.param_groups:
        block_size = group["block_size"]
        for p in group["params"]:
            state = optimizer.state.get(p)
            if not state or "exp_avg" not in state:
                continue
            bit = state.get("current_bit", 32)
            m = state["exp_avg"]
            v = state["exp_avg_sq"]
            q_m = optimizer.robust_quantize(m, bit, is_v=False, block_size=block_size)
            q_v = optimizer.robust_quantize(v, bit, is_v=True, block_size=block_size)
            num += (m - q_m).pow(2).sum().item() + (v - q_v).pow(2).sum().item()
            den += m.pow(2).sum().item() + v.pow(2).sum().item()
    return math.sqrt(num / (den + 1e-12))


def build_adam_mini(model, config: GPT2Config):
    """Adam-mini 官方 API + GPT-2 参数名映射（c_attn / attn.c_proj）。"""
    m = actual_model(model)
    optimizer = Adam_mini(
        named_parameters=m.named_parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        dim=config.n_embd,
        n_heads=config.n_head,
        n_kv_heads=None,
    )
    # GPT-2 的 QKV 合并在 c_attn，输出投影为 attn.c_proj（区别于 mlp.c_proj）
    optimizer.wqk_names.add("c_attn")
    optimizer.wv_names.add("c_attn")
    optimizer.attn_proj_names.add("attn.c_proj")
    return optimizer


def build_galore_optimizer(model):
    """GaLore-AdamW：对 2D 权重做低秩梯度投影，bias/LayerNorm 走普通 AdamW。"""
    if GaLoreAdamW is None:
        raise ImportError("请先执行: pip install galore-torch")

    m = actual_model(model)
    no_decay = ["bias", "LayerNorm.weight"]
    galore_params = []
    non_galore_params = []

    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and not any(nd in name for nd in no_decay):
            galore_params.append(p)
        else:
            non_galore_params.append(p)

    param_groups = [
        {"params": non_galore_params, "weight_decay": 0.0},
        {
            "params": galore_params,
            "weight_decay": 0.01,
            "rank": GALORE_RANK,
            "update_proj_gap": GALORE_UPDATE_PROJ_GAP,
            "scale": GALORE_SCALE,
            "proj_type": GALORE_PROJ_TYPE,
        },
    ]
    return GaLoreAdamW(
        param_groups,
        lr=LR,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        no_deprecation_warning=True,
    )


def get_optimizer(model, opt_name: str, config, world_size: int):
    m = actual_model(model)
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {
            "params": [
                p for n, p in m.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in m.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if opt_name == "AdamW":
        return torch.optim.AdamW(params, lr=LR)
    if opt_name == "8bit-AdamW":
        return bnb.optim.Adam8bit(params, lr=LR)
    if opt_name == "Adam-mini":
        if Adam_mini is None:
            raise ImportError("请先执行: pip install adam-mini")
        return build_adam_mini(model, config)
    if opt_name == "MDQ":
        return MDQAdamW(
            m.parameters(),
            lr=LR,
            layer_count=config.n_layer,
            batch_size=BATCH_SIZE_PER_GPU * world_size,
        )
    if opt_name == "GaLore":
        return build_galore_optimizer(model)
    raise ValueError(f"未知优化器: {opt_name}")


def ckpt_tag(opt_name: str) -> str:
    return opt_name.replace("-", "_")


def skip_dataloader_batches(data_iter, dataloader, n_batches: int):
    for _ in range(n_batches):
        try:
            next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            next(data_iter)
    return data_iter


def train_one_optimizer(
    opt_name: str,
    device: torch.device,
    local_rank: int,
    world_size: int,
    dataset,
    resume: bool,
) -> None:
    if local_rank == 0:
        print(f"\n{'=' * 20} 正在开始优化器: {opt_name} {'=' * 20}")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    config = GPT2Config.from_pretrained(MODEL_TYPE)
    model = GPT2LMHeadModel(config).to(device)
    model.gradient_checkpointing_enable()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = get_optimizer(model, opt_name, config, world_size)

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=local_rank, shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_PER_GPU,
        sampler=sampler,
        pin_memory=True,
    )
    sampler.set_epoch(0)
    data_iter = iter(dataloader)

    step = 0
    tokens_done = 0
    history: list[dict] = []
    ckpt_path = os.path.join(CKPT_DIR, f"{ckpt_tag(opt_name)}_latest.pt")

    if resume and os.path.exists(ckpt_path):
        if local_rank == 0:
            print(f"正在从 Checkpoint 恢复: {ckpt_path}")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint["step"])
        tokens_done = int(checkpoint.get("tokens", 0))
        history = checkpoint.get("history", [])
        if step > 0:
            if local_rank == 0:
                print(f"正在跳过已训练的 {step} 步")
            data_iter = skip_dataloader_batches(data_iter, dataloader, step * GRAD_ACCUM_STEPS)

    tps_per_step = BATCH_SIZE_PER_GPU * world_size * GRAD_ACCUM_STEPS * SEQ_LEN
    model.train()

    while step < MAX_STEPS:
        optimizer.zero_grad(set_to_none=True)

        for _ in range(GRAD_ACCUM_STEPS):
            try:
                batch = next(data_iter)
            except StopIteration:
                sampler.set_epoch(step // max(1, len(dataloader)))
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

        grad_norm = compute_grad_norm(model)
        optimizer.step()
        step += 1
        tokens_done += tps_per_step

        if step % LOG_INTERVAL == 0 and local_rank == 0:
            cur_loss = loss.item() * GRAD_ACCUM_STEPS
            quant_error = float("nan")
            if opt_name == "MDQ":
                quant_error = compute_mdq_quant_error(optimizer)

            history.append(
                {
                    "step": step,
                    "loss": cur_loss,
                    "grad_norm": grad_norm,
                    "quant_error": quant_error,
                }
            )
            print(
                f"[{opt_name}] step={step} | tokens={tokens_done / 1e6:.1f}M | "
                f"loss={cur_loss:.4f} | grad_norm={grad_norm:.4f} | "
                f"quant_error={quant_error:.6f}"
                if not math.isnan(quant_error)
                else f"[{opt_name}] step={step} | tokens={tokens_done / 1e6:.1f}M | "
                f"loss={cur_loss:.4f} | grad_norm={grad_norm:.4f}"
            )

        if step % CKPT_INTERVAL == 0 and local_rank == 0:
            torch.save(
                {
                    "step": step,
                    "tokens": tokens_done,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    "opt_name": opt_name,
                },
                ckpt_path,
            )

    if local_rank == 0:
        csv_path = os.path.join(RESULT_DIR, f"{ckpt_tag(opt_name)}_metrics.csv")
        pd.DataFrame(history).to_csv(csv_path, index=False)
        torch.save(
            {
                "step": step,
                "tokens": tokens_done,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "opt_name": opt_name,
            },
            os.path.join(CKPT_DIR, f"{ckpt_tag(opt_name)}_final.pt"),
        )
        print(f"优化器 {opt_name} 训练完成。指标已保存: {csv_path}")

    del model, optimizer
    torch.cuda.empty_cache()


def train():
    args = parse_args()
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        os.makedirs(CKPT_DIR, exist_ok=True)
        os.makedirs(RESULT_DIR, exist_ok=True)

    opts = ALL_OPTIMIZERS if args.optimizer == "all" else [args.optimizer]

    if local_rank == 0:
        print(f"Baseline GPT benchmark | GPUs={world_size} | opts={opts}")
        print(f"输出目录: {RESULT_DIR}")

    dataset = load_from_disk(DATA_PATH)
    dataset.set_format(type="torch", columns=["input_ids"])

    for opt_name in opts:
        dist.barrier()
        train_one_optimizer(
            opt_name, device, local_rank, world_size, dataset, args.resume
        )
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    train()

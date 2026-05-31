"""
GPT2-small + WikiText-103：MDQ score 平滑参数敏感性实验。

对照组：AdamW 32-bit
实验组：MDQ，单因子 OAT sweep（alpha / tau_scale / update_freq / score_bias / w_n / init_score）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import pandas as pd
import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, default_data_collator

from mdqbock_newhook import MDQAdamW

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(_EXP_DIR, "results")
RUN_DIR = os.path.join(RESULT_DIR, "runs")
CKPT_DIR = os.path.join(_EXP_DIR, "ckpts")

MODEL_TYPE = "gpt2"
MAX_STEPS = 400
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
SEQ_LEN = 512
LR = 5e-4
WEIGHT_DECAY = 0.01
LOG_INTERVAL = 20
EVAL_BATCHES = 200

DEFAULT_MDQ = {
    "alpha": 0.9,
    "tau_scale": 1.0,
    "update_freq": 20,
    "score_bias": 7.2,
    "w_n": 1.0,
    "init_score": 12.0,
}

SWEEP_GRID = {
    "alpha": [0.75, 0.85, 0.9, 0.95, 0.99],
    "tau_scale": [0.5, 0.75, 1.0, 1.25, 1.5],
    "update_freq": [5, 10, 20, 40, 80],
    "score_bias": [6.8, 7.0, 7.2, 7.4, 7.6],
    "w_n": [0.5, 0.75, 1.0, 1.25, 1.5],
    "init_score": [8, 10, 12, 14, 16],
}


def parse_args():
    p = argparse.ArgumentParser(description="MDQ score-smoothing parameter sensitivity")
    p.add_argument(
        "--mode",
        choices=["adamw32", "mdq", "stress"],
        default="mdq",
        help="adamw32=32-bit baseline; mdq=OAT sweep; stress=combined perturbation",
    )
    p.add_argument(
        "--sweep-param",
        choices=list(SWEEP_GRID.keys()),
        default="alpha",
        help="OAT 模式下要扰动的参数名",
    )
    p.add_argument("--sweep-value", type=float, default=None, help="参数取值")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--run-id", type=str, default=None, help="结果文件名标识")
    p.add_argument("--list-runs", action="store_true", help="打印全部 sweep 组合后退出")
    return p.parse_args()


def is_distributed() -> bool:
    return "LOCAL_RANK" in os.environ


def get_rank_info():
    if is_distributed():
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        local_rank = 0
        world_size = 1
        rank = 0
    return rank, local_rank, world_size


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


def score_stats(optimizer: MDQAdamW) -> tuple[float, float]:
    scores = optimizer.get_all_raw_scores()
    if not scores:
        return float("nan"), float("nan")
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / len(scores)
    return mean, math.sqrt(var)


def load_wikitext(tokenizer: GPT2Tokenizer, rank: int):
    cache_path = os.path.join(_EXP_DIR, "wikitext103_tokenized")
    if os.path.isdir(cache_path):
        if rank == 0:
            print(f"Loading cached tokenized dataset: {cache_path}")
        from datasets import load_from_disk

        ds = load_from_disk(cache_path)
        return ds["train"], ds["validation"]

    if rank == 0:
        print("Tokenizing WikiText-103 (first run only)...")
    raw = load_dataset("wikitext", "wikitext-103-raw-v1")
    raw = raw.filter(lambda x: len(x["text"].strip()) > 10)

    def tokenize_fn(examples):
        res = tokenizer(
            examples["text"],
            truncation=True,
            max_length=SEQ_LEN,
            padding="max_length",
        )
        res["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in ids]
            for ids in res["input_ids"]
        ]
        return res

    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        num_proc=4,
    )
    if rank == 0:
        tokenized.save_to_disk(cache_path)
        print(f"Saved tokenized dataset to {cache_path}")
    return tokenized["train"], tokenized["validation"]


def build_run_config(args) -> dict:
    if args.mode == "adamw32":
        return {
            "mode": "adamw32",
            "optimizer": "AdamW-32bit",
            "sweep_param": "none",
            "sweep_value": None,
            "mdq_params": None,
        }

    mdq_params = dict(DEFAULT_MDQ)
    if args.mode == "stress":
        mdq_params.update({"alpha": 0.75, "tau_scale": 1.5, "update_freq": 40})
        sweep_param = "stress_combo"
        sweep_value = None
    else:
        sweep_param = args.sweep_param
        if args.sweep_value is None:
            raise ValueError("--sweep-value is required for mode=mdq")
        if sweep_param == "update_freq" or sweep_param == "init_score":
            mdq_params[sweep_param] = int(args.sweep_value)
        else:
            mdq_params[sweep_param] = float(args.sweep_value)
        sweep_value = mdq_params[sweep_param]

    return {
        "mode": args.mode,
        "optimizer": "MDQ",
        "sweep_param": sweep_param,
        "sweep_value": sweep_value,
        "mdq_params": mdq_params,
    }


def make_run_id(run_cfg: dict, seed: int) -> str:
    if run_cfg["mode"] == "adamw32":
        return f"adamw32_seed{seed}"
    if run_cfg["mode"] == "stress":
        return f"mdq_stress_seed{seed}"
    param = run_cfg["sweep_param"]
    val = run_cfg["sweep_value"]
    if isinstance(val, float) and val == int(val):
        val_str = str(int(val))
    else:
        val_str = str(val).replace(".", "p")
    return f"mdq_{param}_{val_str}_seed{seed}"


def build_optimizer(model, run_cfg: dict, config: GPT2Config, batch_size: int):
    m = actual_model(model)
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {
            "params": [
                p for n, p in m.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p for n, p in m.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if run_cfg["mode"] == "adamw32":
        return torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), eps=1e-8)

    p = run_cfg["mdq_params"]
    return MDQAdamW(
        m.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
        layer_count=config.n_layer,
        batch_size=batch_size,
        alpha=p["alpha"],
        tau_scale=p["tau_scale"],
        update_freq=p["update_freq"],
        score_bias=p["score_bias"],
        w_n=p["w_n"],
        init_score=p["init_score"],
    )


@torch.no_grad()
def evaluate(
    model,
    eval_loader: DataLoader,
    device: torch.device,
    rank: int,
    world_size: int,
    max_batches: int,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        total_loss += outputs.loss.item()
        n_batches += 1
        if n_batches >= max_batches:
            break

    if is_distributed():
        t = torch.tensor([total_loss, float(n_batches)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, n_batches = t[0].item(), int(t[1].item())

    model.train()
    return total_loss / max(n_batches, 1)


def train(args):
    if args.list_runs:
        runs = ["adamw32"]
        for param, values in SWEEP_GRID.items():
            for v in values:
                runs.append(f"mdq {param}={v}")
        runs.append("mdq stress_combo")
        for line in runs:
            print(line)
        return

    run_cfg = build_run_config(args)
    run_id = args.run_id or make_run_id(run_cfg, args.seed)
    rank, local_rank, world_size = get_rank_info()

    if is_distributed():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        os.makedirs(RUN_DIR, exist_ok=True)
        os.makedirs(CKPT_DIR, exist_ok=True)
        print(f"Run ID: {run_id}")
        print(f"Config: {json.dumps(run_cfg, indent=2)}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_TYPE)
    tokenizer.pad_token = tokenizer.eos_token

    train_ds, eval_ds = load_wikitext(tokenizer, rank)
    if is_distributed():
        dist.barrier()

    config = GPT2Config.from_pretrained(MODEL_TYPE)
    model = GPT2LMHeadModel(config).to(device)
    model.gradient_checkpointing_enable()

    effective_batch = BATCH_SIZE * world_size
    optimizer = build_optimizer(model, run_cfg, config, effective_batch)

    if is_distributed():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=local_rank, shuffle=True
        )
        eval_sampler = DistributedSampler(
            eval_ds, num_replicas=world_size, rank=local_rank, shuffle=False
        )
    else:
        train_sampler = None
        eval_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=default_data_collator,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE,
        sampler=eval_sampler,
        shuffle=False,
        collate_fn=default_data_collator,
        pin_memory=torch.cuda.is_available(),
    )

    if train_sampler is not None:
        train_sampler.set_epoch(0)
    data_iter = iter(train_loader)

    step = 0
    history: list[dict] = []
    last_train_loss = float("nan")
    model.train()

    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)

        for _ in range(GRAD_ACCUM_STEPS):
            try:
                batch = next(data_iter)
            except StopIteration:
                if train_sampler is not None:
                    train_sampler.set_epoch(step // max(1, len(train_loader)))
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

        grad_norm = compute_grad_norm(model)
        optimizer.step()
        step += 1
        last_train_loss = loss.item() * GRAD_ACCUM_STEPS

        if step % LOG_INTERVAL == 0 and rank == 0:
            row = {
                "step": step,
                "train_loss": last_train_loss,
                "grad_norm": grad_norm,
            }
            if run_cfg["mode"] != "adamw32":
                row["quant_error"] = compute_mdq_quant_error(optimizer)
                s_mean, s_std = score_stats(optimizer)
                row["score_mean"] = s_mean
                row["score_std"] = s_std
                bits = optimizer.get_bit_distribution()
                for b in (4, 8, 16, 32):
                    row[f"bit_{b}_pct"] = bits.get(b, 0.0)
            history.append(row)
            print(
                f"[{run_id}] step={step}/{args.max_steps} "
                f"train_loss={last_train_loss:.4f} grad_norm={grad_norm:.4f}"
            )

    eval_loss = evaluate(model, eval_loader, device, rank, world_size, EVAL_BATCHES)
    eval_ppl = math.exp(min(eval_loss, 20.0))

    result = {
        "run_id": run_id,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "optimizer": run_cfg["optimizer"],
        "sweep_param": run_cfg["sweep_param"],
        "sweep_value": run_cfg["sweep_value"],
        "final_train_loss": last_train_loss,
        "eval_loss": eval_loss,
        "eval_ppl": eval_ppl,
        "mdq_params": run_cfg["mdq_params"],
    }

    if run_cfg["mode"] != "adamw32":
        result["quant_error"] = compute_mdq_quant_error(optimizer)
        s_mean, s_std = score_stats(optimizer)
        result["score_mean"] = s_mean
        result["score_std"] = s_std
        bits = optimizer.get_bit_distribution()
        for b in (4, 8, 16, 32):
            result[f"bit_{b}_pct"] = bits.get(b, 0.0)

    if rank == 0:
        json_path = os.path.join(RUN_DIR, f"{run_id}.json")
        csv_path = os.path.join(RUN_DIR, f"{run_id}_history.csv")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        if history:
            pd.DataFrame(history).to_csv(csv_path, index=False)
        print(f"Done. eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f}")
        print(f"Saved: {json_path}")

    if is_distributed():
        dist.barrier()

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    if is_distributed():
        dist.init_process_group(backend="nccl")
    try:
        train(args)
    finally:
        if is_distributed() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

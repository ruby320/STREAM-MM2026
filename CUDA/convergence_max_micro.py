#!/usr/bin/env python3
"""
固定 grad_accum + 各 optimizer 最大 micro 长训，直到滑动平均 loss 稳定低于目标值。

指标：端到端墙钟、tokens、吞吐、Peak、OptState、time/tokens-to-target-loss。

用法（单 optimizer，4 卡）:
  torchrun --standalone --nproc_per_node=4 convergence_max_micro.py \\
    --optimizer MDQAdamW-Simple-FusedIO --micro-batch 44

或读配置:
  torchrun --standalone --nproc_per_node=4 convergence_max_micro.py \\
    --optimizer AdamW-32bit --config convergence_max_micro_config.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque

import pandas as pd
import torch
from accelerate import Accelerator
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, set_seed

_CUDA_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUDA_DIR not in sys.path:
    sys.path.insert(0, _CUDA_DIR)

from Cfinetuning_timetest import (  # noqa: E402
    DATA_PATH,
    LR,
    MODEL_TYPE,
    SEQ_LEN,
    WEIGHT_DECAY,
    build_optimizer as build_cuda_optimizer,
    seed_everything,
    sync_cuda,
    _gather_peak_memory_gb,
    _gather_per_rank_gb,
    _optimizer_state_bytes,
    _unwrap_optimizer,
)

try:
    from adam_mini import Adam_mini
except ImportError:
    Adam_mini = None

try:
    from galore_torch import GaLoreAdamW
except ImportError:
    GaLoreAdamW = None

DEFAULT_CONFIG = os.path.join(_CUDA_DIR, "convergence_max_micro_config.json")
DEFAULT_SAVE_DIR = os.path.join(_CUDA_DIR, "results", "convergence_max_micro")

ALL_OPTIMIZERS = [
    "AdamW-32bit",
    "8bit-Adam-bnb",
    "MDQAdamW-Simple-FusedIO",
    "GaLore",
    "Adam-mini",
]

GALORE_RANK = int(os.environ.get("GALORE_RANK", "128"))
GALORE_UPDATE_PROJ_GAP = int(os.environ.get("GALORE_UPDATE_PROJ_GAP", "200"))
GALORE_SCALE = float(os.environ.get("GALORE_SCALE", "0.25"))
GALORE_PROJ_TYPE = os.environ.get("GALORE_PROJ_TYPE", "std")


def parse_args():
    p = argparse.ArgumentParser(description="Max-micro 长训至滑动平均 loss 达标")
    p.add_argument("--optimizer", choices=ALL_OPTIMIZERS, required=True)
    p.add_argument("--config", default=DEFAULT_CONFIG, help="含各 optimizer micro_batch 的 JSON")
    p.add_argument("--micro-batch", type=int, default=None, help="覆盖 config 中的 micro_batch")
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--target-loss", type=float, default=None)
    p.add_argument("--smooth-window", type=int, default=None)
    p.add_argument("--smooth-consecutive", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--stop-on-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="滑动平均达标后停止（默认 True）",
    )
    p.add_argument("--output-dir", default=DEFAULT_SAVE_DIR)
    return p.parse_args()


def load_merged_config(args) -> dict:
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    opt_cfg = cfg.get("optimizers", {}).get(args.optimizer, {})
    micro = args.micro_batch if args.micro_batch is not None else opt_cfg.get("micro_batch")
    if micro is None:
        raise ValueError(
            f"未指定 micro_batch：请在 {args.config} 的 optimizers.{args.optimizer} "
            "或命令行 --micro-batch 中设置"
        )
    return {
        "micro_batch": int(micro),
        "grad_accum": int(args.grad_accum if args.grad_accum is not None else cfg.get("grad_accum", 16)),
        "seq_len": int(cfg.get("seq_len", SEQ_LEN)),
        "lr": float(cfg.get("lr", LR)),
        "weight_decay": float(cfg.get("weight_decay", WEIGHT_DECAY)),
        "target_loss": float(
            args.target_loss if args.target_loss is not None else cfg.get("target_loss", 5.0)
        ),
        "smooth_window": int(
            args.smooth_window if args.smooth_window is not None else cfg.get("smooth_window", 5)
        ),
        "smooth_consecutive": int(
            args.smooth_consecutive
            if args.smooth_consecutive is not None
            else cfg.get("smooth_consecutive", 3)
        ),
        "max_steps": int(args.max_steps if args.max_steps is not None else cfg.get("max_steps", 1500)),
        "warmup_steps": int(
            args.warmup_steps if args.warmup_steps is not None else cfg.get("warmup_steps", 10)
        ),
        "log_interval": int(
            args.log_interval if args.log_interval is not None else cfg.get("log_interval", 10)
        ),
        "seed": int(args.seed if args.seed is not None else cfg.get("seed", 42)),
    }


def _unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def _mean_dict(values: dict) -> float:
    return float(sum(values.values()) / len(values)) if values else 0.0


def _opt_state_gb(accelerator, optimizer) -> dict:
    local_gb = _optimizer_state_bytes(_unwrap_optimizer(optimizer)) / (1024**3)
    return _gather_per_rank_gb(accelerator, local_gb)


def _build_adam_mini(model, config: GPT2Config, lr: float):
    if Adam_mini is None:
        raise ImportError("请先安装 adam-mini: pip install adam-mini")
    m = _unwrap_model(model)
    optimizer = Adam_mini(
        named_parameters=m.named_parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
        dim=config.n_embd,
        n_heads=config.n_head,
        n_kv_heads=None,
    )
    optimizer.wqk_names.add("c_attn")
    optimizer.wv_names.add("c_attn")
    optimizer.attn_proj_names.add("attn.c_proj")
    return optimizer


def _build_galore(model, lr: float):
    if GaLoreAdamW is None:
        raise ImportError("请先安装 galore-torch: pip install galore-torch")
    m = _unwrap_model(model)
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
            "weight_decay": WEIGHT_DECAY,
            "rank": GALORE_RANK,
            "update_proj_gap": GALORE_UPDATE_PROJ_GAP,
            "scale": GALORE_SCALE,
            "proj_type": GALORE_PROJ_TYPE,
        },
    ]
    return GaLoreAdamW(
        param_groups,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
        no_deprecation_warning=True,
    )


def build_optimizer(opt_name, model, config, lr, layer_count, global_batch_per_step):
    """global_batch_per_step = micro * num_gpus（与 timetest MDQ batch_size 对齐，不含 accum）。"""
    if opt_name in ("AdamW-32bit", "8bit-Adam-bnb", "MDQAdamW-Simple-FusedIO"):
        return build_cuda_optimizer(opt_name, model, lr, layer_count, global_batch_per_step)
    if opt_name == "Adam-mini":
        return _build_adam_mini(model, config, lr)
    if opt_name == "GaLore":
        return _build_galore(model, lr)
    raise ValueError(f"未知 optimizer: {opt_name}")


def _compute_grad_norm(model) -> float:
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().float().norm(2).item() ** 2
    return math.sqrt(total_sq)


def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class SmoothLossTracker:
    """滑动平均 loss + 连续达标计数。"""

    def __init__(self, window: int, consecutive: int, target: float):
        self.window = max(1, window)
        self.consecutive = max(1, consecutive)
        self.target = target
        self.history: deque[float] = deque(maxlen=self.window)
        self.streak = 0
        self.hit = False
        self.hit_step: int | None = None
        self.hit_tokens: int | None = None
        self.hit_wall_s: float | None = None
        self.hit_smooth_loss: float | None = None

    def update(self, loss: float, step: int, tokens: int, wall_s: float) -> float:
        self.history.append(loss)
        smooth = sum(self.history) / len(self.history)
        if len(self.history) >= self.window and smooth <= self.target:
            self.streak += 1
        else:
            self.streak = 0
        if not self.hit and self.streak >= self.consecutive:
            self.hit = True
            self.hit_step = step
            self.hit_tokens = tokens
            self.hit_wall_s = wall_s
            self.hit_smooth_loss = smooth
        return smooth


def train_one(args, cfg: dict) -> None:
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["grad_accum"],
        mixed_precision="bf16",
    )
    seed_everything(cfg["seed"])
    set_seed(cfg["seed"])

    micro = cfg["micro_batch"]
    accum = cfg["grad_accum"]
    num_gpus = accelerator.num_processes
    global_batch = micro * accum * num_gpus
    tokens_per_step = global_batch * cfg["seq_len"]
    global_batch_per_step = micro * num_gpus

    if accelerator.is_main_process:
        accelerator.print(f"\n{'=' * 60}")
        accelerator.print(f"Convergence max-micro | optimizer={args.optimizer}")
        accelerator.print(
            f"  micro={micro} accum={accum} GPUs={num_gpus} "
            f"global_batch={global_batch} tokens/step={tokens_per_step}"
        )
        accelerator.print(
            f"  target_loss={cfg['target_loss']} "
            f"smooth_window={cfg['smooth_window']} "
            f"smooth_consecutive={cfg['smooth_consecutive']}"
        )
        accelerator.print(
            f"  max_steps={cfg['max_steps']} warmup={cfg['warmup_steps']} "
            f"log_interval={cfg['log_interval']}"
        )

    with accelerator.main_process_first():
        dataset = load_from_disk(DATA_PATH)
    dataset.set_format(type="torch", columns=["input_ids"])

    config = GPT2Config.from_pretrained(MODEL_TYPE)
    model = GPT2LMHeadModel(config)
    model.gradient_checkpointing_enable()

    optimizer = build_optimizer(
        args.optimizer,
        model,
        config,
        cfg["lr"],
        config.n_layer,
        global_batch_per_step,
    )

    train_dl = DataLoader(
        dataset,
        batch_size=micro,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )
    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)

    tracker = SmoothLossTracker(
        cfg["smooth_window"],
        cfg["smooth_consecutive"],
        cfg["target_loss"],
    )

    opt_state_gb: dict | None = None
    log_rows: list[dict] = []
    global_step = 0
    tokens_done = 0
    train_start = time.perf_counter()
    last_log_wall = train_start
    stop_training = False

    _reset_peak()
    data_iter = iter(train_dl)

    while global_step < cfg["max_steps"] and not stop_training:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dl)
            batch = next(data_iter)

        with accelerator.accumulate(model):
            outputs = model(batch["input_ids"], labels=batch["input_ids"])
            accelerator.backward(outputs.loss)

        if not accelerator.sync_gradients:
            continue

        sync_cuda()
        grad_norm = _compute_grad_norm(model)
        optimizer.step()
        sync_cuda()
        optimizer.zero_grad()
        global_step += 1
        tokens_done += tokens_per_step

        if global_step == cfg["warmup_steps"]:
            _reset_peak()
            if accelerator.is_main_process:
                accelerator.print(f">>> warmup {cfg['warmup_steps']} steps 完成，开始统计 peak/计时")

        if opt_state_gb is None and global_step >= max(1, cfg["warmup_steps"]):
            opt_state_gb = _opt_state_gb(accelerator, optimizer)

        if global_step % cfg["log_interval"] == 0:
            # 与 baseline 一致：最后一个 micro 的 mean CE（不乘 accum）
            step_loss = float(outputs.loss.item())
            wall_now = time.perf_counter()
            wall_elapsed = wall_now - train_start
            dt = wall_now - last_log_wall
            last_log_wall = wall_now
            inst_tps = tokens_per_step / dt if dt > 0 else 0.0
            smooth = tracker.update(step_loss, global_step, tokens_done, wall_elapsed)

            log_rows.append(
                {
                    "Optimizer": args.optimizer,
                    "Step": global_step,
                    "Tokens": tokens_done,
                    "Wall_s": round(wall_elapsed, 3),
                    "Loss": step_loss,
                    "Loss_Smooth": round(smooth, 6),
                    "Smooth_Streak": tracker.streak,
                    "Target_Loss": cfg["target_loss"],
                    "Throughput_tok_s": round(inst_tps, 1),
                    "Grad_Norm": round(grad_norm, 6),
                    "Micro_Batch": micro,
                    "Grad_Accum": accum,
                    "Global_Batch": global_batch,
                }
            )

            if accelerator.is_main_process:
                hit_tag = " [HIT]" if tracker.hit and tracker.hit_step == global_step else ""
                accelerator.print(
                    f"step={global_step} tokens={tokens_done/1e6:.2f}M "
                    f"loss={step_loss:.4f} smooth={smooth:.4f} "
                    f"tps={inst_tps:.0f}{hit_tag}"
                )

            if tracker.hit and args.stop_on_target:
                stop_training = True

    total_wall_s = time.perf_counter() - train_start
    peak_gb = _gather_peak_memory_gb(accelerator)
    if opt_state_gb is None:
        opt_state_gb = _opt_state_gb(accelerator, optimizer)
    peak_mean = _mean_dict(peak_gb)
    opt_mean = _mean_dict(opt_state_gb)
    opt_share = (opt_mean / peak_mean * 100.0) if peak_mean > 0 else 0.0
    avg_tps = tokens_done / total_wall_s if total_wall_s > 0 else 0.0

    summary = {
        "optimizer": args.optimizer,
        "micro_batch": micro,
        "grad_accum": accum,
        "num_gpus": num_gpus,
        "global_batch": global_batch,
        "seq_len": cfg["seq_len"],
        "tokens_per_step": tokens_per_step,
        "target_loss": cfg["target_loss"],
        "smooth_window": cfg["smooth_window"],
        "smooth_consecutive": cfg["smooth_consecutive"],
        "hit_target": tracker.hit,
        "hit_step": tracker.hit_step,
        "hit_tokens": tracker.hit_tokens,
        "hit_wall_s": tracker.hit_wall_s,
        "hit_smooth_loss": tracker.hit_smooth_loss,
        "total_steps": global_step,
        "total_tokens": tokens_done,
        "total_wall_s": round(total_wall_s, 3),
        "avg_throughput_tok_s": round(avg_tps, 1),
        "peak_mean_gb": round(peak_mean, 2),
        "opt_state_mean_gb": round(opt_mean, 2),
        "opt_share_mean_pct": round(opt_share, 2),
        "peak_gpu_gb": {k: round(v, 2) for k, v in peak_gb.items()},
        "opt_state_gpu_gb": {k: round(v, 2) for k, v in opt_state_gb.items()},
        "stopped_reason": (
            "target_reached"
            if tracker.hit and args.stop_on_target
            else ("max_steps" if global_step >= cfg["max_steps"] else "unknown")
        ),
        "config_path": os.path.abspath(args.config),
    }

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        tag = args.optimizer.replace("/", "_").replace("-", "_")
        step_csv = os.path.join(args.output_dir, f"steps_{tag}.csv")
        summary_json = os.path.join(args.output_dir, f"summary_{tag}.json")
        pd.DataFrame(log_rows).to_csv(step_csv, index=False)
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        summary_row = {
            "Optimizer": args.optimizer,
            "Micro_Batch": micro,
            "Grad_Accum": accum,
            "Num_GPUs": num_gpus,
            "Global_Batch": global_batch,
            "Target_Loss": cfg["target_loss"],
            "Smooth_Window": cfg["smooth_window"],
            "Smooth_Consecutive": cfg["smooth_consecutive"],
            "Hit_Target": tracker.hit,
            "Hit_Step": tracker.hit_step,
            "Hit_Tokens": tracker.hit_tokens,
            "Hit_Wall_s": tracker.hit_wall_s,
            "Hit_Smooth_Loss": tracker.hit_smooth_loss,
            "Total_Steps": global_step,
            "Total_Tokens": tokens_done,
            "Total_Wall_s": round(total_wall_s, 3),
            "Avg_Throughput_tok_s": round(avg_tps, 1),
            "Peak_Mean_GB": round(peak_mean, 2),
            "OptState_Mean_GB": round(opt_mean, 2),
            "OptShare_Mean_Pct": round(opt_share, 2),
            "Stopped_Reason": summary["stopped_reason"],
        }
        summary_csv = os.path.join(args.output_dir, "convergence_summary_all.csv")
        if os.path.exists(summary_csv):
            old = pd.read_csv(summary_csv)
            old = old[old["Optimizer"] != args.optimizer]
            pd.concat([old, pd.DataFrame([summary_row])], ignore_index=True).to_csv(
                summary_csv, index=False
            )
        else:
            pd.DataFrame([summary_row]).to_csv(summary_csv, index=False)

        accelerator.print(f"\n>>> hit_target={tracker.hit}")
        if tracker.hit:
            accelerator.print(
                f">>> time-to-target: {tracker.hit_wall_s:.1f}s "
                f"step={tracker.hit_step} tokens={tracker.hit_tokens}"
            )
        else:
            accelerator.print(">>> 未在 max_steps 内达到滑动平均 loss 目标")
        accelerator.print(
            f">>> total: {total_wall_s:.1f}s tokens={tokens_done} avg_tps={avg_tps:.0f}"
        )
        accelerator.print(
            f">>> peak={peak_mean:.2f} GB opt={opt_mean:.2f} GB share={opt_share:.1f}%"
        )
        accelerator.print(f">>> 已写入 {step_csv}")
        accelerator.print(f">>> 已写入 {summary_json}")
        accelerator.print(f">>> 合并表 {summary_csv}")

    del model, optimizer
    accelerator.free_memory()


def main():
    args = parse_args()
    cfg = load_merged_config(args)
    train_one(args, cfg)


if __name__ == "__main__":
    main()

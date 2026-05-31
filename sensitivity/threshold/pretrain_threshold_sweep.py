"""
MDQ bit-allocation threshold scale sensitivity.

Uniform scaling: thresholds = scale × {6.8, 12, 24}.
Tasks: gpt2-medium (WikiText-103 LM), roberta-large (GLUE MNLI).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    default_data_collator,
    set_seed,
)

from mdqbock_newhook import BASE_THRESHOLDS, MDQAdamW, scaled_thresholds

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(_EXP_DIR, "results")
RUN_DIR = os.path.join(RESULT_DIR, "runs")

THRESHOLD_SCALES = [0.75, 0.85, 1.0, 1.15, 1.25]

DEFAULT_MDQ = {
    "alpha": 0.9,
    "tau_scale": 1.0,
    "update_freq": 20,
    "score_bias": 7.2,
    "w_n": 1.0,
    "init_score": 12.0,
}

TASK_CONFIGS = {
    "gpt2-medium": {
        "max_steps": 400,
        "batch_size": 4,
        "grad_accum_steps": 4,
        "lr": 5e-4,
        "weight_decay": 0.01,
        "eval_batches": 200,
        "log_interval": 20,
        "layer_count": 24,
        "data_cache": os.path.join(_EXP_DIR, "..", "parameter", "wikitext103_tokenized"),
    },
    "roberta-large": {
        "max_steps": 1000,
        "batch_size": 8,
        "grad_accum_steps": 1,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "eval_batches": 500,
        "log_interval": 50,
        "layer_count": 24,
        "update_freq": 50,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="MDQ threshold scale sensitivity")
    p.add_argument(
        "--task",
        choices=list(TASK_CONFIGS.keys()),
        default="gpt2-medium",
    )
    p.add_argument(
        "--mode",
        choices=["adamw32", "mdq"],
        default="mdq",
    )
    p.add_argument(
        "--threshold-scale",
        type=float,
        default=1.0,
        help="Uniform multiplier on base thresholds {6.8, 12, 24}",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--list-runs", action="store_true")
    return p.parse_args()


def is_distributed() -> bool:
    return "LOCAL_RANK" in os.environ


def get_rank_info():
    if is_distributed():
        local_rank = int(os.environ["LOCAL_RANK"])
        return dist.get_rank(), local_rank, dist.get_world_size()
    return 0, 0, 1


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


def compute_stability_metrics(history: list[dict]) -> dict:
    if not history:
        return {
            "train_loss_std_last100": float("nan"),
            "grad_norm_max": float("nan"),
            "grad_norm_std": float("nan"),
            "loss_spike_count": 0,
        }

    losses = [h["train_loss"] for h in history if "train_loss" in h]
    grad_norms = [h["grad_norm"] for h in history if "grad_norm" in h]
    tail = losses[-min(100, len(losses)) :]

    spike_count = 0
    for i in range(1, len(losses)):
        prev, cur = losses[i - 1], losses[i]
        if prev > 1e-8 and (cur - prev) / prev > 0.5:
            spike_count += 1

    return {
        "train_loss_std_last100": statistics.pstdev(tail) if len(tail) > 1 else 0.0,
        "grad_norm_max": max(grad_norms) if grad_norms else float("nan"),
        "grad_norm_std": statistics.pstdev(grad_norms) if len(grad_norms) > 1 else 0.0,
        "loss_spike_count": spike_count,
    }


def make_run_id(task: str, mode: str, scale: float | None, seed: int) -> str:
    if mode == "adamw32":
        return f"{task}_adamw32_seed{seed}"
    scale_str = str(scale).replace(".", "p")
    return f"{task}_mdq_scale{scale_str}_seed{seed}"


def build_mdq_params(task_cfg: dict, threshold_scale: float) -> dict:
    p = dict(DEFAULT_MDQ)
    if "update_freq" in task_cfg:
        p["update_freq"] = task_cfg["update_freq"]
    p["threshold_scale"] = threshold_scale
    t8, t16, t32 = scaled_thresholds(threshold_scale)
    p["thresholds"] = {"8": t8, "16": t16, "32": t32}
    return p


def build_optimizer(model, mode: str, task_cfg: dict, threshold_scale: float, batch_size: int):
    m = actual_model(model)
    if mode == "adamw32":
        no_decay = ["bias", "LayerNorm.weight", "layer_norm"]
        params = [
            {
                "params": [
                    p for n, p in m.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": task_cfg["weight_decay"],
            },
            {
                "params": [
                    p for n, p in m.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(
            params, lr=task_cfg["lr"], betas=(0.9, 0.999), eps=1e-8
        )

    p = build_mdq_params(task_cfg, threshold_scale)
    return MDQAdamW(
        m.parameters(),
        lr=task_cfg["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=task_cfg["weight_decay"],
        layer_count=task_cfg["layer_count"],
        batch_size=batch_size,
        alpha=p["alpha"],
        tau_scale=p["tau_scale"],
        update_freq=p["update_freq"],
        score_bias=p["score_bias"],
        w_n=p["w_n"],
        init_score=p["init_score"],
        threshold_scale=threshold_scale,
    )


def setup_gpt2(task_cfg: dict, device: torch.device, rank: int):
    from datasets import load_from_disk

    cache_path = os.path.abspath(task_cfg["data_cache"])
    if not os.path.isdir(cache_path):
        raise FileNotFoundError(
            f"Tokenized WikiText cache not found: {cache_path}\n"
            "Run parameter/pretrain_sensitivity.py once to build it."
        )
    if rank == 0:
        print(f"Loading WikiText cache: {cache_path}")
    ds = load_from_disk(cache_path)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    config = GPT2Config.from_pretrained("gpt2-medium")
    model = GPT2LMHeadModel(config).to(device)
    model.gradient_checkpointing_enable()

    train_loader = DataLoader(
        ds["train"],
        batch_size=task_cfg["batch_size"],
        shuffle=True,
        collate_fn=default_data_collator,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        ds["validation"],
        batch_size=task_cfg["batch_size"],
        shuffle=False,
        collate_fn=default_data_collator,
        pin_memory=torch.cuda.is_available(),
    )
    return model, train_loader, eval_loader, "lm"


def setup_roberta(task_cfg: dict, device: torch.device, rank: int, seed: int):
    from datasets import load_dataset

    set_seed(seed)
    model_name = "roberta-large"
    if rank == 0:
        print("Loading GLUE MNLI...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw = load_dataset("glue", "mnli")

    def tokenize_fn(ex):
        return tokenizer(
            ex["premise"], ex["hypothesis"], truncation=True, max_length=128
        )

    tokenized = raw.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(["premise", "hypothesis", "idx"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        tokenized["train"],
        batch_size=task_cfg["batch_size"],
        shuffle=True,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        tokenized["validation_matched"],
        batch_size=task_cfg["batch_size"],
        shuffle=False,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    ).to(device)
    return model, train_loader, eval_loader, "classification"


@torch.no_grad()
def evaluate_lm(model, eval_loader, device, max_batches: int) -> float:
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
    model.train()
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_classification(model, eval_loader, device, max_batches: int) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0
    for batch in eval_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(**batch)
        total_loss += outputs.loss.item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].numel()
        n_batches += 1
        if n_batches >= max_batches:
            break
    model.train()
    acc = correct / max(total, 1)
    return total_loss / max(n_batches, 1), acc


def train_step_lm(model, batch, device, grad_accum: int):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / grad_accum
    loss.backward()
    return loss.item() * grad_accum


def train_step_cls(model, batch, device):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(**batch)
        loss = outputs.loss
    loss.backward()
    return loss.item()


def train(args):
    if args.list_runs:
        for task in TASK_CONFIGS:
            print(f"{task} adamw32")
            for s in THRESHOLD_SCALES:
                print(f"{task} mdq scale={s}")
        return

    task_cfg = dict(TASK_CONFIGS[args.task])
    if args.max_steps is not None:
        task_cfg["max_steps"] = args.max_steps

    run_id = args.run_id or make_run_id(
        args.task, args.mode, args.threshold_scale if args.mode == "mdq" else None, args.seed
    )
    rank, local_rank, world_size = get_rank_info()

    if is_distributed():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        os.makedirs(RUN_DIR, exist_ok=True)
        print(f"Run ID: {run_id}")
        print(f"Task: {args.task}  Mode: {args.mode}  Scale: {args.threshold_scale}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.task == "gpt2-medium":
        model, train_loader, eval_loader, task_type = setup_gpt2(task_cfg, device, rank)
    else:
        model, train_loader, eval_loader, task_type = setup_roberta(
            task_cfg, device, rank, args.seed
        )

    grad_accum = task_cfg.get("grad_accum_steps", 1)
    effective_batch = task_cfg["batch_size"] * world_size * grad_accum
    threshold_scale = args.threshold_scale if args.mode == "mdq" else 1.0
    optimizer = build_optimizer(
        model, args.mode, task_cfg, threshold_scale, effective_batch
    )

    if is_distributed():
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    data_iter = iter(train_loader)
    step = 0
    history: list[dict] = []
    last_train_loss = float("nan")
    model.train()

    while step < task_cfg["max_steps"]:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            if task_type == "lm":
                accum_loss += train_step_lm(model, batch, device, grad_accum)
            else:
                accum_loss = train_step_cls(model, batch, device)

        grad_norm = compute_grad_norm(model)
        optimizer.step()
        step += 1
        last_train_loss = accum_loss if task_type == "lm" else accum_loss

        if step % task_cfg["log_interval"] == 0 and rank == 0:
            row = {"step": step, "train_loss": last_train_loss, "grad_norm": grad_norm}
            if args.mode == "mdq":
                row["quant_error"] = compute_mdq_quant_error(optimizer)
                row["avg_bit"] = optimizer.get_avg_bit()
                bits = optimizer.get_bit_distribution()
                for b in (4, 8, 16, 32):
                    row[f"bit_{b}_pct"] = bits.get(b, 0.0)
            history.append(row)
            extra = ""
            if args.mode == "mdq":
                extra = f" avg_bit={row['avg_bit']:.2f}"
            print(
                f"[{run_id}] step={step}/{task_cfg['max_steps']} "
                f"loss={last_train_loss:.4f} grad_norm={grad_norm:.4f}{extra}"
            )

    if task_type == "lm":
        eval_loss = evaluate_lm(model, eval_loader, device, task_cfg["eval_batches"])
        eval_ppl = math.exp(min(eval_loss, 20.0))
        eval_acc = None
    else:
        eval_loss, eval_acc = evaluate_classification(
            model, eval_loader, device, task_cfg["eval_batches"]
        )
        eval_ppl = None

    stability = compute_stability_metrics(history)
    mdq_params = build_mdq_params(task_cfg, threshold_scale) if args.mode == "mdq" else None

    result = {
        "run_id": run_id,
        "task": args.task,
        "seed": args.seed,
        "max_steps": task_cfg["max_steps"],
        "optimizer": "AdamW-32bit" if args.mode == "adamw32" else "MDQ",
        "mode": args.mode,
        "threshold_scale": threshold_scale if args.mode == "mdq" else None,
        "base_thresholds": list(BASE_THRESHOLDS),
        "final_train_loss": last_train_loss,
        "eval_loss": eval_loss,
        "eval_ppl": eval_ppl,
        "eval_accuracy": eval_acc,
        "mdq_params": mdq_params,
        **stability,
    }

    if args.mode == "mdq":
        result["quant_error"] = compute_mdq_quant_error(optimizer)
        result["avg_bit"] = optimizer.get_avg_bit()
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
        msg = f"eval_loss={eval_loss:.4f}"
        if eval_ppl is not None:
            msg += f" eval_ppl={eval_ppl:.2f}"
        if eval_acc is not None:
            msg += f" eval_acc={eval_acc:.4f}"
        print(f"Done. {msg}")
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

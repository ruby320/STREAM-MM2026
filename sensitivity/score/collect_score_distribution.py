"""
Collect per-parameter MDQ score distributions across architectures.

Models: gpt2-medium (LM), vit-base (vision), roberta-large (GLUE MNLI).
Outputs JSON snapshots under results/<model>/ for plot_score_distributions.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from mdqbock_newhook import MDQAdamW

_EXP_DIR = Path(__file__).resolve().parent
RESULT_DIR = _EXP_DIR / "results"
CONFIG_PATH = _EXP_DIR / "configs" / "models.yaml"

THRESHOLDS = [6.8, 12.0, 24.0]

DEFAULT_MDQ = {
    "alpha": 0.9,
    "tau_scale": 1.0,
    "update_freq": 20,
    "score_bias": 7.2,
    "w_n": 1.0,
    "init_score": 12.0,
}


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def actual_model(model):
    return model.module if hasattr(model, "module") else model


def score_to_bit(score: float) -> int:
    if score >= 24:
        return 32
    if score >= 12:
        return 16
    if score >= 6.8:
        return 8
    return 4


def collect_score_snapshot(
    optimizer: MDQAdamW, model, step: int, model_name: str
) -> dict:
    m = actual_model(model)
    scores: list[float] = []
    bits: list[int] = []
    names: list[str] = []

    for name, p in m.named_parameters():
        state = optimizer.state.get(p)
        if not state or "last_score" not in state:
            continue
        s = float(state["last_score"].item())
        b = int(state.get("current_bit", score_to_bit(s)))
        scores.append(s)
        bits.append(b)
        names.append(name)

    arr = np.array(scores, dtype=np.float64)
    bit_counts = {4: 0, 8: 0, 16: 0, 32: 0}
    for b in bits:
        bit_counts[b] = bit_counts.get(b, 0) + 1
    n = max(len(bits), 1)
    bit_pct = {str(k): v / n * 100 for k, v in bit_counts.items()}

    return {
        "model": model_name,
        "step": step,
        "n_params": len(scores),
        "thresholds": THRESHOLDS,
        "scores": scores,
        "bits": bits,
        "param_names": names,
        "score_mean": float(arr.mean()) if len(arr) else float("nan"),
        "score_std": float(arr.std()) if len(arr) else float("nan"),
        "score_quantiles": {
            "p5": float(np.percentile(arr, 5)) if len(arr) else float("nan"),
            "p25": float(np.percentile(arr, 25)) if len(arr) else float("nan"),
            "p50": float(np.percentile(arr, 50)) if len(arr) else float("nan"),
            "p75": float(np.percentile(arr, 75)) if len(arr) else float("nan"),
            "p95": float(np.percentile(arr, 95)) if len(arr) else float("nan"),
        },
        "bit_pct": bit_pct,
    }


def build_mdq_optimizer(model, layer_count: int, batch_size: int, mdq_overrides: dict | None = None):
    m = actual_model(model)
    params = [p for p in m.parameters() if p.requires_grad]
    cfg = dict(DEFAULT_MDQ)
    if mdq_overrides:
        cfg.update(mdq_overrides)
    return MDQAdamW(
        params,
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 0.01),
        layer_count=layer_count,
        batch_size=batch_size,
        alpha=cfg["alpha"],
        tau_scale=cfg["tau_scale"],
        update_freq=cfg["update_freq"],
        score_bias=cfg["score_bias"],
        w_n=cfg["w_n"],
        init_score=cfg["init_score"],
    )


def should_snapshot(step: int, snapshot_steps: list[int], update_freq: int) -> bool:
    if step not in snapshot_steps:
        return False
    return (step % update_freq == 0) or (step < 5)


def save_snapshot(snapshot: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"scores_step{snapshot['step']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"  Saved snapshot: {path}")


def train_gpt2_medium(cfg: dict, seed: int, max_steps: int | None):
    from datasets import load_from_disk
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, default_data_collator

    model_cfg = cfg["models"]["gpt2-medium"]
    max_steps = max_steps or model_cfg["max_steps"]
    snapshot_steps = set(model_cfg["snapshot_steps"])
    cache_path = (_EXP_DIR / model_cfg["data_cache"]).resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = GPT2Tokenizer.from_pretrained(model_cfg["hf_name"])
    tokenizer.pad_token = tokenizer.eos_token
    train_ds = load_from_disk(str(cache_path))["train"]

    gpt_cfg = GPT2Config.from_pretrained(model_cfg["hf_name"])
    model = GPT2LMHeadModel(gpt_cfg).to(device)
    model.gradient_checkpointing_enable()
    model.train()

    batch_size = model_cfg["batch_size"]
    grad_accum = model_cfg["grad_accum_steps"]
    mdq_cfg = {**DEFAULT_MDQ, "lr": model_cfg["lr"], "weight_decay": model_cfg["weight_decay"]}
    optimizer = build_mdq_optimizer(
        model, model_cfg["layer_count"], batch_size * grad_accum, mdq_cfg
    )
    update_freq = mdq_cfg["update_freq"]

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
        pin_memory=torch.cuda.is_available(),
    )
    data_iter = iter(loader)
    out_dir = RESULT_DIR / "gpt2-medium"
    snapshots: list[dict] = []

    step = 0
    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                (outputs.loss / grad_accum).backward()

        optimizer.step()
        step += 1

        if should_snapshot(step, snapshot_steps, update_freq):
            snap = collect_score_snapshot(optimizer, model, step, "gpt2-medium")
            snapshots.append(snap)
            save_snapshot(snap, out_dir)
            print(
                f"[gpt2-medium] step={step} mean={snap['score_mean']:.2f} "
                f"bit_pct={snap['bit_pct']}"
            )

    final = snapshots[-1] if snapshots else {}
    summary = {
        "model": "gpt2-medium",
        "seed": seed,
        "max_steps": max_steps,
        "thresholds": THRESHOLDS,
        "mdq_params": mdq_cfg,
        "final_snapshot": {
            k: v for k, v in final.items() if k not in ("scores", "bits", "param_names")
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Done gpt2-medium -> {out_dir}")


class SimpleImageDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        if not self.samples:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.samples[idx]).convert("RGB")
        except OSError:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, 0


def train_vit_base(cfg: dict, seed: int, max_steps: int | None):
    import timm

    model_cfg = cfg["models"]["vit-base"]
    max_steps = max_steps or model_cfg["max_steps"]
    snapshot_steps = set(model_cfg["snapshot_steps"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = SimpleImageDataset(model_cfg["data_path"], transform=transform)
    batch_size = model_cfg["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    model = timm.create_model(
        model_cfg["timm_name"], pretrained=False, num_classes=768
    ).to(device)
    model.train()
    criterion = torch.nn.MSELoss()

    mdq_cfg = {**DEFAULT_MDQ, "lr": model_cfg["lr"], "weight_decay": model_cfg["weight_decay"]}
    optimizer = build_mdq_optimizer(
        model, model_cfg["layer_count"], batch_size, mdq_cfg
    )
    update_freq = mdq_cfg["update_freq"]

    data_iter = iter(loader)
    out_dir = RESULT_DIR / "vit-base"
    snapshots: list[dict] = []
    step = 0

    while step < max_steps:
        try:
            imgs, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            imgs, _ = next(data_iter)

        imgs = imgs.to(device)
        with torch.no_grad():
            targets = torch.nn.functional.interpolate(imgs, size=(16, 16))
            targets = targets.permute(0, 2, 3, 1).flatten(1)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(imgs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        step += 1

        if should_snapshot(step, snapshot_steps, update_freq):
            snap = collect_score_snapshot(optimizer, model, step, "vit-base")
            snapshots.append(snap)
            save_snapshot(snap, out_dir)
            print(
                f"[vit-base] step={step} mean={snap['score_mean']:.2f} "
                f"bit_pct={snap['bit_pct']}"
            )

    summary = {
        "model": "vit-base",
        "seed": seed,
        "max_steps": max_steps,
        "thresholds": THRESHOLDS,
        "mdq_params": mdq_cfg,
        "final_snapshot": {
            k: v
            for k, v in (snapshots[-1] if snapshots else {}).items()
            if k not in ("scores", "bits", "param_names")
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Done vit-base -> {out_dir}")


def train_roberta_large(cfg: dict, seed: int, max_steps: int | None):
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        set_seed,
    )

    model_cfg = cfg["models"]["roberta-large"]
    max_steps = max_steps or model_cfg["max_steps"]
    snapshot_steps = set(model_cfg["snapshot_steps"])
    update_freq = model_cfg.get("update_freq", DEFAULT_MDQ["update_freq"])

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = model_cfg["hf_name"]
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
    batch_size = model_cfg["batch_size"]
    loader = DataLoader(
        tokenized["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    ).to(device)
    model.train()

    mdq_cfg = {
        **DEFAULT_MDQ,
        "lr": model_cfg["lr"],
        "weight_decay": model_cfg["weight_decay"],
        "update_freq": update_freq,
    }
    optimizer = build_mdq_optimizer(
        model, model_cfg["layer_count"], batch_size, mdq_cfg
    )

    data_iter = iter(loader)
    out_dir = RESULT_DIR / "roberta-large"
    snapshots: list[dict] = []
    step = 0

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss
        loss.backward()
        optimizer.step()
        step += 1

        if should_snapshot(step, snapshot_steps, update_freq):
            snap = collect_score_snapshot(optimizer, model, step, "roberta-large")
            snapshots.append(snap)
            save_snapshot(snap, out_dir)
            print(
                f"[roberta-large] step={step} mean={snap['score_mean']:.2f} "
                f"bit_pct={snap['bit_pct']}"
            )

    summary = {
        "model": "roberta-large",
        "seed": seed,
        "max_steps": max_steps,
        "thresholds": THRESHOLDS,
        "mdq_params": mdq_cfg,
        "final_snapshot": {
            k: v
            for k, v in (snapshots[-1] if snapshots else {}).items()
            if k not in ("scores", "bits", "param_names")
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Done roberta-large -> {out_dir}")


TRAINERS = {
    "gpt2-medium": train_gpt2_medium,
    "vit-base": train_vit_base,
    "roberta-large": train_roberta_large,
}


def parse_args():
    p = argparse.ArgumentParser(description="Collect MDQ score distributions")
    p.add_argument(
        "--model",
        choices=list(TRAINERS.keys()) + ["all"],
        default="all",
        help="Which model to run",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=None, help="Override config max_steps")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    models = list(TRAINERS.keys()) if args.model == "all" else [args.model]
    for name in models:
        print(f"\n=== Collecting scores: {name} ===")
        TRAINERS[name](cfg, args.seed, args.max_steps)


if __name__ == "__main__":
    main()

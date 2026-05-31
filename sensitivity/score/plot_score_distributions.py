"""
Plot MDQ score distributions with fixed bit-allocation thresholds.

Reads results/<model>/scores_step*.json and produces:
  - results/score_distribution_combined.png  (1x3 panel for rebuttal)
  - results/<model>/score_dist.png           (per-model figure)
  - results/score_summary.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_EXP_DIR = Path(__file__).resolve().parent
RESULT_DIR = _EXP_DIR / "results"

MODELS = ["gpt2-medium", "vit-base", "roberta-large"]
MODEL_LABELS = {
    "gpt2-medium": "GPT-2 Medium\n(LM Pretrain)",
    "vit-base": "ViT-Base\n(Vision Pretrain)",
    "roberta-large": "RoBERTa-Large\n(MNLI Finetune)",
}
THRESHOLDS = [6.8, 12.0, 24.0]
BIT_COLORS = {4: "#4393c3", 8: "#92c5de", 16: "#f4a582", 32: "#d6604d"}
THRESHOLD_COLORS = ["#666666", "#888888", "#aaaaaa"]


plt.rcParams.update({
    "pdf.fonttype": 42,
    "font.family": "serif",
    "font.size": 10,
})


def find_latest_snapshot(model_dir: Path) -> Path | None:
    files = sorted(model_dir.glob("scores_step*.json"))
    return files[-1] if files else None


def load_snapshot(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def bit_pct_text(bit_pct: dict) -> str:
    parts = []
    for b in (4, 8, 16, 32):
        key = str(b)
        if key in bit_pct:
            parts.append(f"{b}b: {bit_pct[key]:.1f}%")
    return " | ".join(parts)


def plot_single_ax(ax, scores: list[float], bit_pct: dict, title: str):
    arr = np.array(scores, dtype=np.float64)
    if len(arr) == 0:
        ax.text(0.5, 0.5, "No score data", ha="center", va="center")
        ax.set_title(title)
        return

    x_min = min(arr.min(), -2) - 1
    x_max = max(arr.max(), THRESHOLDS[-1] + 2) + 1

    zones = [
        (x_min, THRESHOLDS[0], BIT_COLORS[4], "4-bit"),
        (THRESHOLDS[0], THRESHOLDS[1], BIT_COLORS[8], "8-bit"),
        (THRESHOLDS[1], THRESHOLDS[2], BIT_COLORS[16], "16-bit"),
        (THRESHOLDS[2], x_max, BIT_COLORS[32], "32-bit"),
    ]
    for lo, hi, color, _ in zones:
        ax.axvspan(lo, hi, alpha=0.12, color=color, linewidth=0)

    bins = np.linspace(x_min, x_max, 50)
    ax.hist(
        arr,
        bins=bins,
        density=True,
        alpha=0.55,
        color="#2166ac",
        edgecolor="white",
        linewidth=0.4,
        label="Score histogram",
    )

    try:
        from scipy.stats import gaussian_kde

        kde_x = np.linspace(x_min, x_max, 300)
        kde = gaussian_kde(arr, bw_method=0.35)
        ax.plot(kde_x, kde(kde_x), color="#b2182b", lw=2, label="KDE")
    except Exception:
        pass

    for thr, color in zip(THRESHOLDS, THRESHOLD_COLORS):
        ax.axvline(thr, color=color, ls="--", lw=1.5)
        ax.text(
            thr,
            ax.get_ylim()[1] * 0.95,
            f"{thr:g}",
            ha="center",
            va="top",
            fontsize=8,
            color=color,
            rotation=90,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("MDQ Sensitivity Score")
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=11)
    ax.text(
        0.02,
        0.98,
        bit_pct_text(bit_pct),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )


def build_summary_row(model: str, snap: dict, path: Path) -> dict:
    q = snap.get("score_quantiles", {})
    bp = snap.get("bit_pct", {})
    return {
        "model": model,
        "step": snap.get("step"),
        "snapshot_file": path.name,
        "n_params": snap.get("n_params"),
        "score_mean": snap.get("score_mean"),
        "score_std": snap.get("score_std"),
        "p5": q.get("p5"),
        "p25": q.get("p25"),
        "p50": q.get("p50"),
        "p75": q.get("p75"),
        "p95": q.get("p95"),
        "bit_4_pct": bp.get("4"),
        "bit_8_pct": bp.get("8"),
        "bit_16_pct": bp.get("16"),
        "bit_32_pct": bp.get("32"),
    }


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    snapshots: dict[str, dict] = {}
    summary_rows: list[dict] = []

    for model in MODELS:
        model_dir = RESULT_DIR / model
        path = find_latest_snapshot(model_dir)
        if path is None:
            print(f"Warning: no snapshots for {model} (run collect_score_distribution.py first)")
            continue
        snap = load_snapshot(path)
        snapshots[model] = snap
        summary_rows.append(build_summary_row(model, snap, path))

        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=200)
        plot_single_ax(
            ax,
            snap["scores"],
            snap["bit_pct"],
            f"{MODEL_LABELS[model]} (step {snap['step']})",
        )
        fig.tight_layout()
        out = model_dir / "score_dist.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

    if not snapshots:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=200, sharey=True)
    for ax, model in zip(axes, MODELS):
        if model not in snapshots:
            ax.set_visible(False)
            continue
        snap = snapshots[model]
        plot_single_ax(
            ax,
            snap["scores"],
            snap["bit_pct"],
            MODEL_LABELS[model],
        )

    fig.suptitle(
        "MDQ Score Distribution Across Architectures (fixed thresholds: 6.8 / 12 / 24)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    combined = RESULT_DIR / "score_distribution_combined.png"
    fig.savefig(combined, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {combined}")

    df = pd.DataFrame(summary_rows)
    csv_path = RESULT_DIR / "score_summary.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()

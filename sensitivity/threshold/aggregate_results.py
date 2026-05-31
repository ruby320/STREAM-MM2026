"""
Aggregate threshold scale sweep results and generate rebuttal figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_EXP_DIR = Path(__file__).resolve().parent
RUN_DIR = _EXP_DIR / "results" / "runs"
FIG_DIR = _EXP_DIR / "results" / "figures"
OUT_CSV = _EXP_DIR / "results" / "threshold_summary.csv"
OUT_MD = _EXP_DIR / "results" / "threshold_summary.md"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "font.family": "serif",
    "font.size": 10,
})


def load_run_results() -> list[dict]:
    rows = []
    if not RUN_DIR.is_dir():
        return rows
    for path in sorted(RUN_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        mdq = data.get("mdq_params") or {}
        thresholds = mdq.get("thresholds") or {}
        rows.append({
            "run_id": data.get("run_id", path.stem),
            "task": data.get("task", ""),
            "optimizer": data.get("optimizer", ""),
            "threshold_scale": data.get("threshold_scale"),
            "seed": data.get("seed"),
            "max_steps": data.get("max_steps"),
            "eval_loss": data.get("eval_loss"),
            "eval_ppl": data.get("eval_ppl"),
            "eval_accuracy": data.get("eval_accuracy"),
            "avg_bit": data.get("avg_bit"),
            "quant_error": data.get("quant_error"),
            "bit_4_pct": data.get("bit_4_pct"),
            "bit_8_pct": data.get("bit_8_pct"),
            "bit_16_pct": data.get("bit_16_pct"),
            "bit_32_pct": data.get("bit_32_pct"),
            "train_loss_std_last100": data.get("train_loss_std_last100"),
            "grad_norm_max": data.get("grad_norm_max"),
            "grad_norm_std": data.get("grad_norm_std"),
            "loss_spike_count": data.get("loss_spike_count"),
            "thresh_8": thresholds.get("8"),
            "thresh_16": thresholds.get("16"),
            "thresh_32": thresholds.get("32"),
        })
    return rows


def add_delta_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for task in df["task"].unique():
        mask = df["task"] == task
        sub = df[mask]

        adamw = sub[sub["optimizer"] == "AdamW-32bit"]
        if len(adamw):
            if task == "gpt2-medium":
                ref_col = "eval_ppl"
                ref_val = adamw[ref_col].iloc[0]
                df.loc[mask, "delta_vs_adamw_pct"] = (
                    (df.loc[mask, ref_col] - ref_val) / ref_val * 100.0
                )
            else:
                ref_col = "eval_accuracy"
                ref_val = adamw[ref_col].iloc[0]
                df.loc[mask, "delta_vs_adamw_pct"] = (
                    (df.loc[mask, ref_col] - ref_val) / ref_val * 100.0
                )

        default = sub[
            (sub["optimizer"] == "MDQ") & (sub["threshold_scale"].astype(float) == 1.0)
        ]
        if len(default):
            if task == "gpt2-medium":
                ref_val = default["eval_ppl"].iloc[0]
                mdq_mask = mask & (df["optimizer"] == "MDQ")
                df.loc[mdq_mask, "delta_ppl_vs_default_pct"] = (
                    (df.loc[mdq_mask, "eval_ppl"] - ref_val) / ref_val * 100.0
                )
            else:
                ref_val = default["eval_accuracy"].iloc[0]
                mdq_mask = mask & (df["optimizer"] == "MDQ")
                df.loc[mdq_mask, "delta_acc_vs_default_pct"] = (
                    (df.loc[mdq_mask, "eval_accuracy"] - ref_val) / ref_val * 100.0
                )

    return df


def to_markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "run_id", "task", "threshold_scale", "eval_ppl", "eval_accuracy",
        "avg_bit", "delta_ppl_vs_default_pct", "delta_acc_vs_default_pct",
        "bit_4_pct", "bit_8_pct", "bit_16_pct", "bit_32_pct",
        "train_loss_std_last100", "loss_spike_count",
    ]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].copy()
    for c in sub.columns:
        if sub[c].dtype == float:
            sub[c] = sub[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def plot_task_figures(df: pd.DataFrame, task: str):
    sub = df[(df["task"] == task) & (df["optimizer"] == "MDQ")].copy()
    if sub.empty:
        print(f"No MDQ runs for {task}, skip plots")
        return

    sub = sub.sort_values("threshold_scale")
    scales = sub["threshold_scale"].astype(float).values
    is_lm = task == "gpt2-medium"

    # --- Pareto: avg_bit vs performance ---
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    perf = sub["eval_ppl"].values if is_lm else sub["eval_accuracy"].values
    avg_bits = sub["avg_bit"].values
    ax.scatter(avg_bits, perf, s=80, c=scales, cmap="viridis", zorder=3)
    for x, y, s in zip(avg_bits, perf, scales):
        label = f"s={s:g}" + (" *" if abs(s - 1.0) < 1e-9 else "")
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Average Bit Width")
    ax.set_ylabel("Eval PPL" if is_lm else "Eval Accuracy")
    ax.set_title(f"{task}: Performance vs Compression")
    if not is_lm:
        ax.invert_yaxis()
    fig.tight_layout()
    path = FIG_DIR / f"pareto_{task}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # --- Delta performance vs scale ---
    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=200)
    if is_lm and "delta_ppl_vs_default_pct" in sub.columns:
        y = sub["delta_ppl_vs_default_pct"].values
        ylabel = "ΔPPL vs default (%)"
    elif "delta_acc_vs_default_pct" in sub.columns:
        y = sub["delta_acc_vs_default_pct"].values
        ylabel = "ΔAccuracy vs default (%)"
    else:
        y = np.zeros(len(scales))
        ylabel = "Δ metric (%)"
    ax.axhline(0, color="gray", lw=0.8)
    ax.axhspan(-5, 5, alpha=0.12, color="green", label="±5% band")
    ax.plot(scales, y, "o-", color="#2166ac", lw=2, markersize=8)
    ax.axvline(1.0, color="#b2182b", ls="--", lw=1.2, label="default s=1.0")
    ax.set_xlabel("Threshold Scale s")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{task}: Threshold Perturbation Sensitivity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / f"delta_vs_scale_{task}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # --- Bit stack + stability ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=200)
    bit_cols = ["bit_4_pct", "bit_8_pct", "bit_16_pct", "bit_32_pct"]
    bit_labels = ["4-bit", "8-bit", "16-bit", "32-bit"]
    colors = ["#4393c3", "#92c5de", "#f4a582", "#d6604d"]
    bottom = np.zeros(len(scales))
    for col, label, color in zip(bit_cols, bit_labels, colors):
        vals = sub[col].fillna(0).values
        axes[0].bar(scales.astype(str), vals, bottom=bottom, label=label, color=color)
        bottom += vals
    axes[0].set_xlabel("Threshold Scale s")
    axes[0].set_ylabel("Parameter Fraction (%)")
    axes[0].set_title("Bit Allocation")
    axes[0].legend(fontsize=7, loc="upper right")

    x_pos = np.arange(len(scales))
    axes[1].bar(x_pos - 0.2, sub["avg_bit"].values, width=0.4, label="avg_bit", color="#2166ac")
    ax2 = axes[1].twinx()
    ax2.plot(
        x_pos, sub["train_loss_std_last100"].values, "s--",
        color="#b2182b", label="loss_std(last100)",
    )
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f"{s:g}" for s in scales])
    axes[1].set_xlabel("Threshold Scale s")
    axes[1].set_ylabel("Avg Bit Width", color="#2166ac")
    ax2.set_ylabel("Train Loss Std", color="#b2182b")
    axes[1].set_title("Compression & Stability")
    fig.tight_layout()
    path = FIG_DIR / f"bitstack_stability_{task}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    rows = load_run_results()
    if not rows:
        print(f"No results in {RUN_DIR}. Run: bash run_sweep.sh")
        return

    df = pd.DataFrame(rows)
    df = add_delta_columns(df)
    df = df.sort_values(["task", "optimizer", "threshold_scale"], na_position="first")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.6f")

    md = to_markdown_table(df)
    OUT_MD.write_text(
        "# MDQ Threshold Scale Sensitivity\n\n" + md + "\n",
        encoding="utf-8",
    )

    print(f"Loaded {len(df)} runs")
    print(f"Summary CSV: {OUT_CSV}")
    print(f"Summary MD:  {OUT_MD}")

    for task in df["task"].unique():
        plot_task_figures(df, task)

    mdq = df[df["optimizer"] == "MDQ"]
    if "delta_ppl_vs_default_pct" in mdq.columns:
        valid = mdq["delta_ppl_vs_default_pct"].dropna()
        if len(valid):
            print(f"Max |ΔPPL vs default| (GPT2): {valid.abs().max():.2f}%")


if __name__ == "__main__":
    main()

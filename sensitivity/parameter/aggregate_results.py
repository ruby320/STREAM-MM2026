"""
汇总 sensitivity 实验结果，生成 rebuttal 用表格 CSV。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

_EXP_DIR = Path(__file__).resolve().parent
RUN_DIR = _EXP_DIR / "results" / "runs"
OUT_CSV = _EXP_DIR / "results" / "sensitivity_summary.csv"
OUT_MD = _EXP_DIR / "results" / "sensitivity_summary.md"


def load_run_results() -> list[dict]:
    rows = []
    if not RUN_DIR.is_dir():
        return rows
    for path in sorted(RUN_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        row = {
            "run_id": data.get("run_id", path.stem),
            "optimizer": data.get("optimizer", ""),
            "sweep_param": data.get("sweep_param", ""),
            "sweep_value": data.get("sweep_value", ""),
            "seed": data.get("seed", ""),
            "max_steps": data.get("max_steps", ""),
            "final_train_loss": data.get("final_train_loss"),
            "eval_loss": data.get("eval_loss"),
            "eval_ppl": data.get("eval_ppl"),
            "quant_error": data.get("quant_error"),
            "score_mean": data.get("score_mean"),
            "score_std": data.get("score_std"),
            "bit_4_pct": data.get("bit_4_pct"),
            "bit_8_pct": data.get("bit_8_pct"),
            "bit_16_pct": data.get("bit_16_pct"),
            "bit_32_pct": data.get("bit_32_pct"),
        }
        mdq = data.get("mdq_params") or {}
        for k in ("alpha", "tau_scale", "update_freq", "score_bias", "w_n", "init_score"):
            row[f"mdq_{k}"] = mdq.get(k)
        rows.append(row)
    return rows


def add_delta_columns(df: pd.DataFrame) -> pd.DataFrame:
    adamw = df[df["optimizer"] == "AdamW-32bit"]
    adamw_ppl = adamw["eval_ppl"].iloc[0] if len(adamw) else float("nan")

    default_mask = (
        (df["optimizer"] == "MDQ")
        & (df["sweep_param"].isin(SWEEP_PARAM_NAMES))
        & (df.apply(_is_default_sweep_row, axis=1))
    )
    default_rows = df[default_mask]
    default_ppl = default_rows["eval_ppl"].iloc[0] if len(default_rows) else float("nan")

    df = df.copy()
    df["delta_ppl_vs_adamw_pct"] = (df["eval_ppl"] - adamw_ppl) / adamw_ppl * 100.0
    df["delta_ppl_vs_default_pct"] = (df["eval_ppl"] - default_ppl) / default_ppl * 100.0
    return df


SWEEP_PARAM_NAMES = {
    "alpha",
    "tau_scale",
    "update_freq",
    "score_bias",
    "w_n",
    "init_score",
}

DEFAULTS = {
    "alpha": 0.9,
    "tau_scale": 1.0,
    "update_freq": 20,
    "score_bias": 7.2,
    "w_n": 1.0,
    "init_score": 12.0,
}


def _is_default_sweep_row(row) -> bool:
    param = row["sweep_param"]
    val = row["sweep_value"]
    if param not in DEFAULTS:
        return False
    default = DEFAULTS[param]
    if isinstance(default, int):
        try:
            return int(float(val)) == default
        except (TypeError, ValueError):
            return False
    return abs(float(val) - float(default)) < 1e-9


def to_markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "run_id",
        "sweep_param",
        "sweep_value",
        "eval_ppl",
        "eval_loss",
        "final_train_loss",
        "delta_ppl_vs_default_pct",
        "delta_ppl_vs_adamw_pct",
        "quant_error",
        "bit_4_pct",
        "bit_8_pct",
        "bit_16_pct",
        "bit_32_pct",
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


def main():
    rows = load_run_results()
    if not rows:
        print(f"No results found in {RUN_DIR}")
        print("Run: bash run_sweep.sh")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["optimizer", "sweep_param", "sweep_value"], na_position="first")
    df = add_delta_columns(df)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.6f")

    md = to_markdown_table(df)
    OUT_MD.write_text(
        "# MDQ Score-Smoothing Parameter Sensitivity\n\n" + md + "\n",
        encoding="utf-8",
    )

    print(f"Loaded {len(df)} runs")
    print(f"Summary CSV: {OUT_CSV}")
    print(f"Summary MD:  {OUT_MD}")

    if "eval_ppl" in df.columns:
        mdq = df[df["optimizer"] == "MDQ"]
        if len(mdq):
            max_delta = mdq["delta_ppl_vs_default_pct"].abs().max()
            print(f"Max |ΔPPL vs default| among MDQ runs: {max_delta:.2f}%")


if __name__ == "__main__":
    main()

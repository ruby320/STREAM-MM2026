#!/usr/bin/env python3
"""从 results/history_*.json 重新生成 summary_metrics.csv。"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob

import pandas as pd


def _gpu_dict_keys(records: list[dict], field: str) -> list[str]:
    keys: set[str] = set()
    for r in records:
        data = r.get(field) or {}
        if isinstance(data, dict):
            keys.update(data.keys())
    return sorted(keys)


def record_to_row(
    r: dict, peak_keys: list[str], opt_state_keys: list[str]
) -> dict:
    def _fmt_opt(v, nd=2):
        if v is None:
            return ""
        return f"{float(v):.{nd}f}"

    row = {
        "Optimizer": r["Opt"],
        "Avg_Loss": f"{r.get('Avg_Loss', r.get('Acc', 0.0)):.4f}",
        "Measure_Train_Wall(ms)": f"{r.get('Measure_Train_Wall_ms', r.get('Total_Train_Time_h', 0.0) * 3_600_000):.2f}",
        "Avg_Step_Wall(ms)": f"{r.get('Avg_Step_Wall_ms', 0.0):.2f}",
        "Throughput(tokens/s)": f"{r.get('Throughput_tokens_per_s', 0.0):.0f}",
        "Warmup_Steps": r.get("Warmup_Steps", ""),
        "Measured_Steps": r.get("Measured_Steps", ""),
        "Run_Index": r.get("Run_Index", ""),
        "Num_GPUs": r.get("Num_GPUs", ""),
        "Avg_Forward(ms)": f"{r.get('Avg_Forward_ms', 0.0):.2f}",
        "Avg_Backward(ms)": f"{r.get('Avg_Backward_ms', 0.0):.2f}",
        "Avg_LayerBackward_Mean(ms)": f"{r.get('Avg_LayerBackward_Mean_ms', 0.0):.2f}",
        "Avg_OptimizerStep(ms)": f"{r.get('Avg_OptimizerStep_ms', 0.0):.2f}",
        "Avg_Unpack(ms)": f"{r.get('Avg_Unpack_ms', r.get('Avg_Dequant_ms', 0.0)):.2f}",
        "Avg_Pack(ms)": f"{r.get('Avg_Pack_ms', r.get('Avg_Quant_ms', 0.0)):.2f}",
        "Avg_MDQ_FusedKernel(ms)": _fmt_opt(r.get("Avg_MDQ_FusedKernel_ms")),
        "Avg_Kernel_Dequant(ms)": _fmt_opt(r.get("Avg_Kernel_Dequant_ms")),
        "Avg_Kernel_Adam_mv(ms)": _fmt_opt(r.get("Avg_Kernel_Adam_mv_ms")),
        "Avg_Kernel_ScaleReduce(ms)": _fmt_opt(r.get("Avg_Kernel_ScaleReduce_ms")),
        "Avg_Kernel_Quant(ms)": _fmt_opt(r.get("Avg_Kernel_Quant_ms")),
        "Avg_Kernel_UpdateP(ms)": _fmt_opt(r.get("Avg_Kernel_UpdateP_ms")),
        "Avg_MDQ_Extra_Stats(ms)": _fmt_opt(r.get("Avg_MDQ_Extra_Stats_ms")),
        "Avg_MDQ_Extra_Stats_on_update(ms)": _fmt_opt(
            r.get("Avg_MDQ_Extra_Stats_on_update_ms")
        ),
        "Avg_MDQ_GradStats_on_update(ms)": _fmt_opt(
            r.get("Avg_MDQ_GradStats_on_update_ms")
        ),
        "Avg_MDQ_AllReduce_on_update(ms)": _fmt_opt(
            r.get("Avg_MDQ_AllReduce_on_update_ms")
        ),
        "Avg_MDQ_ScoreBits_on_update(ms)": _fmt_opt(
            r.get("Avg_MDQ_ScoreBits_on_update_ms")
        ),
        "MDQ_Update_Steps_in_measure": r.get("MDQ_Update_Steps_in_measure", ""),
    }

    peak = r.get("Peak_GPU_Memory_GB") or {}
    for k in peak_keys:
        col = f"Peak_Mem_{k}(GB)"
        row[col] = f"{float(peak.get(k, 0.0)):.2f}" if k in peak else ""

    opt_state = r.get("Optimizer_State_GB") or {}
    for k in opt_state_keys:
        col = f"OptState_{k}(GB)"
        row[col] = f"{float(opt_state.get(k, 0.0)):.2f}" if k in opt_state else ""

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    pattern = os.path.join(args.dir, "history_*.json")
    files = sorted(glob(pattern))
    if not files:
        raise SystemExit(f"未找到: {pattern}")

    records = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            records.append(json.load(f))

    peak_keys = _gpu_dict_keys(records, "Peak_GPU_Memory_GB")
    opt_state_keys = _gpu_dict_keys(records, "Optimizer_State_GB")
    rows = [record_to_row(r, peak_keys, opt_state_keys) for r in records]

    out = args.output or os.path.join(args.dir, "summary_metrics.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"已写入 {len(rows)} 行 -> {out}")


if __name__ == "__main__":
    main()

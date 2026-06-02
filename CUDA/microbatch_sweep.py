#!/usr/bin/env python3
"""
Micro-batch sweep：验证 MDQ 节省 optimizer state 后能否增大 per-GPU batch 并提升吞吐。

对齐 Cfinetuning_timetest / baseline 配置（GPT2-XL, accum, seq, bf16）。
对每个 (optimizer, micro_batch) 探测是否 OOM，并记录 Peak / OptState / OptShare / 吞吐。

用法（4 卡，指数跳跃 + 二分找 OOM）:
  CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 bash run_microbatch_sweep.sh

固定 Global_Batch=256（micro×accum×GPUs=256，默认 2 卡 P=128）:
  CUDA_VISIBLE_DEVICES=4,5 NUM_GPUS=2 bash run_fixed_global_batch_sweep.sh

单优化器:
  torchrun --standalone --nproc_per_node=4 microbatch_sweep.py \\
    --optimizer AdamW-32bit --find-max-micro --micro-min 4 --micro-max 64 --benchmark
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time

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
    ALL_OPTIMIZERS,
    DATA_PATH,
    GRAD_ACCUM_STEPS,
    LR,
    MODEL_TYPE,
    SEQ_LEN,
    build_optimizer,
    seed_everything,
    sync_cuda,
    _gather_peak_memory_gb,
    _gather_per_rank_gb,
    _optimizer_state_bytes,
    _unwrap_optimizer,
)

DEFAULT_SAVE_DIR = os.path.join(_CUDA_DIR, "results", "microbatch_sweep")
DEFAULT_FIXED_G_DIR = os.path.join(_CUDA_DIR, "results", "fixed_global_batch")
DEFAULT_FIXED_GLOBAL_BATCH = int(os.environ.get("FIXED_GLOBAL_BATCH", "256"))
DEFAULT_WARMUP = int(os.environ.get("MICRO_WARMUP_STEPS", "2"))
DEFAULT_PROBE_STEPS = int(os.environ.get("MICRO_PROBE_STEPS", "8"))
DEFAULT_BENCHMARK_STEPS = int(os.environ.get("MICRO_BENCHMARK_STEPS", "20"))
DEFAULT_BENCHMARK_WARMUP = int(os.environ.get("MICRO_BENCHMARK_WARMUP", "10"))


def parse_args():
    p = argparse.ArgumentParser(description="GPT2-XL micro-batch / OptState 比例 sweep")
    p.add_argument(
        "--optimizer",
        choices=ALL_OPTIMIZERS,
        required=True,
        help="单个 optimizer（建议每 optimizer 独立进程）",
    )
    p.add_argument("--micro-min", type=int, default=4)
    p.add_argument(
        "--micro-max",
        type=int,
        default=64,
        help="安全上限；find-max 时指数跳跃不超过此值",
    )
    p.add_argument(
        "--micro-list",
        default="",
        help="逗号分隔 micro 列表，指定时忽略 min/max 与 find-max-micro",
    )
    p.add_argument(
        "--find-max-micro",
        action="store_true",
        help="指数跳跃 + 二分搜索 max micro（需 OOM 或达到 micro-max）",
    )
    p.add_argument(
        "--search-mode",
        choices=("exp_binary", "linear"),
        default=os.environ.get("MICRO_SEARCH_MODE", "exp_binary"),
        help="find-max 搜索策略：exp_binary（默认）或 linear",
    )
    p.add_argument("--grad-accum", type=int, default=GRAD_ACCUM_STEPS)
    p.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--probe-steps", type=int, default=DEFAULT_PROBE_STEPS)
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="在 probe max 上 benchmark；失败时默认向更小 micro 回退",
    )
    p.add_argument(
        "--no-benchmark-fallback",
        action="store_true",
        help="仅在 probe max_ok_micro 上 benchmark，失败不回退",
    )
    p.add_argument("--benchmark-steps", type=int, default=DEFAULT_BENCHMARK_STEPS)
    p.add_argument(
        "--peak-budget-gb",
        type=float,
        default=None,
        help="可选：Peak 超过此预算(GB)则标记 over_budget（仍记录是否 OOM）",
    )
    p.add_argument("--output-dir", default=DEFAULT_SAVE_DIR)
    p.add_argument(
        "--fixed-global-batch",
        type=int,
        default=None,
        help="固定 global batch（micro×accum×GPUs）；设后从大到小搜 max micro 并 benchmark",
    )
    p.add_argument(
        "--benchmark-warmup-steps",
        type=int,
        default=DEFAULT_BENCHMARK_WARMUP,
        help="fixed-G benchmark 阶段 warmup global steps（默认 10，对齐 timetest）",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--orchestrate",
        action="store_true",
        help="单进程编排 find-max：每个 micro 独立 torchrun 子进程，避免 OOM 后 NCCL 死锁",
    )
    p.add_argument(
        "--probe-json-out",
        default="",
        help="仅 probe 单个 micro，结果写入 JSON 后退出（供 --orchestrate 子进程调用）",
    )
    p.add_argument(
        "--benchmark-json-out",
        default="",
        help="仅 benchmark 单个 micro，结果写入 JSON 后退出（供 --orchestrate 子进程调用）",
    )
    return p.parse_args()


def _num_gpus() -> int:
    return int(os.environ.get("NUM_GPUS", "4"))


def _torchrun_cmd(extra_args: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={_num_gpus()}",
        os.path.abspath(__file__),
        *extra_args,
    ]


def _fallback_probe_row(args, micro: int, phase: str, *, status: str = "oom") -> dict:
    return {
        "Optimizer": args.optimizer,
        "Micro_Batch": micro,
        "Grad_Accum": args.grad_accum,
        "Num_GPUs": _num_gpus(),
        "Global_Batch": micro * args.grad_accum * _num_gpus(),
        "Seq_Len": SEQ_LEN,
        "Status": status,
        "OOM": status == "oom",
        "Over_Peak_Budget": False,
        "Search_Phase": phase,
    }


def _run_isolated_probe(args, micro: int, phase: str) -> dict:
    """每个 micro 独立子进程 probe，OOM 后不影响后续搜索。"""
    fd, path = tempfile.mkstemp(suffix=".json", prefix="micro_probe_")
    os.close(fd)
    cmd = _torchrun_cmd(
        [
            "--optimizer",
            args.optimizer,
            "--micro-list",
            str(micro),
            "--grad-accum",
            str(args.grad_accum),
            "--warmup-steps",
            str(args.warmup_steps),
            "--probe-steps",
            str(args.probe_steps),
            "--seed",
            str(args.seed),
            "--output-dir",
            args.output_dir,
            "--probe-json-out",
            path,
        ]
    )
    if args.peak_budget_gb is not None:
        cmd.extend(["--peak-budget-gb", str(args.peak_budget_gb)])

    print(f"\n--- micro={micro} accum={args.grad_accum} ({phase}, isolated) ---")
    result = subprocess.run(cmd, cwd=_CUDA_DIR)

    row = None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            row = json.load(f)
        os.remove(path)

    if row is None:
        status = "oom" if result.returncode != 0 else "error"
        row = _fallback_probe_row(args, micro, phase, status=status)
    else:
        row["Search_Phase"] = phase

    print(
        f"  status={row.get('Status')} peak_mean={row.get('Peak_Mean_GB', 'n/a')} "
        f"opt={row.get('OptState_Mean_GB', 'n/a')} "
        f"share={row.get('OptShare_Mean_Pct', 'n/a')}%"
    )
    return row


def _run_isolated_benchmark(args, micro: int) -> dict:
    fd, path = tempfile.mkstemp(suffix=".json", prefix="micro_bench_")
    os.close(fd)
    cmd = _torchrun_cmd(
        [
            "--optimizer",
            args.optimizer,
            "--micro-list",
            str(micro),
            "--grad-accum",
            str(args.grad_accum),
            "--warmup-steps",
            str(args.warmup_steps),
            "--probe-steps",
            str(args.probe_steps),
            "--benchmark-steps",
            str(args.benchmark_steps),
            "--seed",
            str(args.seed),
            "--output-dir",
            args.output_dir,
            "--benchmark",
            "--benchmark-json-out",
            path,
        ]
    )
    if args.no_benchmark_fallback:
        cmd.append("--no-benchmark-fallback")

    print(f"\n>>> Benchmark micro={micro} (isolated), steps={args.benchmark_steps}")
    result = subprocess.run(cmd, cwd=_CUDA_DIR)

    bench = None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            bench = json.load(f)
        os.remove(path)

    if bench is None:
        bench = {
            "Benchmark_Micro": micro,
            "Benchmark_Status": "oom" if result.returncode != 0 else "error",
        }
    print(f">>> benchmark micro={micro} status={bench.get('Benchmark_Status')}")
    return bench


def _orchestrate_find_max_exp_binary(args, rows: list[dict]) -> None:
    row = _run_isolated_probe(args, args.micro_min, "exp")
    rows.append(row)
    if not _is_probe_ok(row):
        return

    last_ok = args.micro_min
    first_fail: int | None = None

    while True:
        candidate = _next_exp_candidate(last_ok, args.micro_max)
        if candidate is None:
            break
        row = _run_isolated_probe(args, candidate, "exp")
        rows.append(row)
        if _is_probe_ok(row):
            last_ok = candidate
            if last_ok >= args.micro_max:
                break
        else:
            first_fail = candidate
            break

    if first_fail is None:
        return

    lo, hi = last_ok, first_fail
    print(f"\n>>> 指数阶段: ok<={last_ok}, fail>={first_fail}，二分搜索 ...")

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        row = _run_isolated_probe(args, mid, "binary")
        rows.append(row)
        if _is_probe_ok(row):
            lo = mid
        else:
            hi = mid

    print(f">>> 二分完成: max_ok_micro={lo}, first_oom_micro={hi}")


def _orchestrate_find_max_linear(args, rows: list[dict]) -> None:
    micro = args.micro_min
    while micro <= args.micro_max:
        row = _run_isolated_probe(args, micro, "linear")
        rows.append(row)
        if _is_probe_ok(row):
            micro += 1
            continue
        break


def _orchestrate_benchmark_with_fallback(
    args,
    rows: list[dict],
    probe_max_ok: int | None,
) -> tuple[dict | None, int | None]:
    ok_micros = _probe_ok_micros(rows)
    if not ok_micros:
        return None, None

    if probe_max_ok is not None and probe_max_ok in ok_micros:
        try_order = list(reversed(ok_micros[: ok_micros.index(probe_max_ok) + 1]))
    else:
        try_order = list(reversed(ok_micros))

    if args.no_benchmark_fallback:
        try_order = try_order[:1]

    last_bench = None
    benchmark_max_ok = None
    for micro in try_order:
        bench = _run_isolated_benchmark(args, micro)
        last_bench = bench
        if bench.get("Benchmark_Status") == "ok":
            benchmark_max_ok = micro
            for r in rows:
                if r.get("Micro_Batch") == micro and _is_probe_ok(r):
                    r.update(bench)
            break
        if args.no_benchmark_fallback:
            break
        if micro != try_order[-1]:
            print(">>> benchmark 失败，尝试更小 micro ...")

    return last_bench, benchmark_max_ok


def _write_sweep_results(args, rows: list[dict], summary: dict, sweep_meta: dict) -> None:
    tag = args.optimizer.replace("/", "_")
    json_path = os.path.join(args.output_dir, f"sweep_{tag}.json")
    csv_path = os.path.join(args.output_dir, f"sweep_{tag}.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_rows = []
    for row in rows:
        csv_row = dict(row)
        csv_row.update(
            {
                "Probe_Max_OK_Micro": sweep_meta["probe_max_ok_micro"],
                "Max_OK_Micro": sweep_meta["probe_max_ok_micro"],
                "Benchmark_Max_OK_Micro": summary.get("benchmark_max_ok_micro"),
                "First_OOM_Micro": sweep_meta["first_oom_micro"],
                "Reached_OOM_Boundary": sweep_meta["reached_oom_boundary"],
                "Hit_MicroMax_Without_OOM": sweep_meta["hit_micro_max_without_oom"],
            }
        )
        csv_rows.append(csv_row)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    combined_path = os.path.join(args.output_dir, "sweep_all.csv")
    if os.path.exists(combined_path):
        old = pd.read_csv(combined_path)
        old = old[old["Optimizer"] != args.optimizer]
        pd.concat([old, pd.DataFrame(csv_rows)], ignore_index=True).to_csv(
            combined_path, index=False
        )
    else:
        pd.DataFrame(csv_rows).to_csv(combined_path, index=False)

    print(f"\n>>> probe_max_ok_micro={sweep_meta['probe_max_ok_micro']}")
    print(f">>> benchmark_max_ok_micro={summary.get('benchmark_max_ok_micro')}")
    print(f">>> first_oom_micro={sweep_meta['first_oom_micro']}")
    print(f">>> reached_oom_boundary={sweep_meta['reached_oom_boundary']}")
    if summary.get("benchmark") and summary["benchmark"].get("Benchmark_Status") != "ok":
        print(">>> 警告: benchmark 未成功，请查看 Benchmark_Status 或增大 probe_steps")
    if sweep_meta["hit_micro_max_without_oom"]:
        print(
            f">>> 警告: 扫到 micro_max={args.micro_max} 仍未 OOM，"
            "请增大 MICRO_MAX 继续探测"
        )
    print(f">>> 已写入 {csv_path}")
    print(f">>> 合并表 {combined_path}")


def _orchestrate_sweep(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Micro-batch sweep (orchestrated) | optimizer={args.optimizer}")
    print(
        f"  num_gpus={_num_gpus()} grad_accum={args.grad_accum} "
        f"find_max={args.micro_min}..{args.micro_max} mode={args.search_mode}"
    )

    rows: list[dict] = []
    if args.search_mode == "linear":
        _orchestrate_find_max_linear(args, rows)
    else:
        _orchestrate_find_max_exp_binary(args, rows)

    rows.sort(key=lambda r: r["Micro_Batch"])
    sweep_meta = _summarize_sweep(rows, args.micro_max)
    probe_max_ok = sweep_meta["probe_max_ok_micro"]

    summary = {
        "optimizer": args.optimizer,
        "num_gpus": _num_gpus(),
        "orchestrated": True,
        "find_max_micro": True,
        "search_mode": args.search_mode,
        "micro_min": args.micro_min,
        "micro_max": args.micro_max,
        "micro_list": sorted({r["Micro_Batch"] for r in rows}),
        "grad_accum": args.grad_accum,
        "peak_budget_gb": args.peak_budget_gb,
        "warmup_steps": args.warmup_steps,
        "probe_steps": args.probe_steps,
        "benchmark_steps": args.benchmark_steps,
        "benchmark_fallback": not args.no_benchmark_fallback,
        "benchmark_max_ok_micro": None,
        **sweep_meta,
        "rows": rows,
    }

    if args.benchmark and probe_max_ok is not None:
        print(f"\n>>> probe 完成 (probe_max_ok={probe_max_ok})，开始 isolated benchmark")
        bench, benchmark_max_ok = _orchestrate_benchmark_with_fallback(
            args, rows, probe_max_ok
        )
        summary["benchmark"] = bench
        summary["benchmark_max_ok_micro"] = benchmark_max_ok

    _write_sweep_results(args, rows, summary, sweep_meta)


def _fixed_g_micro_accum_pairs(fixed_global_batch: int, num_gpus: int) -> list[tuple[int, int]]:
    """返回 (micro, grad_accum) 列表，按 micro 降序；满足 micro×accum×num_gpus=fixed_global_batch。"""
    if fixed_global_batch % num_gpus != 0:
        raise ValueError(
            f"fixed_global_batch={fixed_global_batch} 不能被 num_gpus={num_gpus} 整除"
        )
    product = fixed_global_batch // num_gpus
    pairs: list[tuple[int, int]] = []
    micro = 1
    while micro <= product:
        if product % micro == 0:
            pairs.append((micro, product // micro))
        micro += 1
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs


def _accum_for_fixed_g(fixed_global_batch: int, num_gpus: int, micro: int) -> int:
    if (fixed_global_batch % num_gpus) != 0 or (fixed_global_batch % (micro * num_gpus)) != 0:
        raise ValueError(
            f"micro={micro} 与 fixed_global_batch={fixed_global_batch}, "
            f"num_gpus={num_gpus} 无法组成整数 accum"
        )
    return fixed_global_batch // (micro * num_gpus)


def _parse_micro_list(args) -> list[int] | None:
    if args.micro_list.strip():
        return sorted({int(x.strip()) for x in args.micro_list.split(",") if x.strip()})
    if args.find_max_micro:
        return None
    if args.micro_max < args.micro_min:
        raise ValueError("micro-max 必须 >= micro-min")
    return list(range(args.micro_min, args.micro_max + 1))


def _is_probe_ok(row: dict) -> bool:
    return row.get("Status") == "ok" and not row.get("Over_Peak_Budget")


def _summarize_sweep(rows: list[dict], micro_max: int) -> dict:
    ok_rows = sorted(
        [r for r in rows if _is_probe_ok(r)],
        key=lambda r: r["Micro_Batch"],
    )
    oom_rows = sorted(
        [r for r in rows if r.get("OOM")],
        key=lambda r: r["Micro_Batch"],
    )
    max_ok_micro = ok_rows[-1]["Micro_Batch"] if ok_rows else None
    first_oom_micro = oom_rows[0]["Micro_Batch"] if oom_rows else None
    reached_oom_boundary = first_oom_micro is not None
    tested_max = max(r["Micro_Batch"] for r in rows) if rows else None
    return {
        "probe_max_ok_micro": max_ok_micro,
        "max_ok_micro": max_ok_micro,
        "first_oom_micro": first_oom_micro,
        "reached_oom_boundary": reached_oom_boundary,
        "tested_max_micro": tested_max,
        "hit_micro_max_without_oom": (
            not reached_oom_boundary and tested_max is not None and tested_max >= micro_max
        ),
    }


def _probe_ok_micros(rows: list[dict]) -> list[int]:
    return sorted(
        r["Micro_Batch"] for r in rows if _is_probe_ok(r)
    )


def _run_benchmark_with_fallback(
    accelerator,
    dataset,
    args,
    rows: list[dict],
    probe_max_ok: int | None,
) -> tuple[dict | None, int | None]:
    """从 probe_max 开始 benchmark；OOM 时向更小 micro 回退，直到成功或无候选。"""
    ok_micros = _probe_ok_micros(rows)
    if not ok_micros:
        return None, None

    if probe_max_ok is not None and probe_max_ok in ok_micros:
        try_order = list(reversed(ok_micros[: ok_micros.index(probe_max_ok) + 1]))
    else:
        try_order = list(reversed(ok_micros))

    if args.no_benchmark_fallback:
        try_order = try_order[:1]

    last_bench = None
    benchmark_max_ok = None
    for micro in try_order:
        if accelerator.is_main_process:
            accelerator.print(
                f"\n>>> Benchmark micro={micro}, steps={args.benchmark_steps}"
            )
        accelerator.wait_for_everyone()
        _release_gpu_memory(accelerator)
        bench = _benchmark_micro(
            accelerator,
            dataset,
            args.optimizer,
            micro,
            args.grad_accum,
            args.warmup_steps,
            args.benchmark_steps,
            args.seed,
        )
        last_bench = bench
        status = bench.get("Benchmark_Status", "error")
        if accelerator.is_main_process:
            accelerator.print(f">>> benchmark micro={micro} status={status}")

        if status == "ok":
            benchmark_max_ok = micro
            for row in rows:
                if row.get("Micro_Batch") == micro and _is_probe_ok(row):
                    row.update(bench)
            break

        if args.no_benchmark_fallback:
            break
        if accelerator.is_main_process and micro != try_order[-1]:
            accelerator.print(">>> benchmark 失败，尝试更小 micro ...")

    return last_bench, benchmark_max_ok


def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _opt_state_gb(accelerator, optimizer) -> dict:
    local_gb = _optimizer_state_bytes(_unwrap_optimizer(optimizer)) / (1024**3)
    return _gather_per_rank_gb(accelerator, local_gb)


def _opt_share(peak_gb: dict, opt_gb: dict) -> dict:
    out = {}
    for k, peak in peak_gb.items():
        opt = opt_gb.get(k, 0.0)
        out[k] = (opt / peak * 100.0) if peak > 0 else 0.0
    return out


def _mean_dict(values: dict) -> float:
    return float(sum(values.values()) / len(values)) if values else 0.0


def _run_training_steps(
    accelerator,
    model,
    optimizer,
    train_dl,
    micro_batch: int,
    grad_accum: int,
    warmup_steps: int,
    total_steps: int,
    measure_throughput: bool,
) -> dict:
    """跑 warmup + total_steps 个 global optimizer step。"""
    model.train()
    global_step = 0
    phase = "warmup"
    measure_wall_start = None
    tokens_per_step = accelerator.num_processes * micro_batch * grad_accum * SEQ_LEN
    measure_steps_done = 0
    losses: list[float] = []
    opt_state_gb = None

    _reset_peak()
    data_iter = iter(train_dl)

    while global_step < warmup_steps + total_steps:
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
        optimizer.step()
        sync_cuda()
        optimizer.zero_grad()
        global_step += 1

        if global_step == warmup_steps and measure_throughput:
            phase = "measure"
            _reset_peak()
            measure_wall_start = time.perf_counter()

        if global_step >= max(1, warmup_steps) and opt_state_gb is None:
            opt_state_gb = _opt_state_gb(accelerator, optimizer)

        if phase == "measure":
            step_loss = float(outputs.loss.item()) * grad_accum
            losses.append(step_loss)
            measure_steps_done += 1

    peak_gb = _gather_peak_memory_gb(accelerator)
    if opt_state_gb is None:
        opt_state_gb = _opt_state_gb(accelerator, optimizer)
    share = _opt_share(peak_gb, opt_state_gb)

    wall_ms = 0.0
    throughput = 0.0
    if measure_throughput and measure_wall_start is not None and measure_steps_done > 0:
        wall_ms = (time.perf_counter() - measure_wall_start) * 1000.0
        tokens = measure_steps_done * tokens_per_step
        throughput = tokens / (wall_ms / 1000.0) if wall_ms > 0 else 0.0

    return {
        "measure_steps": measure_steps_done,
        "avg_loss": float(sum(losses) / len(losses)) if losses else None,
        "peak_gpu_gb": peak_gb,
        "opt_state_gpu_gb": opt_state_gb,
        "opt_share_pct": share,
        "peak_mean_gb": _mean_dict(peak_gb),
        "opt_state_mean_gb": _mean_dict(opt_state_gb),
        "opt_share_mean_pct": _mean_dict(share),
        "measure_wall_ms": wall_ms,
        "throughput_tokens_per_s": throughput,
        "tokens_per_global_step": tokens_per_step,
        "global_batch": micro_batch * grad_accum * accelerator.num_processes,
    }


def _release_gpu_memory(accelerator, *, skip_sync: bool = False) -> None:
    """probe/benchmark 之间释放 Accelerate + CUDA 缓存显存。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    try:
        accelerator.free_memory()
    except Exception:
        pass
    if not skip_sync:
        accelerator.wait_for_everyone()


def _cleanup(accelerator, model=None, optimizer=None, train_dl=None, *, had_oom: bool = False):
    if train_dl is not None:
        del train_dl
    if model is not None:
        del model
    if optimizer is not None:
        del optimizer
    _release_gpu_memory(accelerator, skip_sync=had_oom)


def _try_micro_config(
    accelerator,
    dataset,
    opt_name: str,
    micro_batch: int,
    grad_accum: int,
    warmup_steps: int,
    probe_steps: int,
    peak_budget_gb: float | None,
    seed: int,
) -> dict:
    row = {
        "Optimizer": opt_name,
        "Micro_Batch": micro_batch,
        "Grad_Accum": grad_accum,
        "Num_GPUs": accelerator.num_processes,
        "Global_Batch": micro_batch * grad_accum * accelerator.num_processes,
        "Seq_Len": SEQ_LEN,
        "Status": "ok",
        "OOM": False,
        "Over_Peak_Budget": False,
    }

    global_batch = micro_batch * accelerator.num_processes * grad_accum
    config = GPT2Config.from_pretrained(MODEL_TYPE)
    model = None
    optimizer = None
    train_dl = None
    had_oom = False

    try:
        seed_everything(seed)
        set_seed(seed)

        model = GPT2LMHeadModel(config)
        model.gradient_checkpointing_enable()
        optimizer = build_optimizer(opt_name, model, LR, config.n_layer, global_batch)

        train_dl = DataLoader(
            dataset,
            batch_size=micro_batch,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
            persistent_workers=False,
        )
        model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)

        metrics = _run_training_steps(
            accelerator,
            model,
            optimizer,
            train_dl,
            micro_batch,
            grad_accum,
            warmup_steps,
            probe_steps,
            measure_throughput=False,
        )
        row.update(
            {
                "Probe_Steps": probe_steps,
                "Warmup_Steps": warmup_steps,
                "Peak_Mean_GB": round(metrics["peak_mean_gb"], 2),
                "OptState_Mean_GB": round(metrics["opt_state_mean_gb"], 2),
                "OptShare_Mean_Pct": round(metrics["opt_share_mean_pct"], 2),
                "Avg_Loss": metrics["avg_loss"],
                "Tokens_per_Global_Step": metrics["tokens_per_global_step"],
            }
        )
        for k, v in metrics["peak_gpu_gb"].items():
            row[f"Peak_{k}(GB)"] = round(v, 2)
        for k, v in metrics["opt_state_gpu_gb"].items():
            row[f"OptState_{k}(GB)"] = round(v, 2)
        for k, v in metrics["opt_share_pct"].items():
            row[f"OptShare_{k}(Pct)"] = round(v, 2)

        if peak_budget_gb is not None and metrics["peak_mean_gb"] > peak_budget_gb:
            row["Over_Peak_Budget"] = True
            row["Peak_Budget_GB"] = peak_budget_gb

    except torch.cuda.OutOfMemoryError:
        had_oom = True
        row["Status"] = "oom"
        row["OOM"] = True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            had_oom = True
            row["Status"] = "oom"
            row["OOM"] = True
        else:
            row["Status"] = "error"
            row["Error"] = str(e)[:200]
    finally:
        _cleanup(accelerator, model, optimizer, train_dl, had_oom=had_oom)

    return row


def _benchmark_micro(
    accelerator,
    dataset,
    opt_name: str,
    micro_batch: int,
    grad_accum: int,
    warmup_steps: int,
    benchmark_steps: int,
    seed: int,
) -> dict:
    global_batch = micro_batch * accelerator.num_processes * grad_accum
    config = GPT2Config.from_pretrained(MODEL_TYPE)
    model = None
    optimizer = None
    train_dl = None
    base = {
        "Benchmark_Micro": micro_batch,
        "Benchmark_Status": "ok",
    }
    had_oom = False

    try:
        seed_everything(seed)
        set_seed(seed)

        model = GPT2LMHeadModel(config)
        model.gradient_checkpointing_enable()
        optimizer = build_optimizer(opt_name, model, LR, config.n_layer, global_batch)
        train_dl = DataLoader(
            dataset,
            batch_size=micro_batch,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )
        model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)

        metrics = _run_training_steps(
            accelerator,
            model,
            optimizer,
            train_dl,
            micro_batch,
            grad_accum,
            warmup_steps,
            benchmark_steps,
            measure_throughput=True,
        )
        return {
            **base,
            "Benchmark_Global_Batch": metrics["global_batch"],
            "Benchmark_Grad_Accum": grad_accum,
            "Benchmark_Measure_Steps": benchmark_steps,
            "Benchmark_Warmup_Steps": warmup_steps,
            "Benchmark_Avg_Loss": metrics["avg_loss"],
            "Benchmark_Peak_Mean_GB": round(metrics["peak_mean_gb"], 2),
            "Benchmark_OptState_Mean_GB": round(metrics["opt_state_mean_gb"], 2),
            "Benchmark_OptShare_Mean_Pct": round(metrics["opt_share_mean_pct"], 2),
            "Benchmark_Wall_ms": round(metrics["measure_wall_ms"], 2),
            "Benchmark_Avg_Global_Step_ms": round(
                metrics["measure_wall_ms"] / benchmark_steps if benchmark_steps else 0.0,
                2,
            ),
            "Benchmark_Throughput_tok_s": round(metrics["throughput_tokens_per_s"], 0),
        }
    except torch.cuda.OutOfMemoryError:
        had_oom = True
        return {**base, "Benchmark_Status": "oom"}
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            had_oom = True
            return {**base, "Benchmark_Status": "oom"}
        return {**base, "Benchmark_Status": "error", "Benchmark_Error": str(e)[:200]}
    finally:
        _cleanup(accelerator, model, optimizer, train_dl, had_oom=had_oom)


def _probe_one(
    accelerator,
    dataset,
    args,
    micro: int,
    rows: list[dict],
    *,
    phase: str = "list",
    cache: dict[int, dict] | None = None,
    grad_accum: int | None = None,
) -> dict:
    accum = grad_accum if grad_accum is not None else args.grad_accum
    cache_key = (micro, accum) if args.fixed_global_batch else micro
    if cache is not None and cache_key in cache:
        if accelerator.is_main_process:
            accelerator.print(
                f"\n--- micro={micro} accum={accum} ({phase}, cached) "
                f"status={cache[cache_key].get('Status')} ---"
            )
        return cache[cache_key]

    if accelerator.is_main_process:
        accelerator.print(f"\n--- micro={micro} accum={accum} ({phase}) ---")
    accelerator.wait_for_everyone()
    row = _try_micro_config(
        accelerator,
        dataset,
        args.optimizer,
        micro,
        accum,
        args.warmup_steps,
        args.probe_steps,
        args.peak_budget_gb,
        args.seed,
    )
    row["Search_Phase"] = phase
    if args.fixed_global_batch is not None:
        row["Fixed_Global_Batch"] = args.fixed_global_batch
    rows.append(row)
    if cache is not None:
        cache[cache_key] = row
    if accelerator.is_main_process:
        accelerator.print(
            f"  status={row.get('Status')} peak_mean={row.get('Peak_Mean_GB', 'n/a')} "
            f"opt={row.get('OptState_Mean_GB', 'n/a')} "
            f"share={row.get('OptShare_Mean_Pct', 'n/a')}%"
        )
    return row


def _next_exp_candidate(last_ok: int, micro_max: int) -> int | None:
    """从 last_ok 指数跳跃到下一个候选 micro（不超过 micro_max）。"""
    if last_ok >= micro_max:
        return None
    candidate = min(last_ok * 2, micro_max)
    if candidate <= last_ok:
        candidate = last_ok + 1
    if candidate > micro_max:
        return None
    return candidate


def _find_max_micro_linear(
    accelerator,
    dataset,
    args,
    rows: list[dict],
    cache: dict[int, dict],
) -> None:
    micro = args.micro_min
    while micro <= args.micro_max:
        row = _probe_one(
            accelerator, dataset, args, micro, rows, phase="linear", cache=cache
        )
        if _is_probe_ok(row):
            micro += 1
            continue
        break


def _find_max_micro_exp_binary(
    accelerator,
    dataset,
    args,
    rows: list[dict],
    cache: dict[int, dict],
) -> None:
    """指数跳跃定位 OOM 区间，再二分精确 max_ok_micro。"""
    row = _probe_one(
        accelerator,
        dataset,
        args,
        args.micro_min,
        rows,
        phase="exp",
        cache=cache,
    )
    if not _is_probe_ok(row):
        return

    last_ok = args.micro_min
    first_fail: int | None = None

    while True:
        candidate = _next_exp_candidate(last_ok, args.micro_max)
        if candidate is None:
            break
        row = _probe_one(
            accelerator,
            dataset,
            args,
            candidate,
            rows,
            phase="exp",
            cache=cache,
        )
        if _is_probe_ok(row):
            last_ok = candidate
            if last_ok >= args.micro_max:
                break
        else:
            first_fail = candidate
            break

    if first_fail is None:
        return

    lo, hi = last_ok, first_fail
    if accelerator.is_main_process:
        accelerator.print(
            f"\n>>> 指数阶段: ok<={last_ok}, fail>={first_fail}，二分搜索 ..."
        )
    accelerator.wait_for_everyone()

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        row = _probe_one(
            accelerator,
            dataset,
            args,
            mid,
            rows,
            phase="binary",
            cache=cache,
        )
        if _is_probe_ok(row):
            lo = mid
        else:
            hi = mid

    if accelerator.is_main_process:
        accelerator.print(f">>> 二分完成: max_ok_micro={lo}, first_oom_micro={hi}")


def _run_find_max_micro(
    accelerator,
    dataset,
    args,
    rows: list[dict],
) -> None:
    cache: dict = {}
    if args.search_mode == "linear":
        _find_max_micro_linear(accelerator, dataset, args, rows, cache)
    else:
        _find_max_micro_exp_binary(accelerator, dataset, args, rows, cache)


def _summarize_fixed_g(rows: list[dict]) -> dict:
    ok_rows = sorted(
        [r for r in rows if _is_probe_ok(r)],
        key=lambda r: r["Micro_Batch"],
        reverse=True,
    )
    fail_rows = sorted(
        [r for r in rows if r.get("OOM")],
        key=lambda r: r["Micro_Batch"],
        reverse=True,
    )
    probe_max = ok_rows[0]["Micro_Batch"] if ok_rows else None
    first_fail = fail_rows[0]["Micro_Batch"] if fail_rows else None
    return {
        "probe_max_ok_micro": probe_max,
        "max_ok_micro": probe_max,
        "first_oom_micro": first_fail,
        "reached_oom_boundary": first_fail is not None,
        "tested_max_micro": max(r["Micro_Batch"] for r in rows) if rows else None,
        "hit_micro_max_without_oom": first_fail is None and bool(ok_rows),
    }


def _run_benchmark_fixed_g_with_fallback(
    dataset,
    args,
    rows: list[dict],
    num_gpus: int,
    probe_max_ok: int | None,
) -> tuple[dict | None, int | None, int | None]:
    """fixed-G：从 probe max micro 起 benchmark，OOM 时向更小 micro 回退。"""
    ok_rows = sorted(
        [r for r in rows if _is_probe_ok(r)],
        key=lambda r: r["Micro_Batch"],
        reverse=True,
    )
    if not ok_rows:
        return None, None, None

    if probe_max_ok is not None:
        ok_rows = [r for r in ok_rows if r["Micro_Batch"] <= probe_max_ok]
    if not ok_rows:
        return None, None, None

    try_order = ok_rows if not args.no_benchmark_fallback else ok_rows[:1]

    last_bench = None
    benchmark_max_ok = None
    benchmark_accum = None
    for row in try_order:
        micro = row["Micro_Batch"]
        accum = row["Grad_Accum"]
        accelerator = Accelerator(
            gradient_accumulation_steps=accum,
            mixed_precision="bf16",
        )
        try:
            if accelerator.is_main_process:
                accelerator.print(
                    f"\n>>> Benchmark micro={micro} accum={accum}, "
                    f"G={args.fixed_global_batch}, steps={args.benchmark_steps}"
                )
            accelerator.wait_for_everyone()
            _release_gpu_memory(accelerator)
            bench = _benchmark_micro(
                accelerator,
                dataset,
                args.optimizer,
                micro,
                accum,
                args.benchmark_warmup_steps,
                args.benchmark_steps,
                args.seed,
            )
        finally:
            try:
                accelerator.free_memory()
            except Exception:
                pass
            del accelerator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        last_bench = bench
        status = bench.get("Benchmark_Status", "error")
        if torch.distributed.is_initialized():
            do_print = torch.distributed.get_rank() == 0
        else:
            do_print = True
        if do_print:
            print(f">>> benchmark micro={micro} status={status}")

        if status == "ok":
            benchmark_max_ok = micro
            benchmark_accum = accum
            for r in rows:
                if (
                    r.get("Micro_Batch") == micro
                    and r.get("Grad_Accum") == accum
                    and _is_probe_ok(r)
                ):
                    r.update(bench)
            break

        if args.no_benchmark_fallback:
            break
        if row is not try_order[-1] and do_print:
            print(">>> benchmark 失败，尝试更小 micro ...")

    return last_bench, benchmark_max_ok, benchmark_accum


def _find_max_micro_fixed_g(
    dataset,
    args,
    rows: list[dict],
    num_gpus: int,
) -> None:
    """固定 global batch：按 micro 从大到小 probe，首个 ok 即为 max。"""
    pairs = _fixed_g_micro_accum_pairs(args.fixed_global_batch, num_gpus)
    cache: dict = {}
    probe_max: int | None = None

    if torch.distributed.is_initialized():
        rank0 = torch.distributed.get_rank() == 0
    else:
        rank0 = True

    for micro, accum in pairs:
        accelerator = Accelerator(
            gradient_accumulation_steps=accum,
            mixed_precision="bf16",
        )
        try:
            row = _probe_one(
                accelerator,
                dataset,
                args,
                micro,
                rows,
                phase="fixed_g",
                cache=cache,
                grad_accum=accum,
            )
        finally:
            try:
                accelerator.free_memory()
            except Exception:
                pass
            del accelerator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        if _is_probe_ok(row):
            probe_max = micro
            if rank0:
                print(f">>> fixed-G probe: max_ok_micro={micro} accum={accum}")
            break

    if rank0 and probe_max is None:
        print(">>> fixed-G probe: 所有候选 micro 均 OOM")


def _write_fixed_g_results(args, summary: dict, rows: list[dict], sweep_meta: dict) -> None:
    tag = args.optimizer.replace("/", "_")
    g_tag = f"g{summary['fixed_global_batch']}"
    json_path = os.path.join(args.output_dir, f"{g_tag}_sweep_{tag}.json")
    csv_path = os.path.join(args.output_dir, f"{g_tag}_sweep_{tag}.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_rows = []
    for row in rows:
        csv_row = dict(row)
        csv_row.update(
            {
                "Fixed_Global_Batch": summary["fixed_global_batch"],
                "Probe_Max_OK_Micro": sweep_meta["probe_max_ok_micro"],
                "Max_OK_Micro": sweep_meta["probe_max_ok_micro"],
                "Benchmark_Max_OK_Micro": summary.get("benchmark_max_ok_micro"),
                "Benchmark_Max_Grad_Accum": summary.get("benchmark_max_grad_accum"),
                "First_OOM_Micro": sweep_meta["first_oom_micro"],
                "Reached_OOM_Boundary": sweep_meta["reached_oom_boundary"],
            }
        )
        csv_rows.append(csv_row)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    combined_path = os.path.join(args.output_dir, f"{g_tag}_sweep_all.csv")
    if os.path.exists(combined_path):
        old = pd.read_csv(combined_path)
        old = old[old["Optimizer"] != args.optimizer]
        pd.concat([old, pd.DataFrame(csv_rows)], ignore_index=True).to_csv(
            combined_path, index=False
        )
    else:
        pd.DataFrame(csv_rows).to_csv(combined_path, index=False)

    print(f"\n>>> fixed_G={summary['fixed_global_batch']} probe_max={sweep_meta['probe_max_ok_micro']}")
    print(f">>> benchmark_max_micro={summary.get('benchmark_max_ok_micro')} "
          f"accum={summary.get('benchmark_max_grad_accum')}")
    bench = summary.get("benchmark") or {}
    if bench.get("Benchmark_Status") == "ok":
        print(
            f">>> avg_global_step_ms={bench.get('Benchmark_Avg_Global_Step_ms')} "
            f"throughput={bench.get('Benchmark_Throughput_tok_s')} tok/s"
        )
    print(f">>> 已写入 {csv_path}")
    print(f">>> 合并表 {combined_path}")


def _run_fixed_global_batch_sweep(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)
    set_seed(args.seed)

    init_acc = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16")
    num_gpus = init_acc.num_processes
    pairs = _fixed_g_micro_accum_pairs(args.fixed_global_batch, num_gpus)

    if init_acc.is_main_process:
        init_acc.print(
            f"Fixed-G sweep | optimizer={args.optimizer} "
            f"fixed_global_batch={args.fixed_global_batch}"
        )
        init_acc.print(f"  num_gpus={num_gpus} candidates={pairs}")
        init_acc.print(
            f"  probe: warmup={args.warmup_steps} steps={args.probe_steps} | "
            f"benchmark: warmup={args.benchmark_warmup_steps} "
            f"steps={args.benchmark_steps}"
        )
    init_acc.wait_for_everyone()

    with init_acc.main_process_first():
        dataset = load_from_disk(DATA_PATH)
    dataset.set_format(type="torch", columns=["input_ids"])

    rows: list[dict] = []
    _find_max_micro_fixed_g(dataset, args, rows, num_gpus)

    rows.sort(key=lambda r: r["Micro_Batch"], reverse=True)
    sweep_meta = _summarize_fixed_g(rows)
    probe_max_ok = sweep_meta["probe_max_ok_micro"]

    summary = {
        "mode": "fixed_global_batch",
        "optimizer": args.optimizer,
        "fixed_global_batch": args.fixed_global_batch,
        "num_gpus": num_gpus,
        "seq_len": SEQ_LEN,
        "candidate_pairs": [{"micro": m, "grad_accum": a} for m, a in pairs],
        "warmup_steps": args.warmup_steps,
        "probe_steps": args.probe_steps,
        "benchmark_warmup_steps": args.benchmark_warmup_steps,
        "benchmark_steps": args.benchmark_steps,
        "benchmark_fallback": not args.no_benchmark_fallback,
        "benchmark_max_ok_micro": None,
        "benchmark_max_grad_accum": None,
        **sweep_meta,
        "rows": rows,
    }

    if args.benchmark and probe_max_ok is not None:
        if init_acc.is_main_process:
            init_acc.print(
                f"\n>>> fixed-G probe 完成 (max_micro={probe_max_ok})，开始 benchmark"
            )
        init_acc.wait_for_everyone()
        bench, benchmark_max_ok, benchmark_accum = _run_benchmark_fixed_g_with_fallback(
            dataset, args, rows, num_gpus, probe_max_ok
        )
        summary["benchmark"] = bench
        summary["benchmark_max_ok_micro"] = benchmark_max_ok
        summary["benchmark_max_grad_accum"] = benchmark_accum

    if init_acc.is_main_process:
        _write_fixed_g_results(args, summary, rows, sweep_meta)

    init_acc.wait_for_everyone()
    try:
        init_acc.free_memory()
    except Exception:
        pass


def main():
    args = parse_args()
    if args.fixed_global_batch is not None:
        if args.output_dir == DEFAULT_SAVE_DIR:
            args.output_dir = DEFAULT_FIXED_G_DIR
        _run_fixed_global_batch_sweep(args)
        return

    if args.orchestrate:
        _orchestrate_sweep(args)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    micro_list = _parse_micro_list(args)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16",
    )
    seed_everything(args.seed)
    set_seed(args.seed)

    if accelerator.is_main_process:
        accelerator.print(f"Micro-batch sweep | optimizer={args.optimizer}")
        accelerator.print(f"  num_gpus={accelerator.num_processes} grad_accum={args.grad_accum}")
        if args.benchmark_json_out:
            accelerator.print(f"  benchmark-only subprocess -> {args.benchmark_json_out}")
        elif args.probe_json_out:
            accelerator.print(f"  probe-only subprocess -> {args.probe_json_out}")
        elif micro_list is None:
            accelerator.print(
                f"  find_max_micro: {args.micro_min}..{args.micro_max} "
                f"mode={args.search_mode} warmup={args.warmup_steps} "
                f"probe={args.probe_steps}"
            )
        else:
            accelerator.print(
                f"  micro_list={micro_list} warmup={args.warmup_steps} "
                f"probe={args.probe_steps}"
            )
        if args.peak_budget_gb is not None:
            accelerator.print(f"  peak_budget={args.peak_budget_gb} GB")

    with accelerator.main_process_first():
        dataset = load_from_disk(DATA_PATH)
    dataset.set_format(type="torch", columns=["input_ids"])

    if args.benchmark_json_out:
        if not micro_list or len(micro_list) != 1:
            raise ValueError("--benchmark-json-out 需要 --micro-list 指定单个 micro")
        micro = micro_list[0]
        bench = _benchmark_micro(
            accelerator,
            dataset,
            args.optimizer,
            micro,
            args.grad_accum,
            args.warmup_steps,
            args.benchmark_steps,
            args.seed,
        )
        if accelerator.is_main_process:
            with open(args.benchmark_json_out, "w", encoding="utf-8") as f:
                json.dump(bench, f)
        accelerator.wait_for_everyone()
        return

    rows: list[dict] = []

    if micro_list is None:
        _run_find_max_micro(accelerator, dataset, args, rows)
    else:
        for micro in micro_list:
            _probe_one(accelerator, dataset, args, micro, rows, phase="list")

    if args.probe_json_out:
        if not rows:
            raise RuntimeError("--probe-json-out 未产生 probe 结果")
        if accelerator.is_main_process:
            with open(args.probe_json_out, "w", encoding="utf-8") as f:
                json.dump(rows[0], f)
        accelerator.wait_for_everyone()
        return

    rows.sort(key=lambda r: r["Micro_Batch"])
    sweep_meta = _summarize_sweep(rows, args.micro_max)
    probe_max_ok = sweep_meta["probe_max_ok_micro"]

    summary = {
        "optimizer": args.optimizer,
        "num_gpus": accelerator.num_processes,
        "find_max_micro": micro_list is None,
        "search_mode": args.search_mode if micro_list is None else None,
        "micro_min": args.micro_min,
        "micro_max": args.micro_max,
        "micro_list": sorted({r["Micro_Batch"] for r in rows}),
        "grad_accum": args.grad_accum,
        "peak_budget_gb": args.peak_budget_gb,
        "warmup_steps": args.warmup_steps,
        "probe_steps": args.probe_steps,
        "benchmark_steps": args.benchmark_steps,
        "benchmark_fallback": not args.no_benchmark_fallback,
        "benchmark_max_ok_micro": None,
        **sweep_meta,
        "rows": rows,
    }

    if args.benchmark and probe_max_ok is not None:
        if accelerator.is_main_process:
            accelerator.print(
                f"\n>>> probe 完成 (probe_max_ok={probe_max_ok})，"
                "已释放 GPU 缓存，开始 benchmark"
            )
        bench, benchmark_max_ok = _run_benchmark_with_fallback(
            accelerator, dataset, args, rows, probe_max_ok
        )
        summary["benchmark"] = bench
        summary["benchmark_max_ok_micro"] = benchmark_max_ok

    if accelerator.is_main_process:
        _write_sweep_results(args, rows, summary, sweep_meta)


if __name__ == "__main__":
    main()

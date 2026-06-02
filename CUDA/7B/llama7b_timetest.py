#!/usr/bin/env python3
"""
LLaMA-7B 预训练 optimizer 计时（DeepSpeed ZeRO-2, 默认随机初始化）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import set_seed

_7B_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_CUDA_DIR = os.path.abspath(os.path.join(_7B_DIR, ".."))
if _PARENT_CUDA_DIR not in sys.path:
    sys.path.insert(0, _PARENT_CUDA_DIR)
if _7B_DIR not in sys.path:
    sys.path.insert(0, _7B_DIR)

from llama7b_common import (  # noqa: E402
    ALL_OPTIMIZERS,
    DATA_PATH,
    GRAD_ACCUM_STEPS,
    LR,
    MODEL_ID,
    LLAMA_INIT,
    PARALLEL_TAG,
    SEQ_LEN,
    build_optimizer,
    create_accelerator,
    load_llama_model,
    seed_everything,
    sync_cuda,
    truncate_batch,
    _gather_peak_memory_gb,
    _gather_per_rank_gb,
    _optimizer_state_bytes,
    _unwrap_optimizer,
)

SAVE_DIR = os.path.join(_7B_DIR, "results")
PER_DEVICE_BATCH_SIZE = int(os.environ.get("LLAMA_MICRO_BATCH", "4"))
DEBUG_MODE = os.environ.get("TIMETEST_DEBUG", "1") == "1"
MEASURE_STEPS = int(os.environ.get("TIMETEST_MAX_STEPS", "40"))
WARMUP_STEPS = int(os.environ.get("TIMETEST_WARMUP_STEPS", "10"))
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="LLaMA-7B ZeRO-2 optimizer timetest")
    p.add_argument("--optimizer", choices=ALL_OPTIMIZERS, default=None)
    return p.parse_args()


def _is_mdq(opt_name: str) -> bool:
    return opt_name in ("MDQAdamW-Simple", "MDQAdamW-Simple-FusedIO")


def _avg(xs):
    return float(np.mean(xs)) if xs else 0.0


def _avg_on_flag(xs, flags):
    pairs = [x for x, f in zip(xs, flags) if f]
    return float(np.mean(pairs)) if pairs else 0.0


def _mdq_stats_summary(history):
    extra = history.get("mdq_extra_stats_ms", [])
    flags = history.get("mdq_update_decision", [])
    gs = history.get("mdq_grad_stats_ms", [])
    ar = history.get("mdq_allreduce_ms", [])
    score = history.get("mdq_score_bits_ms", [])
    update_count = sum(1 for f in flags if f)
    return {
        "Avg_MDQ_Extra_Stats_ms": _avg(extra),
        "Avg_MDQ_Extra_Stats_on_update_ms": _avg_on_flag(extra, flags),
        "Avg_MDQ_GradStats_on_update_ms": _avg_on_flag(gs, flags),
        "Avg_MDQ_AllReduce_on_update_ms": _avg_on_flag(ar, flags),
        "Avg_MDQ_ScoreBits_on_update_ms": _avg_on_flag(score, flags),
        "MDQ_Update_Steps_in_measure": update_count,
    }


def _new_history():
    return {
        "loss": [],
        "forward_ms": [],
        "backward_ms": [],
        "step_wall_ms": [],
        "optimizer_step_ms": [],
        "unpack_ms": [],
        "pack_ms": [],
        "mdq_extra_stats_ms": [],
        "mdq_grad_stats_ms": [],
        "mdq_allreduce_ms": [],
        "mdq_score_bits_ms": [],
        "mdq_update_decision": [],
        "mdq_fused_kernel_ms": [],
        "mdq_kernel_dequant_ms": [],
        "mdq_kernel_adam_mv_ms": [],
        "mdq_kernel_scale_reduce_ms": [],
        "mdq_kernel_quant_ms": [],
        "mdq_kernel_update_p_ms": [],
        "mdq_fused_io_fast_path_ratio": [],
    }


def _reset_peak_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _train_one_optimizer(opt_name, accelerator, train_dataloader):
    seed_everything(SEED)
    set_seed(SEED)

    global_batch = PER_DEVICE_BATCH_SIZE * accelerator.num_processes * GRAD_ACCUM_STEPS
    accelerator.print(f"\n{'=' * 50}\n>>> {opt_name}\n{'=' * 50}")
    accelerator.print(
        f"   model={MODEL_ID} init={LLAMA_INIT} parallel={PARALLEL_TAG} | "
        f"warmup={WARMUP_STEPS} measure={MEASURE_STEPS}"
    )

    model, config = load_llama_model()
    layer_count = getattr(config, "num_hidden_layers", 32)
    optimizer = build_optimizer(opt_name, model, LR, layer_count, global_batch)
    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dataloader)

    history = _new_history()
    phase = "warmup" if WARMUP_STEPS > 0 else "measure"
    global_step = 0
    measure_step = 0
    forward_accum_ms = 0.0
    backward_accum_ms = 0.0
    measure_wall_start = time.perf_counter() if phase == "measure" else None
    step_wall_start = time.perf_counter() if phase == "measure" else None
    optimizer_state_gb = None

    if phase == "measure":
        _reset_peak_memory_stats()

    stop_training = False
    data_iter = iter(train_dl)
    progress_bar = tqdm(
        disable=not accelerator.is_local_main_process,
        desc=f"{opt_name} [{phase}]",
    )

    while not stop_training:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dl)
            batch = next(data_iter)

        with accelerator.accumulate(model):
            input_ids = truncate_batch(batch["input_ids"], SEQ_LEN)
            sync_cuda()
            tf = time.perf_counter()
            outputs = model(input_ids, labels=input_ids)
            sync_cuda()
            forward_accum_ms += (time.perf_counter() - tf) * 1000.0

            sync_cuda()
            tb = time.perf_counter()
            accelerator.backward(outputs.loss)
            sync_cuda()
            backward_accum_ms += (time.perf_counter() - tb) * 1000.0

        if accelerator.sync_gradients:
            sync_cuda()
            t_opt = time.perf_counter()
            optimizer.step()
            sync_cuda()
            optimizer_step_ms = (time.perf_counter() - t_opt) * 1000.0
            optimizer.zero_grad()
            global_step += 1

            if phase == "warmup":
                if global_step >= WARMUP_STEPS:
                    phase = "measure"
                    history = _new_history()
                    forward_accum_ms = 0.0
                    backward_accum_ms = 0.0
                    measure_wall_start = time.perf_counter()
                    step_wall_start = time.perf_counter()
                    _reset_peak_memory_stats()
                    accelerator.print(f">>> Warmup 完成，开始计时")
                    progress_bar.set_description(f"{opt_name} [measure]")
                    continue
                forward_accum_ms = 0.0
                backward_accum_ms = 0.0
                progress_bar.update(1)
                continue

            step_wall_ms = (
                (time.perf_counter() - step_wall_start) * 1000.0
                if step_wall_start is not None
                else 0.0
            )
            step_loss = float(outputs.loss.item()) * GRAD_ACCUM_STEPS
            history["loss"].append(step_loss)
            history["forward_ms"].append(forward_accum_ms)
            history["backward_ms"].append(backward_accum_ms)
            history["step_wall_ms"].append(step_wall_ms)
            history["optimizer_step_ms"].append(optimizer_step_ms)

            if optimizer_state_gb is None:
                optimizer_state_gb = _gather_per_rank_gb(
                    accelerator,
                    _optimizer_state_bytes(optimizer) / (1024**3),
                )

            if _is_mdq(opt_name):
                timings = _unwrap_optimizer(optimizer).get_last_step_timings()
                history["unpack_ms"].append(timings.get("mdq_unpack_ms", 0.0))
                history["pack_ms"].append(timings.get("mdq_pack_ms", 0.0))
                history["mdq_fused_kernel_ms"].append(
                    timings.get("mdq_fused_kernel_ms", 0.0)
                )
                history["mdq_extra_stats_ms"].append(
                    timings.get("mdq_extra_stats_ms", 0.0)
                )
                history["mdq_grad_stats_ms"].append(
                    timings.get("mdq_grad_stats_ms", 0.0)
                )
                history["mdq_allreduce_ms"].append(
                    timings.get("mdq_allreduce_ms", 0.0)
                )
                history["mdq_score_bits_ms"].append(
                    timings.get("mdq_score_bits_ms", 0.0)
                )
                history["mdq_update_decision"].append(
                    bool(timings.get("mdq_update_decision", False))
                )
            else:
                history["unpack_ms"].append(0.0)
                history["pack_ms"].append(0.0)

            progress_bar.update(1)
            measure_step += 1
            if measure_step >= MEASURE_STEPS:
                stop_training = True
            else:
                step_wall_start = time.perf_counter()
            forward_accum_ms = 0.0
            backward_accum_ms = 0.0

    progress_bar.close()

    measure_wall_ms = (
        (time.perf_counter() - measure_wall_start) * 1000.0
        if measure_wall_start is not None
        else 0.0
    )
    peak_mem_gb = _gather_peak_memory_gb(accelerator)
    tokens_per_step = (
        accelerator.num_processes * PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS * SEQ_LEN
    )
    throughput = (
        measure_step * tokens_per_step / (measure_wall_ms / 1000.0)
        if measure_wall_ms > 0
        else 0.0
    )

    if accelerator.is_main_process:
        mdq_summary = _mdq_stats_summary(history) if _is_mdq(opt_name) else {}
        result = {
            "Opt": opt_name,
            "Model": "Llama-7B",
            "Model_ID": MODEL_ID,
            "Init": LLAMA_INIT,
            "Parallel": PARALLEL_TAG,
            "Avg_Loss": _avg(history["loss"]),
            "Measure_Train_Wall_ms": measure_wall_ms,
            "Throughput_tokens_per_s": throughput,
            "Peak_GPU_Memory_GB": peak_mem_gb,
            "Optimizer_State_GB": optimizer_state_gb or {},
            "Measured_Steps": measure_step,
            "Num_GPUs": accelerator.num_processes,
            "History": history,
            **{k: v for k, v in mdq_summary.items() if v is not None},
        }
        json_path = os.path.join(SAVE_DIR, f"history_{opt_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        accelerator.print(
            f">>> [{opt_name}] loss={result['Avg_Loss']:.4f} "
            f"throughput={throughput:.0f} tok/s -> {json_path}"
        )

    del model, optimizer
    torch.cuda.empty_cache()
    accelerator.free_memory()


def run_experiment(optimizer_filter=None):
    accelerator = create_accelerator(grad_accumulation_steps=GRAD_ACCUM_STEPS)
    seed_everything(SEED)
    set_seed(SEED)

    opts = [optimizer_filter] if optimizer_filter else ALL_OPTIMIZERS
    accelerator.print(f"GPUs={accelerator.num_processes} init={LLAMA_INIT} ZeRO-2")
    if accelerator.is_main_process:
        os.makedirs(SAVE_DIR, exist_ok=True)

    with accelerator.main_process_first():
        dataset = load_from_disk(DATA_PATH)
    dataset.set_format(type="torch", columns=["input_ids"])
    train_dataloader = DataLoader(
        dataset,
        batch_size=PER_DEVICE_BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    for opt_name in opts:
        _train_one_optimizer(opt_name, accelerator, train_dataloader)


if __name__ == "__main__":
    args = parse_args()
    run_experiment(optimizer_filter=args.optimizer)

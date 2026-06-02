"""
GPT2-XL 预训练 optimizer 计时 benchmark（对齐 baseline/pretrain_gpt.py 配置）。

MDQAdamW-Simple / MDQAdamW-Simple-FusedIO + AdamW-32bit / 8bit-Adam-bnb。
公平计时：warmup + 单 optimizer 独立进程（见 run_build_and_train.sh）。
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from accelerate import Accelerator
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, set_seed

_CUDA_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUDA_DIR not in sys.path:
    sys.path.insert(0, _CUDA_DIR)

from encoder_layer_backward_profiler import EncoderLayerBackwardProfiler
from stquant_opt_fused_io import MDQAdamWSimpleFusedIO
from stquant_opt_simple import MDQAdamWSimple

try:
    from adam_mini import Adam_mini
except ImportError:
    Adam_mini = None

try:
    from galore_torch import GaLoreAdamW
except ImportError:
    GaLoreAdamW = None

GALORE_RANK = int(os.environ.get("GALORE_RANK", "128"))
GALORE_UPDATE_PROJ_GAP = int(os.environ.get("GALORE_UPDATE_PROJ_GAP", "200"))
GALORE_SCALE = float(os.environ.get("GALORE_SCALE", "0.25"))
GALORE_PROJ_TYPE = os.environ.get("GALORE_PROJ_TYPE", "std")

# ================= 配置（对齐 baseline/pretrain_gpt.py）=================
MODEL_TYPE = "gpt2-xl"
DATA_PATH = "/workspace/data/openwebtext_processed"
PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 16
SEQ_LEN = 1024
LR = 2e-4
WEIGHT_DECAY = 0.01
SAVE_DIR = os.path.join(_CUDA_DIR, "results")

ALL_OPTIMIZERS = [
    "MDQAdamW-Simple",
    "MDQAdamW-Simple-FusedIO",
    "AdamW-32bit",
    "8bit-Adam-bnb",
    "GaLore",
    "Adam-mini",
]

DEBUG_MODE = os.environ.get("TIMETEST_DEBUG", "1") == "1"
MEASURE_STEPS = int(os.environ.get("TIMETEST_MAX_STEPS", "40"))
WARMUP_STEPS = int(os.environ.get("TIMETEST_WARMUP_STEPS", "10"))
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="GPT2-XL optimizer 计时实验")
    parser.add_argument(
        "--optimizer",
        choices=ALL_OPTIMIZERS,
        default=None,
        help="仅跑指定优化器（独立进程模式）；省略则跑全部",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _mdq_common_kwargs(lr, layer_count, global_batch):
    return dict(
        lr=lr,
        weight_decay=WEIGHT_DECAY,
        layer_count=layer_count,
        batch_size=global_batch,
        block_size=256,
        update_freq=20,
    )


def _unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def build_adam_mini(model, config: GPT2Config, lr: float):
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


def build_galore_optimizer(model, lr: float):
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


def build_optimizer(opt_name, model, lr, layer_count, global_batch):
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if opt_name == "AdamW-32bit":
        return torch.optim.AdamW(params, lr=lr)
    if opt_name == "8bit-Adam-bnb":
        import bitsandbytes as bnb

        return bnb.optim.Adam8bit(params, lr=lr)
    if opt_name == "Adam-mini":
        config = GPT2Config.from_pretrained(MODEL_TYPE)
        return build_adam_mini(model, config, lr)
    if opt_name == "GaLore":
        return build_galore_optimizer(model, lr)
    mdq_kw = _mdq_common_kwargs(lr, layer_count, global_batch)
    if opt_name == "MDQAdamW-Simple":
        return MDQAdamWSimple(model.parameters(), **mdq_kw)
    if opt_name == "MDQAdamW-Simple-FusedIO":
        return MDQAdamWSimpleFusedIO(model.parameters(), **mdq_kw)
    raise ValueError(f"未知优化器: {opt_name}")


def _is_mdq(opt_name: str) -> bool:
    return opt_name in ("MDQAdamW-Simple", "MDQAdamW-Simple-FusedIO")


def _avg(xs):
    return float(np.mean(xs)) if xs else 0.0


def _avg_on_flag(xs, flags):
    pairs = [x for x, f in zip(xs, flags) if f]
    return float(np.mean(pairs)) if pairs else 0.0


def _mdq_stats_summary(history):
    """汇总 MDQ 额外统计量计时（含 update step 子集）。"""
    extra = history.get("mdq_extra_stats_ms", [])
    flags = history.get("mdq_update_decision", [])
    grad = history.get("mdq_grad_stats_ms", [])
    ar = history.get("mdq_allreduce_ms", [])
    score = history.get("mdq_score_bits_ms", [])
    update_count = sum(1 for f in flags if f)
    avg_extra = _avg(extra)
    return {
        "Avg_MDQ_Extra_Stats_ms": avg_extra,
        "Avg_MDQ_Extra_Stats_on_update_ms": _avg_on_flag(extra, flags),
        "Avg_MDQ_GradStats_on_update_ms": _avg_on_flag(grad, flags),
        "Avg_MDQ_AllReduce_on_update_ms": _avg_on_flag(ar, flags),
        "Avg_MDQ_ScoreBits_on_update_ms": _avg_on_flag(score, flags),
        "MDQ_Update_Steps_in_measure": update_count,
    }


def _unwrap_optimizer(optimizer):
    if hasattr(optimizer, "optimizer"):
        return optimizer.optimizer
    return optimizer


def _optimizer_state_bytes(optimizer) -> int:
    """optimizer.state 中 GPU tensor 的字节数（静态 state 占用）。"""
    opt = _unwrap_optimizer(optimizer)
    total = 0
    for st in opt.state.values():
        for v in st.values():
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
    return total


def _gather_per_rank_gb(accelerator, local_value_gb: float) -> dict:
    """将各 rank 的标量（GB）all_gather 后按 gpu id 命名。"""
    if not torch.cuda.is_available():
        return {}
    n = accelerator.num_processes
    if n > 1 and dist.is_initialized():
        t = torch.tensor([local_value_gb], dtype=torch.float64, device=accelerator.device)
        gathered = [torch.zeros_like(t) for _ in range(n)]
        dist.all_gather(gathered, t)
        values = [g.item() for g in gathered]
    else:
        values = [local_value_gb]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_ids = [x.strip() for x in visible.split(",") if x.strip()]
    out = {}
    for rank, gb in enumerate(values):
        if rank < len(visible_ids):
            key = f"gpu_{visible_ids[rank]}"
        else:
            key = f"rank_{rank}"
        out[key] = gb
    return out


def _new_history():
    return {
        "loss": [],
        "forward_ms": [],
        "backward_ms": [],
        "step_wall_ms": [],
        "layer_backward_per_step": [],
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


def _gather_peak_memory_gb(accelerator) -> dict:
    """收集每个 rank 在 measure 阶段的峰值显存（GB，含模型+激活+梯度+优化器）。"""
    if not torch.cuda.is_available():
        return {}
    local_peak = torch.cuda.max_memory_allocated() / (1024**3)
    return _gather_per_rank_gb(accelerator, local_peak)


def _train_one_optimizer(opt_name, accelerator, train_dataloader):
    seed_everything(SEED)
    set_seed(SEED)

    run_index = os.environ.get("TIMETEST_RUN_INDEX", "")
    global_batch = PER_DEVICE_BATCH_SIZE * accelerator.num_processes

    accelerator.print(f"\n{'=' * 50}\n>>> {opt_name} (run_index={run_index})\n{'=' * 50}")
    accelerator.print(
        f"   Warmup steps (不计时): {WARMUP_STEPS} | "
        f"Measure steps: {MEASURE_STEPS} | DEBUG={DEBUG_MODE}"
    )

    config = GPT2Config.from_pretrained(MODEL_TYPE)
    model = GPT2LMHeadModel(config)
    model.gradient_checkpointing_enable()

    optimizer = build_optimizer(
        opt_name, model, LR, config.n_layer, global_batch
    )
    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dataloader)

    layer_profiler = EncoderLayerBackwardProfiler(model).register()
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
            input_ids = batch["input_ids"]
            sync_cuda()
            tf = time.perf_counter()
            outputs = model(input_ids, labels=input_ids)
            sync_cuda()
            forward_accum_ms += (time.perf_counter() - tf) * 1000.0

            layer_profiler.begin_backward_pass()
            sync_cuda()
            tb = time.perf_counter()
            accelerator.backward(outputs.loss)
            sync_cuda()
            backward_accum_ms += (time.perf_counter() - tb) * 1000.0
            layer_profiler.end_backward_pass()

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
                    layer_profiler.remove()
                    layer_profiler = EncoderLayerBackwardProfiler(model).register()
                    history = _new_history()
                    forward_accum_ms = 0.0
                    backward_accum_ms = 0.0
                    measure_wall_start = time.perf_counter()
                    step_wall_start = time.perf_counter()
                    _reset_peak_memory_stats()
                    accelerator.print(
                        f">>> Warmup 完成 ({WARMUP_STEPS} steps)，开始正式计时"
                    )
                    progress_bar.set_description(f"{opt_name} [measure]")
                    continue
                forward_accum_ms = 0.0
                backward_accum_ms = 0.0
                progress_bar.set_description(f"{opt_name} [warmup]")
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
            snap = layer_profiler.flush_optimizer_step()
            history["layer_backward_per_step"].append(
                {str(k): v for k, v in snap.items()}
            )
            forward_accum_ms = 0.0
            backward_accum_ms = 0.0

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
                history["mdq_kernel_dequant_ms"].append(
                    timings.get("mdq_kernel_dequant_ms", 0.0)
                )
                history["mdq_kernel_adam_mv_ms"].append(
                    timings.get("mdq_kernel_adam_mv_ms", 0.0)
                )
                history["mdq_kernel_scale_reduce_ms"].append(
                    timings.get("mdq_kernel_scale_reduce_ms", 0.0)
                )
                history["mdq_kernel_quant_ms"].append(
                    timings.get("mdq_kernel_quant_ms", 0.0)
                )
                history["mdq_kernel_update_p_ms"].append(
                    timings.get("mdq_kernel_update_p_ms", 0.0)
                )
                if opt_name == "MDQAdamW-Simple-FusedIO":
                    history["mdq_fused_io_fast_path_ratio"].append(
                        timings.get("mdq_fused_io_fast_path_ratio", 0.0)
                    )
            else:
                history["unpack_ms"].append(0.0)
                history["pack_ms"].append(0.0)

            postfix = {
                "loss": f"{step_loss:.4f}",
                "step_ms": f"{step_wall_ms:.1f}",
                "fwd_ms": f"{history['forward_ms'][-1]:.1f}",
                "bwd_ms": f"{history['backward_ms'][-1]:.1f}",
                "opt_ms": f"{history['optimizer_step_ms'][-1]:.1f}",
                "mstep": measure_step + 1,
            }
            if _is_mdq(opt_name) and history["mdq_extra_stats_ms"]:
                postfix["mdq_st"] = f"{history['mdq_extra_stats_ms'][-1]:.1f}"
            progress_bar.set_postfix(postfix)
            progress_bar.update(1)

            measure_step += 1
            if measure_step >= MEASURE_STEPS:
                accelerator.print(f"正式计时停止于 {MEASURE_STEPS} steps")
                stop_training = True
            else:
                step_wall_start = time.perf_counter()

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
    total_tokens = measure_step * tokens_per_step
    throughput = (
        total_tokens / (measure_wall_ms / 1000.0) if measure_wall_ms > 0 else 0.0
    )

    if accelerator.is_main_process:

        layer_avg = layer_profiler.get_layer_avg_ms()
        layer_mean, _layer_sum = layer_profiler.get_summary_scalars()
        avg_forward = _avg(history["forward_ms"])
        avg_backward = _avg(history["backward_ms"])
        avg_step_wall = _avg(history["step_wall_ms"])
        mdq_summary = _mdq_stats_summary(history) if _is_mdq(opt_name) else {}

        result = {
            "Opt": opt_name,
            "Avg_Loss": _avg(history["loss"]),
            "Measure_Train_Wall_ms": measure_wall_ms,
            "Avg_Step_Wall_ms": avg_step_wall,
            "Throughput_tokens_per_s": throughput,
            "Peak_GPU_Memory_GB": peak_mem_gb,
            "Optimizer_State_GB": optimizer_state_gb or {},
            "Warmup_Steps": WARMUP_STEPS,
            "Measured_Steps": measure_step,
            "Run_Index": run_index,
            "Num_GPUs": accelerator.num_processes,
            "Avg_Forward_ms": avg_forward,
            "Avg_Backward_ms": avg_backward,
            "Avg_LayerBackward_Mean_ms": layer_mean,
            "LayerBackward_Avg_ms": {str(k): v for k, v in layer_avg.items()},
            "Avg_OptimizerStep_ms": _avg(history["optimizer_step_ms"]),
            "Avg_Unpack_ms": _avg(history["unpack_ms"]),
            "Avg_Pack_ms": _avg(history["pack_ms"]),
            "Avg_MDQ_FusedKernel_ms": _avg(history["mdq_fused_kernel_ms"])
            if _is_mdq(opt_name)
            else None,
            "Avg_Kernel_Dequant_ms": _avg(history["mdq_kernel_dequant_ms"])
            if _is_mdq(opt_name)
            else None,
            "Avg_Kernel_Adam_mv_ms": _avg(history["mdq_kernel_adam_mv_ms"])
            if _is_mdq(opt_name)
            else None,
            "Avg_Kernel_ScaleReduce_ms": _avg(history["mdq_kernel_scale_reduce_ms"])
            if _is_mdq(opt_name)
            else None,
            "Avg_Kernel_Quant_ms": _avg(history["mdq_kernel_quant_ms"])
            if _is_mdq(opt_name)
            else None,
            "Avg_Kernel_UpdateP_ms": _avg(history["mdq_kernel_update_p_ms"])
            if _is_mdq(opt_name)
            else None,
            "Avg_MDQ_Extra_Stats_ms": mdq_summary.get("Avg_MDQ_Extra_Stats_ms")
            if _is_mdq(opt_name)
            else None,
            "Avg_MDQ_Extra_Stats_on_update_ms": mdq_summary.get(
                "Avg_MDQ_Extra_Stats_on_update_ms"
            )
            if _is_mdq(opt_name)
            else None,
            "Avg_MDQ_GradStats_on_update_ms": mdq_summary.get(
                "Avg_MDQ_GradStats_on_update_ms"
            )
            if _is_mdq(opt_name)
            else None,
            "Avg_MDQ_AllReduce_on_update_ms": mdq_summary.get(
                "Avg_MDQ_AllReduce_on_update_ms"
            )
            if _is_mdq(opt_name)
            else None,
            "Avg_MDQ_ScoreBits_on_update_ms": mdq_summary.get(
                "Avg_MDQ_ScoreBits_on_update_ms"
            )
            if _is_mdq(opt_name)
            else None,
            "MDQ_Update_Steps_in_measure": mdq_summary.get("MDQ_Update_Steps_in_measure")
            if _is_mdq(opt_name)
            else None,
            "Avg_FusedIO_FastPath_Ratio": _avg(history["mdq_fused_io_fast_path_ratio"])
            if opt_name == "MDQAdamW-Simple-FusedIO"
            else None,
            "History": history,
        }

        json_path = os.path.join(SAVE_DIR, f"history_{opt_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        layer_rows = [
            {
                "Optimizer": opt_name,
                "Layer": i,
                "Avg_Backward_ms": f"{layer_avg.get(i, 0.0):.4f}",
            }
            for i in range(layer_profiler.num_layers)
        ]
        layer_csv = os.path.join(SAVE_DIR, f"layer_backward_{opt_name}.csv")
        pd.DataFrame(layer_rows).to_csv(layer_csv, index=False)

        step_rows = []
        for i in range(len(history["loss"])):
            row = {
                "Optimizer": opt_name,
                "Step": i + 1,
                "Loss": history["loss"][i],
                "Step_Wall_ms": history["step_wall_ms"][i],
                "Forward_ms": history["forward_ms"][i],
                "Backward_ms": history["backward_ms"][i],
                "OptimizerStep_ms": history["optimizer_step_ms"][i],
            }
            if _is_mdq(opt_name) and i < len(history.get("mdq_extra_stats_ms", [])):
                row["MDQ_UpdateStep"] = int(history["mdq_update_decision"][i])
                row["MDQ_ExtraStats_ms"] = history["mdq_extra_stats_ms"][i]
                row["MDQ_GradStats_ms"] = history["mdq_grad_stats_ms"][i]
                row["MDQ_AllReduce_ms"] = history["mdq_allreduce_ms"][i]
                row["MDQ_ScoreBits_ms"] = history["mdq_score_bits_ms"][i]
            step_rows.append(row)
        step_csv = os.path.join(SAVE_DIR, f"step_timing_{opt_name}.csv")
        pd.DataFrame(step_rows).to_csv(step_csv, index=False)

        accelerator.print(
            f">>> [{opt_name}] Avg_Loss={result['Avg_Loss']:.4f} | "
            f"Wall={measure_wall_ms:.0f}ms | StepWall={avg_step_wall:.0f}ms | "
            f"Throughput={throughput:.0f} tok/s"
        )
        if _is_mdq(opt_name) and mdq_summary:
            accelerator.print(
                f">>> MDQ stats: avg={mdq_summary['Avg_MDQ_Extra_Stats_ms']:.2f}ms | "
                f"on_update={mdq_summary['Avg_MDQ_Extra_Stats_on_update_ms']:.2f}ms "
                f"(grad={mdq_summary['Avg_MDQ_GradStats_on_update_ms']:.2f}, "
                f"ar={mdq_summary['Avg_MDQ_AllReduce_on_update_ms']:.2f}, "
                f"score={mdq_summary['Avg_MDQ_ScoreBits_on_update_ms']:.2f}) | "
                f"update_steps={mdq_summary['MDQ_Update_Steps_in_measure']}"
            )
        accelerator.print(f">>> 已保存 {json_path}、{layer_csv}、{step_csv}")

    layer_profiler.remove()
    del model, optimizer
    torch.cuda.empty_cache()
    accelerator.free_memory()


def run_experiment(optimizer_filter=None):
    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        mixed_precision="bf16",
    )

    seed_everything(SEED)
    set_seed(SEED)

    opts = [optimizer_filter] if optimizer_filter else ALL_OPTIMIZERS

    accelerator.print(f"🖥️  GPUs: {accelerator.num_processes}")
    accelerator.print(f"📁  SAVE_DIR: {SAVE_DIR}")
    accelerator.print(f"📋  Optimizers: {opts}")
    accelerator.print(
        f"📦  batch/GPU={PER_DEVICE_BATCH_SIZE} accum={GRAD_ACCUM_STEPS} seq={SEQ_LEN}"
    )

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
        num_workers=4,
        persistent_workers=True,
    )

    for opt_name in opts:
        _train_one_optimizer(opt_name, accelerator, train_dataloader)

    if accelerator.is_main_process:
        print("\n" + "=" * 50)
        print("GPT2-XL timetest 完成")
        print(f"结果目录: {SAVE_DIR}")
        print("请运行 aggregate_timing.py 生成 summary_metrics.csv")
        print("=" * 50)


if __name__ == "__main__":
    args = parse_args()
    run_experiment(optimizer_filter=args.optimizer)

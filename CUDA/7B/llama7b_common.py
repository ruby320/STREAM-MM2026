"""
LLaMA-7B 预训练 benchmark 共享：随机初始化 / ZeRO-2 / optimizer 构建。
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from transformers import AutoConfig, AutoModelForCausalLM

_7B_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_CUDA_DIR = os.path.abspath(os.path.join(_7B_DIR, ".."))
if _PARENT_CUDA_DIR not in sys.path:
    sys.path.insert(0, _PARENT_CUDA_DIR)

from stquant_opt_fused_io import MDQAdamWSimpleFusedIO  # noqa: E402
from stquant_opt_simple import MDQAdamWSimple  # noqa: E402

try:
    from adam_mini import Adam_mini
except ImportError:
    Adam_mini = None

try:
    from galore_torch import GaLoreAdamW
except ImportError:
    GaLoreAdamW = None

# ================= 配置 =================
# 仅架构：config 来自 MODEL_ID 或本地目录；权重默认随机初始化
MODEL_ID = os.environ.get("LLAMA_MODEL_ID", "meta-llama/Llama-2-7b-hf")
LLAMA_INIT = os.environ.get("LLAMA_INIT", "random").strip().lower()
DATA_PATH = os.environ.get(
    "LLAMA_DATA_PATH", "/workspace/data/openwebtext_llama7b_1024"
)
SEQ_LEN = int(os.environ.get("LLAMA_SEQ_LEN", "1024"))
LR = float(os.environ.get("LLAMA_LR", "2e-4"))
WEIGHT_DECAY = float(os.environ.get("LLAMA_WEIGHT_DECAY", "0.01"))
GRAD_ACCUM_STEPS = int(os.environ.get("LLAMA_GRAD_ACCUM", "16"))
PARALLEL_TAG = "DeepSpeed-ZeRO-2"
MODEL_LABEL = "Llama-7B"

DS_CONFIG_PATH = os.path.join(_7B_DIR, "ds_zero2.json")

ALL_OPTIMIZERS = [
    "MDQAdamW-Simple",
    "MDQAdamW-Simple-FusedIO",
    "AdamW-32bit",
    "8bit-Adam-bnb",
    "GaLore",
    "Adam-mini",
]

GALORE_RANK = int(os.environ.get("GALORE_RANK", "128"))
GALORE_UPDATE_PROJ_GAP = int(os.environ.get("GALORE_UPDATE_PROJ_GAP", "200"))
GALORE_SCALE = float(os.environ.get("GALORE_SCALE", "0.25"))
GALORE_PROJ_TYPE = os.environ.get("GALORE_PROJ_TYPE", "std")

LLAMA_NO_DECAY_SUFFIXES = ("bias", "norm.weight", "layernorm.weight")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def create_setup_accelerator() -> Accelerator:
    """无 DeepSpeed：仅用于 fixed-G 里加载 dataset / 同步 / 打印。

    Accelerate 每个进程只允许一个带 deepspeed_plugin 的 Accelerator；
    若 setup 阶段也创建 DeepSpeed Accelerator，后续 probe 会 NotImplementedError。
    """
    return Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16")


def create_accelerator(grad_accumulation_steps: int = 1) -> Accelerator:
    if not os.path.isfile(DS_CONFIG_PATH):
        raise FileNotFoundError(f"缺少 DeepSpeed 配置: {DS_CONFIG_PATH}")
    plugin = DeepSpeedPlugin(hf_ds_config=DS_CONFIG_PATH)
    return Accelerator(
        gradient_accumulation_steps=grad_accumulation_steps,
        mixed_precision="bf16",
        deepspeed_plugin=plugin,
    )


def reset_accelerator_singleton_state() -> None:
    """释放 DeepSpeed Accelerator 后重置 Accelerate 单例，以便同进程探测下一档 micro。"""
    from accelerate.state import AcceleratorState

    AcceleratorState._reset_state(reset_partial_state=False)


def load_llama_config():
    return AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)


def load_llama_model():
    """LLAMA_INIT=random：仅架构随机初始化；pretrained：加载 HF 权重。"""
    config = load_llama_config()
    if LLAMA_INIT == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            config=config,
            trust_remote_code=True,
        )
    elif LLAMA_INIT == "random":
        model = AutoModelForCausalLM.from_config(config)
    else:
        raise ValueError(f"未知 LLAMA_INIT={LLAMA_INIT!r}，应为 random 或 pretrained")
    model.gradient_checkpointing_enable()
    return model, config


def _unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def _unwrap_optimizer(optimizer):
    if hasattr(optimizer, "optimizer"):
        return optimizer.optimizer
    return optimizer


def _optimizer_state_bytes(optimizer) -> int:
    opt = _unwrap_optimizer(optimizer)
    total = 0
    for st in opt.state.values():
        for v in st.values():
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
    return total


def _gather_per_rank_gb(accelerator, local_value_gb: float) -> dict:
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


def _gather_peak_memory_gb(accelerator) -> dict:
    if not torch.cuda.is_available():
        return {}
    local_peak = torch.cuda.max_memory_allocated() / (1024**3)
    return _gather_per_rank_gb(accelerator, local_peak)


def sweep_meta_fields() -> dict:
    return {
        "Model": MODEL_LABEL,
        "Model_ID": MODEL_ID,
        "Init": LLAMA_INIT,
        "Parallel": PARALLEL_TAG,
    }


def _mdq_common_kwargs(lr, layer_count, global_batch):
    return dict(
        lr=lr,
        weight_decay=WEIGHT_DECAY,
        layer_count=layer_count,
        batch_size=global_batch,
        block_size=256,
        update_freq=20,
    )


def _param_no_decay(name: str) -> bool:
    n = name.lower()
    return any(s in n for s in LLAMA_NO_DECAY_SUFFIXES)


def build_adam_mini(model, config, lr: float):
    if Adam_mini is None:
        raise ImportError("请先安装 adam-mini: pip install adam-mini")
    m = _unwrap_model(model)
    n_heads = getattr(config, "num_attention_heads", None)
    n_kv = getattr(config, "num_key_value_heads", None)
    dim = getattr(config, "hidden_size", None)
    optimizer = Adam_mini(
        named_parameters=m.named_parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv,
    )
    optimizer.wqk_names.update(["q_proj", "k_proj"])
    optimizer.wv_names.add("v_proj")
    optimizer.attn_proj_names.add("o_proj")
    return optimizer


def build_galore_optimizer(model, lr: float):
    if GaLoreAdamW is None:
        raise ImportError("请先安装 galore-torch: pip install galore-torch")
    m = _unwrap_model(model)
    galore_params = []
    non_galore_params = []
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and not _param_no_decay(name):
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
    params = [
        {
            "params": [
                p for n, p in model.named_parameters() if not _param_no_decay(n)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if _param_no_decay(n)
            ],
            "weight_decay": 0.0,
        },
    ]

    if opt_name == "AdamW-32bit":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    if opt_name == "8bit-Adam-bnb":
        import bitsandbytes as bnb

        return bnb.optim.Adam8bit(params, lr=lr)
    if opt_name == "Adam-mini":
        config = load_llama_config()
        return build_adam_mini(model, config, lr)
    if opt_name == "GaLore":
        return build_galore_optimizer(model, lr)
    mdq_kw = _mdq_common_kwargs(lr, layer_count, global_batch)
    if opt_name == "MDQAdamW-Simple":
        return MDQAdamWSimple(model.parameters(), **mdq_kw)
    if opt_name == "MDQAdamW-Simple-FusedIO":
        return MDQAdamWSimpleFusedIO(model.parameters(), **mdq_kw)
    raise ValueError(f"未知优化器: {opt_name}")


def truncate_batch(input_ids: torch.Tensor, seq_len: int = SEQ_LEN) -> torch.Tensor:
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.size(1) > seq_len:
        return input_ids[:, :seq_len].contiguous()
    if input_ids.size(1) < seq_len:
        pad = seq_len - input_ids.size(1)
        return torch.nn.functional.pad(input_ids, (0, pad), value=0)
    return input_ids

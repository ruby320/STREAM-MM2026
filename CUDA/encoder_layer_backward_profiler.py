"""
按 GPT-2 transformer.h.{i} 模块统计 backward 耗时（pre/post backward hook + CUDA sync）。
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _unwrap_model(model: nn.Module) -> nn.Module:
    m = model
    while hasattr(m, "module"):
        m = m.module
    return m


def _sync_cuda(device: Optional[torch.device] = None) -> None:
    if torch.cuda.is_available():
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        else:
            torch.cuda.synchronize()


class EncoderLayerBackwardProfiler:
    """
    在 transformer.h[i] 上挂 backward pre/post hook，
    统计单次 loss.backward() 内各 block 墙钟时间（毫秒），
    再在一个 optimizer step（含 grad accum）内累加。
    """

    def __init__(self, model: nn.Module):
        self.base = _unwrap_model(model)
        self.layers: nn.ModuleList = self.base.transformer.h
        self.num_layers = len(self.layers)
        self._handles: List = []
        self._starts: Dict[int, float] = {}
        self._pass_ms: Dict[int, float] = {}
        self._step_ms: Dict[int, float] = {}
        self._history_per_step: List[Dict[int, float]] = []

    def register(self) -> "EncoderLayerBackwardProfiler":
        self.remove()
        for idx, layer in enumerate(self.layers):
            def _pre(_mod, _grad_output, layer_idx=idx):
                dev = next(_mod.parameters()).device
                _sync_cuda(dev)
                self._starts[layer_idx] = time.perf_counter()

            def _post(_mod, _gin, _gout, layer_idx=idx):
                dev = next(_mod.parameters()).device
                _sync_cuda(dev)
                t0 = self._starts.pop(layer_idx, None)
                if t0 is None:
                    return
                elapsed = (time.perf_counter() - t0) * 1000.0
                self._pass_ms[layer_idx] = self._pass_ms.get(layer_idx, 0.0) + elapsed

            self._handles.append(layer.register_full_backward_pre_hook(_pre))
            self._handles.append(layer.register_full_backward_hook(_post))
        return self

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._starts.clear()

    def begin_backward_pass(self) -> None:
        self._pass_ms = {}
        self._starts = {}

    def end_backward_pass(self) -> None:
        for idx, ms in self._pass_ms.items():
            self._step_ms[idx] = self._step_ms.get(idx, 0.0) + ms

    def flush_optimizer_step(self) -> Dict[int, float]:
        snapshot = dict(self._step_ms)
        self._history_per_step.append(snapshot)
        self._step_ms = {}
        self._pass_ms = {}
        self._starts = {}
        return snapshot

    def get_layer_avg_ms(self) -> Dict[int, float]:
        if not self._history_per_step:
            return {i: 0.0 for i in range(self.num_layers)}
        sums = {i: 0.0 for i in range(self.num_layers)}
        for snap in self._history_per_step:
            for i, ms in snap.items():
                sums[i] = sums.get(i, 0.0) + ms
        n = len(self._history_per_step)
        return {i: sums.get(i, 0.0) / n for i in range(self.num_layers)}

    def get_summary_scalars(self) -> Tuple[float, float]:
        """Returns mean_of_layer_avgs, sum_of_layer_avgs."""
        avgs = self.get_layer_avg_ms()
        if not avgs:
            return 0.0, 0.0
        mean_v = float(sum(avgs.values()) / len(avgs))
        sum_v = float(sum(avgs.values()))
        return mean_v, sum_v

"""
MDQAdamW-Simple-FusedIO: 在 Simple 基础上，当 momentum 存储布局已与 current_bit 一致时，
跳过 Python 的 _align_momentum_storage / _pack_momentum_storage。

4bit packed 的编解码已在 fused_mdq_adamw_kernel_u8_packed4 内完成；
8/16/32bit 在 state 已是 kernel 所需 dtype 时也可走快路径。布局不匹配时回退 Simple。
"""
import time

import stquant_timetest_cpp
import torch

from stquant_opt_simple import MDQAdamWSimple


class MDQAdamWSimpleFusedIO(MDQAdamWSimple):
    """MDQ Simple + 免 Python pack/unpack 快路径（存储布局已对齐时）。"""

    @classmethod
    def _storage_matches_bits(cls, state, bits: int) -> bool:
        if "exp_avg" not in state:
            return False
        ea, easq = state["exp_avg"], state.get("exp_avg_sq")
        if bits == 4:
            return cls._is_momentum_packed4(state)
        if bits == 8:
            return (
                ea.dtype == torch.int8
                and easq is not None
                and easq.dtype == torch.uint8
                and easq.numel() == ea.numel()
            )
        if bits == 16:
            return ea.dtype == torch.int16 and easq is not None and easq.dtype == torch.uint16
        if bits >= 32:
            return ea.dtype == torch.int32 and easq is not None and easq.dtype == torch.uint32
        return False

    def _launch_fused_update(self, p, group):
        state = self.state[p]
        bits = int(state["current_bit"])
        bsz = min(int(group["block_size"]), 1024)

        state["step"] += 1

        if not self._storage_matches_bits(state, bits):
            self.last_step_timings["mdq_fused_io_fallback_params"] = (
                self.last_step_timings.get("mdq_fused_io_fallback_params", 0) + 1
            )
            return super()._launch_fused_update(p, group)

        self.last_step_timings["mdq_fused_io_fast_path_params"] = (
            self.last_step_timings.get("mdq_fused_io_fast_path_params", 0) + 1
        )

        if p.is_cuda:
            torch.cuda.synchronize(p.device)
        tk0 = time.perf_counter()
        profile_buf = self._kernel_profile_buf_for(p.device)
        stquant_timetest_cpp.fused_mdq_adamw(
            p,
            p.grad,
            state["exp_avg"],
            state["exp_avg_sq"],
            state["scale_m"],
            state["scale_v"],
            state["v_min"],
            group["betas"][0],
            group["betas"][1],
            group["lr"],
            group["weight_decay"],
            group["eps"],
            state["step"],
            bits,
            bsz,
            profile_buf,
        )
        if p.is_cuda:
            torch.cuda.synchronize(p.device)
        kernel_ms = (time.perf_counter() - tk0) * 1000.0

        self.last_step_timings["quant_dequant_kernel_ms"] = (
            self.last_step_timings.get("quant_dequant_kernel_ms", 0.0) + kernel_ms
        )

    def _finalize_step_timings(self, profile_device):
        super()._finalize_step_timings(profile_device)
        fast_n = int(self.last_step_timings.get("mdq_fused_io_fast_path_params", 0))
        fall_n = int(self.last_step_timings.get("mdq_fused_io_fallback_params", 0))
        total = fast_n + fall_n
        self.last_step_timings["mdq_fused_io_fast_path_ratio"] = (
            (fast_n / total) if total > 0 else 0.0
        )

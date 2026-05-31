"""
MDQAdamWSimple: 对齐 mdqbock_newhook.py 算法，无 backward hook / 无多流 / 无 stats overlap。

流程（均在 optimizer.step() 内顺序执行）:
  backward 结束后用 p.grad 算统计量 -> EMA + score + current_bit
  -> 低比特 state 解包(反量化) -> fused_mdq_adamw(块量化写回) -> 打包存储

上一步计时见 get_last_step_timings()。
"""
import math
import time

import stquant_timetest_cpp
import torch
import torch.distributed as dist
from torch.optim import Optimizer


class MDQAdamWSimple(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        alpha=0.9,
        layer_count=24,
        batch_size=32,
        update_freq=20,
        block_size=256,
    ):
        tau_adaptive = 300 * math.log(layer_count) * math.sqrt(batch_size)
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            alpha=alpha,
            tau=tau_adaptive,
            update_freq=update_freq,
            block_size=block_size,
        )
        super().__init__(params, defaults)

        self.n_ema = None
        self.r_ema = None
        self.v_global_ema = None
        self.total_steps = 0
        self.last_step_timings = {}
        self._kernel_profile_buf = None
        self._kernel_profile_device = None

    _KERNEL_PROF_SLOTS = 5

    def _kernel_profile_buf_for(self, device):
        if (
            self._kernel_profile_buf is None
            or self._kernel_profile_device != device
        ):
            self._kernel_profile_buf = torch.zeros(
                self._KERNEL_PROF_SLOTS, dtype=torch.int64, device=device
            )
            self._kernel_profile_device = device
        return self._kernel_profile_buf

    def _reset_kernel_profile(self, device):
        self._kernel_profile_buf_for(device).zero_()

    def _read_kernel_profile_cycles(self, device):
        """各 phase 的 clock64 周期累加（跨 block / launch；仅用于算占比）。"""
        buf = self._kernel_profile_buf_for(device)
        return buf.double().cpu().tolist()

    def _kernel_phases_ms_from_wall(self, device, kernel_wall_ms):
        """方案 A：phase_ms = FusedKernel 墙钟 × (该 phase 周期 / 五段周期之和)。"""
        if kernel_wall_ms <= 0:
            return [0.0] * self._KERNEL_PROF_SLOTS
        cycles = self._read_kernel_profile_cycles(device)
        total = sum(cycles)
        if total <= 0:
            return [0.0] * self._KERNEL_PROF_SLOTS
        return [kernel_wall_ms * (c / total) for c in cycles]

    @staticmethod
    def _sync_device(device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # ------------------------------------------------------------------
    # grad stats (mdqbock 同款，在 step 内同步计算)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_grad_stats(grad: torch.Tensor) -> torch.Tensor:
        n_l = torch.sqrt(torch.mean(grad * grad))
        r_l = torch.std(grad) / (grad.abs().mean() + 1e-12)
        v_l = torch.mean(grad * grad)
        return torch.stack([n_l, r_l, v_l])

    @staticmethod
    def _score_to_bits(score: torch.Tensor) -> int:
        s = float(score.item()) if score.numel() == 1 else float(score)
        if s >= 24:
            return 32
        if s >= 12:
            return 16
        if s >= 6.8:
            return 8
        return 4
 
    # ------------------------------------------------------------------
    # momentum pack / unpack / align (与 stquant_opt 一致，供 fused kernel)
    # ------------------------------------------------------------------
    @staticmethod
    def _is_momentum_packed4(state):
        if "exp_avg" not in state or "exp_avg_sq" not in state:
            return False
        ea, easq = state["exp_avg"], state["exp_avg_sq"]
        return ea.dtype == torch.uint8 and easq.numel() == 0 and ea.numel() > 0

    @staticmethod
    def _unpack_packed4_momentum(state):
        packed = state["exp_avg"]
        m_u = (packed.to(torch.int32) & 0x0F).to(torch.int16)
        v_q = ((packed.to(torch.int32) >> 4) & 0x0F).to(torch.int32)
        state["exp_avg"] = (m_u - 8).to(torch.int8)
        state["exp_avg_sq"] = v_q.to(torch.uint8)

    @staticmethod
    def _pack_momentum_to_nibbles(state):
        ea, easq = state["exp_avg"], state["exp_avg_sq"]
        if ea.dtype != torch.int8 or easq.numel() != ea.numel():
            return
        dev = ea.device
        m_u = (ea.to(torch.int16) + 8).clamp(0, 15).to(torch.uint8)
        v_lo = easq.to(torch.int32).clamp(0, 15).to(torch.uint8)
        state["exp_avg"] = (m_u | (v_lo << 4)).to(dev)
        state["exp_avg_sq"] = torch.empty(0, dtype=torch.uint8, device=dev)

    @classmethod
    def _align_momentum_storage(cls, state, bits):
        if cls._is_momentum_packed4(state):
            cls._unpack_packed4_momentum(state)

        ea, easq = state["exp_avg"], state["exp_avg_sq"]

        if bits >= 32:
            if ea.dtype != torch.int32:
                state["exp_avg"] = ea.to(torch.int32)
            if easq.dtype != torch.uint32:
                state["exp_avg_sq"] = easq.to(torch.uint32)
            return

        if bits > 8:
            if ea.dtype == torch.int8:
                state["exp_avg"] = ea.to(torch.int16)
                state["exp_avg_sq"] = easq.to(torch.uint16)
            elif ea.dtype == torch.int32:
                state["exp_avg"] = ea.clamp(-32768, 32767).to(torch.int16)
                state["exp_avg_sq"] = easq.to(torch.int64).clamp(0, 65535).to(torch.uint16)
            elif ea.dtype != torch.int16:
                state["exp_avg"] = ea.to(torch.int16)
                state["exp_avg_sq"] = easq.to(torch.uint16)
            return

        if ea.dtype == torch.int16:
            state["exp_avg"] = ea.clamp(-128, 127).to(torch.int8)
            state["exp_avg_sq"] = easq.to(torch.int64).clamp(0, 255).to(torch.uint8)
        elif ea.dtype == torch.int32:
            state["exp_avg"] = ea.clamp(-128, 127).to(torch.int8)
            state["exp_avg_sq"] = easq.to(torch.int64).clamp(0, 255).to(torch.uint8)
        elif ea.dtype == torch.int8 and easq.numel() > 0 and easq.dtype != torch.uint8:
            state["exp_avg_sq"] = easq.to(torch.uint8)

    @classmethod
    def _pack_momentum_storage(cls, state, bits):
        ea, easq = state["exp_avg"], state["exp_avg_sq"]

        if bits <= 8:
            if ea.dtype == torch.int16:
                state["exp_avg"] = ea.clamp(-128, 127).to(torch.int8)
                state["exp_avg_sq"] = easq.to(torch.int64).clamp(0, 255).to(torch.uint8)
            elif ea.dtype == torch.int32:
                state["exp_avg"] = ea.clamp(-128, 127).to(torch.int8)
                state["exp_avg_sq"] = easq.to(torch.int64).clamp(0, 255).to(torch.uint8)
            elif ea.dtype == torch.int8 and easq.numel() > 0 and easq.dtype != torch.uint8:
                state["exp_avg_sq"] = easq.to(torch.uint8)

            if (
                bits == 4
                and state["exp_avg"].dtype == torch.int8
                and state["exp_avg_sq"].numel() == state["exp_avg"].numel()
            ):
                cls._pack_momentum_to_nibbles(state)
            return

        if bits == 16:
            if ea.dtype == torch.int32:
                state["exp_avg"] = ea.clamp(-32768, 32767).to(torch.int16)
                state["exp_avg_sq"] = easq.to(torch.int64).clamp(0, 65535).to(torch.uint16)
            elif ea.dtype == torch.int8:
                state["exp_avg"] = ea.to(torch.int16)
                state["exp_avg_sq"] = easq.to(torch.uint16)
            return

        if bits >= 32:
            if ea.dtype != torch.int32:
                state["exp_avg"] = ea.to(torch.int32)
            if easq.dtype != torch.uint32:
                state["exp_avg_sq"] = easq.to(torch.uint32)

    def _init_state_if_needed(self, p, state, block_size):
        if len(state) != 0:
            return

        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p, dtype=torch.int16)
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.uint16)

        num_blocks = (p.numel() + block_size - 1) // block_size
        state["scale_m"] = torch.zeros(num_blocks, dtype=torch.float32, device=p.device)
        state["scale_v"] = torch.zeros(num_blocks, dtype=torch.float32, device=p.device)
        state["v_min"] = torch.full((num_blocks,), -20.0, dtype=torch.float32, device=p.device)
        state["current_bit"] = 16
        state["last_score"] = torch.tensor(12.0, device=p.device)

    def get_last_step_timings(self):
        """最近一次 step() 的分项耗时（毫秒）。"""
        return dict(self.last_step_timings)

    def _launch_fused_update(self, p, group):
        state = self.state[p]
        bits = int(state["current_bit"])
        bsz = min(int(group["block_size"]), 1024)

        state["step"] += 1

        t0 = time.perf_counter()
        self._align_momentum_storage(state, bits)
        unpack_ms = (time.perf_counter() - t0) * 1000.0

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

        t1 = time.perf_counter()
        self._pack_momentum_storage(state, bits)
        pack_ms = (time.perf_counter() - t1) * 1000.0

        self.last_step_timings["quant_dequant_unpack_ms"] = (
            self.last_step_timings.get("quant_dequant_unpack_ms", 0.0) + unpack_ms
        )
        self.last_step_timings["quant_dequant_kernel_ms"] = (
            self.last_step_timings.get("quant_dequant_kernel_ms", 0.0) + kernel_ms
        )
        self.last_step_timings["quant_dequant_pack_ms"] = (
            self.last_step_timings.get("quant_dequant_pack_ms", 0.0) + pack_ms
        )

    def _finalize_step_timings(self, profile_device):
        unpack = self.last_step_timings.get("quant_dequant_unpack_ms", 0.0)
        pack = self.last_step_timings.get("quant_dequant_pack_ms", 0.0)
        kernel_wall = self.last_step_timings.get("quant_dequant_kernel_ms", 0.0)

        self.last_step_timings["mdq_unpack_ms"] = unpack
        self.last_step_timings["mdq_pack_ms"] = pack
        self.last_step_timings["mdq_fused_kernel_ms"] = kernel_wall
        self.last_step_timings["mdq_quant_dequant_ms"] = unpack + kernel_wall + pack

        if profile_device is not None:
            k_ms = self._kernel_phases_ms_from_wall(profile_device, kernel_wall)
            (
                self.last_step_timings["mdq_kernel_dequant_ms"],
                self.last_step_timings["mdq_kernel_adam_mv_ms"],
                self.last_step_timings["mdq_kernel_scale_reduce_ms"],
                self.last_step_timings["mdq_kernel_quant_ms"],
                self.last_step_timings["mdq_kernel_update_p_ms"],
            ) = k_ms
            # 五段之和应等于 kernel_wall（浮点误差外）
            self.last_step_timings["mdq_kernel_phases_sum_ms"] = sum(k_ms)
            self.last_step_timings["mdq_kernel_profile_cycles"] = (
                self._read_kernel_profile_cycles(profile_device)
            )
        else:
            for key in (
                "mdq_kernel_dequant_ms",
                "mdq_kernel_adam_mv_ms",
                "mdq_kernel_scale_reduce_ms",
                "mdq_kernel_quant_ms",
                "mdq_kernel_update_p_ms",
                "mdq_kernel_phases_sum_ms",
                "mdq_kernel_profile_cycles",
            ):
                self.last_step_timings[key] = 0.0 if key != "mdq_kernel_profile_cycles" else []

    # ------------------------------------------------------------------
    # metrics / checkpoint
    # ------------------------------------------------------------------
    def get_bit_distribution(self):
        counts = {4: 0, 8: 0, 16: 0, 32: 0}
        num_tensors = 0
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if st and "current_bit" in st:
                    b = int(st["current_bit"])
                    counts[b] = counts.get(b, 0) + 1
                    num_tensors += 1
        if num_tensors == 0:
            return {4: 0.0, 8: 0.0, 16: 0.0, 32: 0.0}
        return {b: (c / num_tensors) * 100.0 for b, c in counts.items()}

    def get_bit_distribution_param_weighted(self):
        counts_numel = {4: 0, 8: 0, 16: 0, 32: 0}
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if st and "current_bit" in st:
                    b = int(st["current_bit"])
                    counts_numel[b] = counts_numel.get(b, 0) + p.numel()
        total = sum(counts_numel.values())
        if total == 0:
            return {4: 0.0, 8: 0.0, 16: 0.0, 32: 0.0}
        return {k: (v / total) * 100.0 for k, v in counts_numel.items()}

    def get_avg_bit_param_weighted(self):
        dist_bits = self.get_bit_distribution_param_weighted()
        return float(sum(b * (perc / 100.0) for b, perc in dist_bits.items()))

    def get_all_raw_scores(self):
        scores = []
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if st and "last_score" in st:
                    scores.append(float(st["last_score"].item()))
        return scores

    def state_dict(self):
        out = super().state_dict()
        out["mdq_global_stats"] = {
            "n_ema": self.n_ema,
            "r_ema": self.r_ema,
            "v_global_ema": self.v_global_ema,
            "total_steps": self.total_steps,
        }
        return out

    def load_state_dict(self, state_dict):
        if "mdq_global_stats" in state_dict:
            stats = state_dict.pop("mdq_global_stats")
            self.n_ema = stats["n_ema"]
            self.r_ema = stats["r_ema"]
            self.v_global_ema = stats["v_global_ema"]
            self.total_steps = stats["total_steps"]
        super().load_state_dict(state_dict)

    # ------------------------------------------------------------------
    # step: 无 hook，backward 完成后在此顺序执行
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.total_steps += 1
        self.last_step_timings = {
            "mdq_stats_ms": 0.0,
            "mdq_bit_decision_ms": 0.0,
            "mdq_grad_stats_ms": 0.0,
            "mdq_allreduce_ms": 0.0,
            "mdq_score_bits_ms": 0.0,
            "mdq_update_decision": False,
            "mdq_update_freq": 0,
            "quant_dequant_unpack_ms": 0.0,
            "quant_dequant_kernel_ms": 0.0,
            "quant_dequant_pack_ms": 0.0,
            "mdq_fused_io_fast_path_params": 0,
            "mdq_fused_io_fallback_params": 0,
        }

        update_freq = self.param_groups[0]["update_freq"]
        update_decision = (self.total_steps % update_freq == 0) or (self.total_steps < 5)
        self.last_step_timings["mdq_update_decision"] = update_decision
        self.last_step_timings["mdq_update_freq"] = update_freq

        params_with_grad = []
        all_stats = []

        for group in self.param_groups:
            b_size = min(int(group.get("block_size", 256)), 1024)
            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append((p, group))
                state = self.state[p]
                self._init_state_if_needed(p, state, b_size)

        grad_stats_ms = 0.0
        allreduce_ms = 0.0
        score_bits_ms = 0.0

        if update_decision and params_with_grad:
            dev = params_with_grad[0][0].device
            self._sync_device(dev)
            t_grad = time.perf_counter()
            for p, _group in params_with_grad:
                all_stats.append(self._compute_grad_stats(p.grad))
            self._sync_device(dev)
            grad_stats_ms = (time.perf_counter() - t_grad) * 1000.0

        if update_decision and all_stats:
            dev = params_with_grad[0][0].device
            stats_tensor = torch.stack(all_stats)
            avg_stats = stats_tensor.mean(dim=0)

            if dist.is_available() and dist.is_initialized():
                self._sync_device(dev)
                t_ar = time.perf_counter()
                dist.all_reduce(avg_stats, op=dist.ReduceOp.SUM)
                avg_stats /= dist.get_world_size()
                self._sync_device(dev)
                allreduce_ms = (time.perf_counter() - t_ar) * 1000.0

            self._sync_device(dev)
            t_score = time.perf_counter()
            if self.n_ema is None:
                self.n_ema = avg_stats[0]
                self.r_ema = avg_stats[1]
                self.v_global_ema = avg_stats[2]
            else:
                alpha = self.param_groups[0]["alpha"]
                self.n_ema = alpha * avg_stats[0] + (1 - alpha) * self.n_ema
                self.r_ema = alpha * avg_stats[1] + (1 - alpha) * self.r_ema
                self.v_global_ema = alpha * avg_stats[2] + (1 - alpha) * self.v_global_ema

            tau = self.param_groups[0]["tau"]
            for i, (p, _group) in enumerate(params_with_grad):
                state = self.state[p]
                n_l, r_l, v_l = stats_tensor[i]

                s_t = 1.0 + (
                    1.0
                    / torch.cosh(
                        torch.tensor(state["step"] / tau, device=p.device, dtype=torch.float32)
                    )
                )

                score = (
                    7.2
                    + torch.log2(r_l / (self.r_ema + 1e-12))
                    + torch.log2(n_l / (self.n_ema + 1e-12))
                    + torch.log2(s_t)
                    + torch.log2(v_l / (self.v_global_ema + 1e-12))
                )

                state["last_score"] = score
                state["current_bit"] = self._score_to_bits(score)
            self._sync_device(dev)
            score_bits_ms = (time.perf_counter() - t_score) * 1000.0

        self.last_step_timings["mdq_grad_stats_ms"] = grad_stats_ms
        self.last_step_timings["mdq_allreduce_ms"] = allreduce_ms
        self.last_step_timings["mdq_score_bits_ms"] = score_bits_ms
        self.last_step_timings["mdq_stats_ms"] = grad_stats_ms
        self.last_step_timings["mdq_bit_decision_ms"] = allreduce_ms + score_bits_ms
        self.last_step_timings["mdq_extra_stats_ms"] = (
            grad_stats_ms + allreduce_ms + score_bits_ms
        )

        profile_device = None
        if params_with_grad:
            profile_device = params_with_grad[0][0].device
            if profile_device.type == "cuda":
                self._reset_kernel_profile(profile_device)

        for p, group in params_with_grad:
            self._launch_fused_update(p, group)

        self._finalize_step_timings(
            profile_device if profile_device is not None and profile_device.type == "cuda" else None
        )

        return loss

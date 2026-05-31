import math
import torch
from torch.optim import Optimizer
import torch.distributed as dist

BASE_THRESHOLDS = (6.8, 12.0, 24.0)


def scaled_thresholds(threshold_scale: float) -> tuple[float, float, float]:
    return tuple(t * threshold_scale for t in BASE_THRESHOLDS)


class MDQAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        alpha=0.9,
        layer_count=12,
        batch_size=8,
        update_freq=20,
        block_size=2048,
        tau_scale=1.0,
        score_bias=7.2,
        w_n=1.0,
        init_score=12.0,
        threshold_scale=1.0,
    ):
        tau_adaptive = 300 * math.log(layer_count) * math.sqrt(batch_size) * tau_scale
        t8, t16, t32 = scaled_thresholds(threshold_scale)
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            alpha=alpha,
            tau=tau_adaptive,
            update_freq=update_freq,
            block_size=block_size,
            score_bias=score_bias,
            w_n=w_n,
            init_score=init_score,
            threshold_scale=threshold_scale,
            thresh_8=t8,
            thresh_16=t16,
            thresh_32=t32,
        )
        super().__init__(params, defaults)

        self.n_ema = None
        self.r_ema = None
        self.v_global_ema = None
        self.total_steps = 0

    def state_dict(self):
        out_state_dict = super().state_dict()
        out_state_dict["mdq_global_stats"] = {
            "n_ema": self.n_ema,
            "r_ema": self.r_ema,
            "v_global_ema": self.v_global_ema,
            "total_steps": self.total_steps,
        }
        return out_state_dict

    def load_state_dict(self, state_dict):
        if "mdq_global_stats" in state_dict:
            stats = state_dict.pop("mdq_global_stats")
            self.n_ema = stats["n_ema"]
            self.r_ema = stats["r_ema"]
            self.v_global_ema = stats["v_global_ema"]
            self.total_steps = stats["total_steps"]
        super().load_state_dict(state_dict)

    def get_bit_distribution(self):
        counts = {4: 0, 8: 0, 16: 0, 32: 0}
        total_params = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if state and "current_bit" in state:
                    b = state["current_bit"]
                    counts[b] += 1
                    total_params += 1
        if total_params == 0:
            return {4: 0.0, 8: 0.0, 16: 0.0, 32: 0.0}
        return {b: (c / total_params) * 100 for b, c in counts.items()}

    def get_avg_bit(self) -> float:
        dist_bits = self.get_bit_distribution()
        return sum(b * (pct / 100.0) for b, pct in dist_bits.items())

    def robust_quantize(self, tensor, bits, is_v=False, block_size=2048):
        if bits >= 31:
            return tensor
        orig_shape = tensor.shape
        flat_tensor = tensor.flatten()
        numel = flat_tensor.numel()
        pad_len = (block_size - (numel % block_size)) % block_size
        if pad_len > 0:
            padded = torch.nn.functional.pad(flat_tensor, (0, pad_len))
        else:
            padded = flat_tensor
        blocked = padded.view(-1, block_size)
        eps = 1e-12
        if is_v:
            log_v = torch.log(blocked + eps)
            v_max = log_v.max(dim=1, keepdim=True)[0]
            v_min = log_v.min(dim=1, keepdim=True)[0]
            scale = (v_max - v_min) / (2**bits - 1)
            q_log_v = torch.round((log_v - v_min) / (scale + eps)).clamp(0, 2**bits - 1)
            dequantized = torch.exp(q_log_v * scale + v_min)
        else:
            q_levels = 2 ** (bits - 1)
            max_val = blocked.abs().max(dim=1, keepdim=True)[0]
            scale = (max_val + eps) / (q_levels - 1)
            q_tensor = torch.round(blocked / (scale + eps)).clamp(-q_levels, q_levels - 1)
            dequantized = q_tensor * scale
        return dequantized.view(-1)[:numel].view(orig_shape)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.total_steps += 1
        group0 = self.param_groups[0]
        update_decision = (self.total_steps % group0["update_freq"] == 0) or (
            self.total_steps < 5
        )

        all_stats = []
        params_to_update = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                params_to_update.append(p)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["current_bit"] = 16
                    state["last_score"] = torch.tensor(
                        group0["init_score"], device=p.device
                    )

                if update_decision:
                    grad = p.grad
                    n_l = torch.sqrt(torch.mean(grad**2))
                    r_l = torch.std(grad) / (grad.abs().mean() + 1e-12)
                    v_l = torch.mean(grad**2)
                    all_stats.append(torch.stack([n_l, r_l, v_l]))

        if update_decision and all_stats:
            stats_tensor = torch.stack(all_stats)
            avg_stats = stats_tensor.mean(dim=0)

            if dist.is_initialized():
                dist.all_reduce(avg_stats, op=dist.ReduceOp.SUM)
                avg_stats /= dist.get_world_size()

            if self.n_ema is None:
                self.n_ema, self.r_ema, self.v_global_ema = (
                    avg_stats[0],
                    avg_stats[1],
                    avg_stats[2],
                )
            else:
                alpha = group0["alpha"]
                self.n_ema = alpha * avg_stats[0] + (1 - alpha) * self.n_ema
                self.r_ema = alpha * avg_stats[1] + (1 - alpha) * self.r_ema
                self.v_global_ema = alpha * avg_stats[2] + (1 - alpha) * self.v_global_ema

            score_bias = group0["score_bias"]
            w_n = group0["w_n"]
            t8 = group0["thresh_8"]
            t16 = group0["thresh_16"]
            t32 = group0["thresh_32"]

            for i, p in enumerate(params_to_update):
                state = self.state[p]
                n_l, r_l, v_l = stats_tensor[i]
                tau = group0["tau"]
                s_t = 1.0 + (
                    1.0 / torch.cosh(torch.tensor(state["step"] / tau, device=p.device))
                )

                score = (
                    score_bias
                    + torch.log2(r_l / (self.r_ema + 1e-12))
                    + w_n * torch.log2(n_l / (self.n_ema + 1e-12))
                    + torch.log2(s_t)
                    + torch.log2(v_l / (self.v_global_ema + 1e-12))
                )

                state["last_score"] = score

                if score >= t32:
                    state["current_bit"] = 32
                elif score >= t16:
                    state["current_bit"] = 16
                elif score >= t8:
                    state["current_bit"] = 8
                else:
                    state["current_bit"] = 4

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                q_m = self.robust_quantize(
                    exp_avg, state["current_bit"], False, group["block_size"]
                )
                q_v = self.robust_quantize(
                    exp_avg_sq, state["current_bit"], True, group["block_size"]
                )

                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                bc1, bc2 = 1 - beta1 ** state["step"], 1 - beta2 ** state["step"]
                denom = (q_v.sqrt() / math.sqrt(bc2)).add_(group["eps"])
                p.addcdiv_(q_m, denom, value=-group["lr"] / bc1)
        return loss

    def get_all_raw_scores(self):
        all_scores = []
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if state and "last_score" in state:
                    all_scores.append(float(state["last_score"].item()))
        return all_scores

#include <torch/extension.h>

void launch_fused_mdq_adamw(
    torch::Tensor p, torch::Tensor grad, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor scale_m, torch::Tensor scale_v, torch::Tensor v_min,
    float beta1, float beta2, float lr, float weight_decay, float eps,
    int step, int bits, int block_size, torch::Tensor profile_cycles);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_mdq_adamw",
        &launch_fused_mdq_adamw,
        "MDQ AdamW fused update (timetest; profile_cycles: optional CUDA int64[5] for clock64 phase sums)"
    );
}

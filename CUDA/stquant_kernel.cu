#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdint.h>

// Kernel phase profiling (划法 A): 0=dequant 1=adam_mv 2=scale_reduce 3=quant 4=update_p
// 周期在各 block 上 atomicAdd 累加；Python 侧用方案 A 按占比 × FusedKernel 墙钟得 ms
enum {
    PROF_K_DEQUANT = 0,
    PROF_K_ADAM_MV = 1,
    PROF_K_SCALE_REDUCE = 2,
    PROF_K_QUANT = 3,
    PROF_K_UPDATE_P = 4,
    PROF_K_SLOTS = 5,
};

__device__ __forceinline__ void mdq_prof_add(
    unsigned long long* prof, int slot, unsigned long long cycles) {
    if (prof != nullptr && threadIdx.x == 0) {
        atomicAdd(reinterpret_cast<unsigned long long*>(prof + slot), cycles);
    }
}

// 块内求最大值规约
__device__ float block_reduce_max(float val) {
    __shared__ float shared_mem[1024];
    int tid = threadIdx.x;
    shared_mem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + s]);
        __syncthreads();
    }
    return shared_mem[0];
}

// 块内求最小值规约
__device__ float block_reduce_min(float val) {
    __shared__ float shared_mem[1024];
    int tid = threadIdx.x;
    shared_mem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared_mem[tid] = fminf(shared_mem[tid], shared_mem[tid + s]);
        __syncthreads();
    }
    return shared_mem[0];
}

// 4-bit 档：每参数 1 字节 — low nibble = (q_m+8)∈[0,15]，high nibble = q_v∈[0,15]
__global__ void fused_mdq_adamw_kernel_u8_packed4(
    float* p, const float* grad,
    uint8_t* mom_packed,
    float* scale_m, float* scale_v, float* v_min_ptr,
    float beta1, float beta2, float lr, float weight_decay, float eps,
    int step, int bits, int total_elements,
    unsigned long long* prof) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool is_valid = idx < total_elements;

    unsigned long long prof_t = 0;
    if (threadIdx.x == 0 && prof) prof_t = clock64();

    float sm = scale_m[blockIdx.x];
    float sv = scale_v[blockIdx.x];
    float v_min_val = v_min_ptr[blockIdx.x];

    uint8_t pk = is_valid ? mom_packed[idx] : 0;
    int m_u = (int)(pk & 0x0F);
    int v_q = (int)((pk >> 4) & 0x0F);
    float m = is_valid ? (((float)m_u - 8.0f) * sm) : 0.0f;
    float v = is_valid ? expf((float)v_q * sv + v_min_val) : 0.0f;

    float g = is_valid ? grad[idx] : 0.0f;

    if (threadIdx.x == 0 && prof) mdq_prof_add(prof, PROF_K_DEQUANT, clock64() - prof_t);

    if (threadIdx.x == 0 && prof) prof_t = clock64();
    if (is_valid) {
        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * g * g;
    }
    if (threadIdx.x == 0 && prof) mdq_prof_add(prof, PROF_K_ADAM_MV, clock64() - prof_t);

    double q_levels_m = pow(2.0, bits - 1);
    double q_levels_v = pow(2.0, bits) - 1.0;

    if (threadIdx.x == 0 && prof) prof_t = clock64();
    float abs_m = is_valid ? fabsf(m) : 0.0f;
    float log_v = is_valid ? logf(v + 1e-12f) : 0.0f;

    float max_m = block_reduce_max(abs_m);
    float max_log_v = block_reduce_max(log_v);
    float min_log_v = block_reduce_min(log_v);

    __shared__ float s_sm, s_sv, s_v_min;
    if (threadIdx.x == 0) {
        s_sm = (max_m + 1e-12f) / (float)q_levels_m;
        s_sv = (max_log_v - min_log_v + 1e-12f) / (float)q_levels_v;
        s_v_min = min_log_v;
        scale_m[blockIdx.x] = s_sm;
        scale_v[blockIdx.x] = s_sv;
        v_min_ptr[blockIdx.x] = s_v_min;
    }
    __syncthreads();
    if (threadIdx.x == 0 && prof) mdq_prof_add(prof, PROF_K_SCALE_REDUCE, clock64() - prof_t);

    if (threadIdx.x == 0 && prof) prof_t = clock64();
    if (is_valid) {
        double q_m = round((double)m / (double)(s_sm + 1e-12f));
        double q_m_lo = -q_levels_m;
        double q_m_hi = q_levels_m - 1.0;
        if (q_m < q_m_lo) q_m = q_m_lo;
        if (q_m > q_m_hi) q_m = q_m_hi;
        int mi = (int)q_m;
        int m_u_w = mi + 8;
        if (m_u_w < 0) m_u_w = 0;
        if (m_u_w > 15) m_u_w = 15;

        double q_v = round((double)(log_v - s_v_min) / (double)(s_sv + 1e-12f));
        if (q_v < 0.0) q_v = 0.0;
        if (q_v > q_levels_v) q_v = q_levels_v;
        int vi = (int)q_v;
        if (vi < 0) vi = 0;
        if (vi > 15) vi = 15;

        mom_packed[idx] = (uint8_t)((vi << 4) | (m_u_w & 0x0F));
    }
    if (threadIdx.x == 0 && prof) mdq_prof_add(prof, PROF_K_QUANT, clock64() - prof_t);

    if (threadIdx.x == 0 && prof) prof_t = clock64();
    if (is_valid) {
        float bc1 = 1.0f - powf(beta1, (float)step);
        float bc2 = 1.0f - powf(beta2, (float)step);
        float denom = (sqrtf(v) / sqrtf(bc2)) + eps;

        float weight = p[idx];
        if (weight_decay != 0.0f) weight *= (1.0f - lr * weight_decay);
        weight -= (lr / bc1) * (m / denom);
        p[idx] = weight;
    }
    if (threadIdx.x == 0 && prof) mdq_prof_add(prof, PROF_K_UPDATE_P, clock64() - prof_t);
}

// 8-bit 存储：bits==8 时 int8 + uint8（2B/参数）
__global__ void fused_mdq_adamw_kernel_s8_u8(
    float* p, const float* grad,
    int8_t* exp_avg, uint8_t* exp_avg_sq,
    float* scale_m, float* scale_v, float* v_min_ptr,
    float beta1, float beta2, float lr, float weight_decay, float eps,
    int step, int bits, int total_elements,
    unsigned long long* prof) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool is_valid = idx < total_elements;

    unsigned long long prof_t = 0;
    if (threadIdx.x == 0 && prof) prof_t = clock64();

    float sm = scale_m[blockIdx.x];
    float sv = scale_v[blockIdx.x];
    float v_min_val = v_min_ptr[blockIdx.x];

    float g = is_valid ? grad[idx] : 0.0f;
    float m = is_valid ? ((float)exp_avg[idx] * sm) : 0.0f;
    float v = is_valid ? expf((float)exp_avg_sq[idx] * sv + v_min_val) : 0.0f;

    if (threadIdx.x == 0 && prof) mdq_prof_add(prof, PROF_K_DEQUANT, clock64() - prof_t);

    if (threadIdx.x == 0 && prof) prof_t = clock64();
    if (is_valid) {
        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * g * g;
    }
    if (threadIdx.x == 0 && prof) mdq_prof_add(prof, PROF_K_ADAM_MV, clock64() - prof_t);

    double q_levels_m = pow(2.0, bits - 1);
    double q_levels_v = pow(2.0, bits) - 1.0;

    if (threadIdx.x == 0 && prof) prof_t = clock64();
    float abs_m = is_valid ? fabsf(m) : 0.0f;
    float log_v = is_valid ? logf(v + 1e-12f) : 0.0f;

    float max_m = block_reduce_max(abs_m);
    float max_log_v = block_reduce_max(log_v);
    float min_log_v = block_reduce_min(log_v);

    __shared__ float s_sm, s_sv, s_v_min;
    if (threadIdx.x == 0) {
        s_sm = (max_m + 1e-12f) / (float)q_levels_m;
        s_sv = (max_log_v - min_log_v + 1e-12f) / (float)q_levels_v;
        s_v_min = min_log_v;
        scale_m[blockIdx.x] = s_sm;
        scale_v[blockIdx.x] = s_sv;
        v_min_ptr[blockIdx.x] = s_v_min;
    }
    __syncthreads();
    if (threadIdx.x == 0 && prof) mdq_prof_add(prof, PROF_K_SCALE_REDUCE, clock64() - prof_t);

    if (threadIdx.x == 0 && prof) prof_t = clock64();
    if (is_valid) {
        double q_m = round((double)m / (double)(s_sm + 1e-12f));
        double q_m_lo = -q_levels_m;
        double q_m_hi = q_levels_m - 1.0;
        if (q_m < q_m_lo) q_m = q_m_lo;
        if (q_m > q_m_hi) q_m = q_m_hi;
        exp_avg[idx] = (int8_t)q_m;

        double q_v = round((double)(log_v - s_v_min) / (double)(s_sv + 1e-12f));
        if (q_v < 0.0) q_v = 0.0;
        if (q_v > q_levels_v) q_v = q_levels_v;
        exp_avg_sq[idx] = (uint8_t)q_v;
    }
    if (threadIdx.x == 0 && prof) mdq_prof_add(prof, PROF_K_QUANT, clock64() - prof_t);

    if (threadIdx.x == 0 && prof) prof_t = clock64();
    if (is_valid) {
        float bc1 = 1.0f - powf(beta1, (float)step);
        float bc2 = 1.0f - powf(beta2, (float)step);
        float denom = (sqrtf(v) / sqrtf(bc2)) + eps;

        float weight = p[idx];
        if (weight_decay != 0.0f) weight *= (1.0f - lr * weight_decay);
        weight -= (lr / bc1) * (m / denom);
        p[idx] = weight;
    }
    if (threadIdx.x == 0 && prof) mdq_prof_add(prof, PROF_K_UPDATE_P, clock64() - prof_t);
}

// 16-bit 存储：支持 bits<=16（初始与多数训练阶段）
__global__ void fused_mdq_adamw_kernel_s16_u16(
    float* p, const float* grad,
    int16_t* exp_avg, uint16_t* exp_avg_sq,
    float* scale_m, float* scale_v, float* v_min_ptr,
    float beta1, float beta2, float lr, float weight_decay, float eps,
    int step, int bits, int total_elements,
    unsigned long long* prof) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool is_valid = idx < total_elements;

    unsigned long long prof_t = 0;
    if (threadIdx.x == 0 && prof) prof_t = clock64();

    float sm = scale_m[blockIdx.x];
    float sv = scale_v[blockIdx.x];
    float v_min_val = v_min_ptr[blockIdx.x];

    float g = is_valid ? grad[idx] : 0.0f;
    float m = is_valid ? ((float)exp_avg[idx] * sm) : 0.0f;
    float v = is_valid ? expf((float)exp_avg_sq[idx] * sv + v_min_val) : 0.0f;

    if (threadIdx.x == 0 && prof) mdq_prof_add(prof, PROF_K_DEQUANT, clock64() - prof_t);

    if (threadIdx.x == 0 && prof) prof_t = clock64();
    if (is_valid) {
        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * g * g;
    }
    if (threadIdx.x == 0 && prof) mdq_prof_add(prof, PROF_K_ADAM_MV, clock64() - prof_t);

    double q_levels_m = pow(2.0, bits - 1);
    double q_levels_v = pow(2.0, bits) - 1.0;

    if (threadIdx.x == 0 && prof) prof_t = clock64();
    float abs_m = is_valid ? fabsf(m) : 0.0f;
    float log_v = is_valid ? logf(v + 1e-12f) : 0.0f;

    float max_m = block_reduce_max(abs_m);
    float max_log_v = block_reduce_max(log_v);
    float min_log_v = block_reduce_min(log_v);

    __shared__ float s_sm, s_sv, s_v_min;
    if (threadIdx.x == 0) {
        s_sm = (max_m + 1e-12f) / (float)q_levels_m;
        s_sv = (max_log_v - min_log_v + 1e-12f) / (float)q_levels_v;
        s_v_min = min_log_v;
        scale_m[blockIdx.x] = s_sm;
        scale_v[blockIdx.x] = s_sv;
        v_min_ptr[blockIdx.x] = s_v_min;
    }
    __syncthreads();
    if (threadIdx.x == 0 && prof) mdq_prof_add(prof, PROF_K_SCALE_REDUCE, clock64() - prof_t);

    if (threadIdx.x == 0 && prof) prof_t = clock64();
    if (is_valid) {
        double q_m = round((double)m / (double)(s_sm + 1e-12f));
        double q_m_lo = -q_levels_m;
        double q_m_hi = q_levels_m - 1.0;
        if (q_m < q_m_lo) q_m = q_m_lo;
        if (q_m > q_m_hi) q_m = q_m_hi;
        exp_avg[idx] = (int16_t)q_m;

        double q_v = round((double)(log_v - s_v_min) / (double)(s_sv + 1e-12f));
        if (q_v < 0.0) q_v = 0.0;
        if (q_v > q_levels_v) q_v = q_levels_v;
        exp_avg_sq[idx] = (uint16_t)q_v;
    }
    if (threadIdx.x == 0 && prof) mdq_prof_add(prof, PROF_K_QUANT, clock64() - prof_t);

    if (threadIdx.x == 0 && prof) prof_t = clock64();
    if (is_valid) {
        float bc1 = 1.0f - powf(beta1, (float)step);
        float bc2 = 1.0f - powf(beta2, (float)step);
        float denom = (sqrtf(v) / sqrtf(bc2)) + eps;

        float weight = p[idx];
        if (weight_decay != 0.0f) weight *= (1.0f - lr * weight_decay);
        weight -= (lr / bc1) * (m / denom);
        p[idx] = weight;
    }
    if (threadIdx.x == 0 && prof) mdq_prof_add(prof, PROF_K_UPDATE_P, clock64() - prof_t);
}

// 32-bit 档位：m 用 int32，v 码最大 2^32-1 用 uint32
__global__ void fused_mdq_adamw_kernel_s32_u32(
    float* p, const float* grad,
    int32_t* exp_avg, uint32_t* exp_avg_sq,
    float* scale_m, float* scale_v, float* v_min_ptr,
    float beta1, float beta2, float lr, float weight_decay, float eps,
    int step, int bits, int total_elements,
    unsigned long long* prof) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool is_valid = idx < total_elements;

    unsigned long long prof_t = 0;
    if (threadIdx.x == 0 && prof) prof_t = clock64();

    float sm = scale_m[blockIdx.x];
    float sv = scale_v[blockIdx.x];
    float v_min_val = v_min_ptr[blockIdx.x];

    float g = is_valid ? grad[idx] : 0.0f;
    float m = is_valid ? ((float)exp_avg[idx] * sm) : 0.0f;
    float v = is_valid ? expf((float)exp_avg_sq[idx] * sv + v_min_val) : 0.0f;

    if (threadIdx.x == 0 && prof) mdq_prof_add(prof, PROF_K_DEQUANT, clock64() - prof_t);

    if (threadIdx.x == 0 && prof) prof_t = clock64();
    if (is_valid) {
        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * g * g;
    }
    if (threadIdx.x == 0 && prof) mdq_prof_add(prof, PROF_K_ADAM_MV, clock64() - prof_t);

    double q_levels_m = pow(2.0, bits - 1);
    double q_levels_v = pow(2.0, bits) - 1.0;

    if (threadIdx.x == 0 && prof) prof_t = clock64();
    float abs_m = is_valid ? fabsf(m) : 0.0f;
    float log_v = is_valid ? logf(v + 1e-12f) : 0.0f;

    float max_m = block_reduce_max(abs_m);
    float max_log_v = block_reduce_max(log_v);
    float min_log_v = block_reduce_min(log_v);

    __shared__ float s_sm, s_sv, s_v_min;
    if (threadIdx.x == 0) {
        s_sm = (max_m + 1e-12f) / (float)q_levels_m;
        s_sv = (max_log_v - min_log_v + 1e-12f) / (float)q_levels_v;
        s_v_min = min_log_v;
        scale_m[blockIdx.x] = s_sm;
        scale_v[blockIdx.x] = s_sv;
        v_min_ptr[blockIdx.x] = s_v_min;
    }
    __syncthreads();
    if (threadIdx.x == 0 && prof) mdq_prof_add(prof, PROF_K_SCALE_REDUCE, clock64() - prof_t);

    if (threadIdx.x == 0 && prof) prof_t = clock64();
    if (is_valid) {
        double q_m = round((double)m / (double)(s_sm + 1e-12f));
        double q_m_lo = -q_levels_m;
        double q_m_hi = q_levels_m - 1.0;
        if (q_m < q_m_lo) q_m = q_m_lo;
        if (q_m > q_m_hi) q_m = q_m_hi;
        exp_avg[idx] = (int32_t)q_m;

        double q_v = round((double)(log_v - s_v_min) / (double)(s_sv + 1e-12f));
        if (q_v < 0.0) q_v = 0.0;
        if (q_v > q_levels_v) q_v = q_levels_v;
        exp_avg_sq[idx] = (uint32_t)q_v;
    }
    if (threadIdx.x == 0 && prof) mdq_prof_add(prof, PROF_K_QUANT, clock64() - prof_t);

    if (threadIdx.x == 0 && prof) prof_t = clock64();
    if (is_valid) {
        float bc1 = 1.0f - powf(beta1, (float)step);
        float bc2 = 1.0f - powf(beta2, (float)step);
        float denom = (sqrtf(v) / sqrtf(bc2)) + eps;

        float weight = p[idx];
        if (weight_decay != 0.0f) weight *= (1.0f - lr * weight_decay);
        weight -= (lr / bc1) * (m / denom);
        p[idx] = weight;
    }
    if (threadIdx.x == 0 && prof) mdq_prof_add(prof, PROF_K_UPDATE_P, clock64() - prof_t);
}

static unsigned long long* profile_ptr(torch::Tensor profile_cycles) {
    if (!profile_cycles.defined() || profile_cycles.numel() == 0) {
        return nullptr;
    }
    TORCH_CHECK(
        profile_cycles.is_cuda() && profile_cycles.scalar_type() == torch::kLong,
        "profile_cycles must be CUDA int64 tensor");
    TORCH_CHECK(profile_cycles.numel() >= PROF_K_SLOTS, "profile_cycles needs >= 5 elements");
    return reinterpret_cast<unsigned long long*>(profile_cycles.data_ptr<int64_t>());
}

void launch_fused_mdq_adamw(
    torch::Tensor p, torch::Tensor grad, torch::Tensor exp_avg, torch::Tensor exp_avg_sq,
    torch::Tensor scale_m, torch::Tensor scale_v, torch::Tensor v_min,
    float beta1, float beta2, float lr, float weight_decay, float eps,
    int step, int bits, int block_size, torch::Tensor profile_cycles) {

    int total_elements = p.numel();
    int threads = (block_size > 1024) ? 1024 : block_size;
    int blocks = (total_elements + threads - 1) / threads;

    TORCH_CHECK(p.is_cuda() && grad.is_cuda(), "p/grad must be CUDA");
    TORCH_CHECK(exp_avg.is_cuda(), "exp_avg must be CUDA");

    unsigned long long* prof = profile_ptr(profile_cycles);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(p.get_device()).stream();

    if (bits == 4 && exp_avg.scalar_type() == torch::kByte &&
        exp_avg_sq.numel() == 0) {
        TORCH_CHECK(exp_avg.numel() == total_elements,
                    "packed4: exp_avg numel must match parameters");
        fused_mdq_adamw_kernel_u8_packed4<<<blocks, threads, 0, stream>>>(
            p.data_ptr<float>(), grad.data_ptr<float>(),
            exp_avg.data_ptr<uint8_t>(),
            scale_m.data_ptr<float>(), scale_v.data_ptr<float>(), v_min.data_ptr<float>(),
            beta1, beta2, lr, weight_decay, eps, step, bits, total_elements, prof);
    } else if (exp_avg.scalar_type() == torch::kChar &&
               exp_avg_sq.scalar_type() == torch::kByte) {
        fused_mdq_adamw_kernel_s8_u8<<<blocks, threads, 0, stream>>>(
            p.data_ptr<float>(), grad.data_ptr<float>(),
            exp_avg.data_ptr<int8_t>(), exp_avg_sq.data_ptr<uint8_t>(),
            scale_m.data_ptr<float>(), scale_v.data_ptr<float>(), v_min.data_ptr<float>(),
            beta1, beta2, lr, weight_decay, eps, step, bits, total_elements, prof);
    } else if (exp_avg.scalar_type() == torch::kShort &&
               exp_avg_sq.scalar_type() == torch::kUInt16) {
        fused_mdq_adamw_kernel_s16_u16<<<blocks, threads, 0, stream>>>(
            p.data_ptr<float>(), grad.data_ptr<float>(),
            exp_avg.data_ptr<int16_t>(), exp_avg_sq.data_ptr<uint16_t>(),
            scale_m.data_ptr<float>(), scale_v.data_ptr<float>(), v_min.data_ptr<float>(),
            beta1, beta2, lr, weight_decay, eps, step, bits, total_elements, prof);
    } else if (exp_avg.scalar_type() == torch::kInt &&
               exp_avg_sq.scalar_type() == torch::kUInt32) {
        fused_mdq_adamw_kernel_s32_u32<<<blocks, threads, 0, stream>>>(
            p.data_ptr<float>(), grad.data_ptr<float>(),
            exp_avg.data_ptr<int32_t>(), exp_avg_sq.data_ptr<uint32_t>(),
            scale_m.data_ptr<float>(), scale_v.data_ptr<float>(), v_min.data_ptr<float>(),
            beta1, beta2, lr, weight_decay, eps, step, bits, total_elements, prof);
    } else {
        TORCH_CHECK(false, "MDQ: state must be packed4 (uint8+empty), (int8,uint8), (int16,uint16) or (int32,uint32), got ",
                    c10::toString(exp_avg.scalar_type()), " / ",
                    c10::toString(exp_avg_sq.scalar_type()), " bits=", bits);
    }
}

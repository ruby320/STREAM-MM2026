# GPT2-XL Optimizer Timetest

GPT2-XL 预训练阶段 optimizer 计时 benchmark，配置对齐 `rebuttal/baseline/pretrain_gpt.py`。

## Optimizers（四路对比）

| 名称 | 说明 |
|------|------|
| MDQAdamW-Simple | stats 在 `step()` 内同步；Python pack/unpack + fused kernel |
| MDQAdamW-Simple-FusedIO | 布局对齐时跳过 Python pack/unpack，I/O 融合进 CUDA kernel |
| AdamW-32bit | 32-bit 对照 |
| 8bit-Adam-bnb | bitsandbytes 8-bit 对照 |

## 训练配置（与 baseline 一致）

| 参数 | 值 |
|------|-----|
| 模型 | gpt2-xl |
| 数据 | `/workspace/data/openwebtext_processed` |
| batch/GPU | 4 |
| grad accum | 16 |
| seq len | 1024 |
| lr | 2e-4 |
| 精度 | bf16 |
| MDQ update_freq | 20（每 20 optimizer step 更新 stats/比特） |

## 启动

```bash
cd /workspace/rebuttal/CUDA
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 bash run_build_and_train.sh
```

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| TIMETEST_WARMUP_STEPS | 10 | 预热 optimizer step（不计入 CSV） |
| TIMETEST_MAX_STEPS | **40** | 正式计时的 optimizer step 数（warmup=10 时 measure 含 step 20、40 两次 stats 更新） |

## 输出

`results/history_*.json` → `aggregate_timing.py` → `results/summary_metrics.csv`

额外：`results/step_timing_{opt}.csv` 逐步明细（MDQ 含 `MDQ_*` 统计列）。

| 列 | 说明 |
|----|------|
| `Measure_Train_Wall(ms)` | measure 阶段总墙钟 |
| `Avg_Step_Wall(ms)` | 单 optimizer step 平均墙钟（含 DataLoader 等待） |
| `Throughput(tokens/s)` | 全局 token 吞吐 |
| `Peak_Mem_gpu_*(GB)` | 每张 GPU measure 阶段**总峰值**显存 |
| `OptState_gpu_*(GB)` | 每张 GPU **optimizer.state** tensor 静态占用 |
| `Avg_Forward/Backward/OptimizerStep(ms)` | 分项 sync 墙钟（每 step 内 accum 累加） |
| `Avg_MDQ_Extra_Stats(ms)` | MDQ 额外统计量**摊销到每 step** 的平均（40 step 中含 2 次 update） |
| `Avg_MDQ_*_on_update(ms)` | 仅在 `update_decision=True` 的 step 上的均值 |
| `MDQ_Update_Steps_in_measure` | measure 阶段触发 stats 更新的 step 数 |

**FusedIO**：`Avg_Unpack/Pack=0` 时 I/O 在 kernel 内（见 `Avg_Kernel_Dequant/Quant`）。

---

## Micro-batch Sweep（显存 → 更大 batch → 吞吐）

验证 MDQ 降低 `OptState` 后能否增大 per-GPU micro batch。

```bash
cd /workspace/rebuttal/CUDA
CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 bash run_microbatch_sweep.sh
```

默认 **4 卡 `torchrun`**，**指数跳跃 + 二分**找 OOM 边界（`MICRO_MAX=64`），在 `max_ok_micro` 上 benchmark。

搜索示例（AdamW，OOM 约在 40 附近）：probe `4 → 8 → 16 → 32 → 64(oom)`，再二分 `32..64` → `max_ok≈43`。

| 变量 | 默认 | 说明 |
|------|------|------|
| `NUM_GPUS` | 4 | >1 时用 `torchrun` 多卡 |
| `MICRO_MIN` / `MICRO_MAX` | 4 / 64 | 指数跳跃上限；未 OOM 时增大 `MICRO_MAX` |
| `FIND_MAX_MICRO` | 1 | 设 `0` 则扫固定 `MICRO_MIN..MAX` 区间 |
| `MICRO_SEARCH_MODE` | exp_binary | `linear` 为逐 +1 慢扫 |
| `MICRO_LIST` | 空 | 指定时忽略 find-max（如 `4,6,8,10`） |
| `GRAD_ACCUM` | 16 | gradient accumulation |
| `MICRO_WARMUP_STEPS` | 2 | 探测前 warmup |
| `MICRO_PROBE_STEPS` | 8 | 每个 micro 探测步数（加长可减少 probe/benchmark 不一致） |
| `MICRO_BENCHMARK_STEPS` | 20 | benchmark 步数 |
| `MICRO_BENCHMARK_FALLBACK` | 1 | benchmark OOM 时向更小 micro 回退 |
| `MICRO_CONTINUE_ON_FAIL` | 1 | 单个 optimizer 失败仍继续跑其余 |
| `PEAK_BUDGET_GB` | 空 | 可选显存预算（如 `36`） |
| `OPTS` | 32bit,bnb,Simple,FusedIO | 要跑的 optimizer 列表 |
| `MICRO_BENCHMARK` | 1 | 设 `0` 仅 probe 不测吞吐 |

输出：`results/microbatch_sweep/sweep_{opt}.csv`、`sweep_all.csv`

| 列 | 说明 |
|----|------|
| `Peak_Mean_GB` | 探测阶段整卡峰值 |
| `OptState_Mean_GB` | optimizer.state 静态占用 |
| `OptShare_Mean_Pct` | OptState / Peak × 100 |
| `Status` | ok / oom |
| `Search_Phase` | exp / binary / linear / list |
| `Max_OK_Micro` / `Probe_Max_OK_Micro` | probe 阶段最大可用 micro |
| `Benchmark_Max_OK_Micro` | benchmark 实际成功的 micro（rebuttal 主指标） |
| `First_OOM_Micro` | 首次 OOM 的 micro |
| `Reached_OOM_Boundary` | 是否在 sweep 内触达 OOM |
| `Benchmark_Throughput_tok_s` | 在 `Benchmark_Max_OK_Micro` 上的 tokens/s |

单 optimizer（4 卡）：

```bash
torchrun --standalone --nproc_per_node=4 microbatch_sweep.py \
  --optimizer MDQAdamW-Simple-FusedIO --find-max-micro --search-mode exp_binary \
  --micro-min 4 --micro-max 64 --benchmark
```

---

## Fixed Global Batch=256（公平对比 micro / step 时间）

固定 `micro×accum×GPUs=256`，从大到小找 max micro，再测 `Benchmark_Avg_Global_Step_ms`。

**默认 2 卡**（`P=128`，合法 micro 含 128/64/32…，比 4 卡多高档位）：

```bash
cd /workspace/rebuttal/CUDA
CUDA_VISIBLE_DEVICES=4,5 NUM_GPUS=2 bash run_fixed_global_batch_sweep.sh
```

| 变量 | 默认 | 说明 |
|------|------|------|
| `NUM_GPUS` | **2** | 须整除 `FIXED_GLOBAL_BATCH` |
| `FIXED_GLOBAL_BATCH` | 256 | 与 baseline token batch 一致 |
| `FIXED_G_OUTPUT_DIR` | `results/fixed_global_batch_2gpu` | 2 卡结果目录 |
| `OPTS` | 32bit / bnb / FusedIO | 三个 optimizer |

4 卡重跑：设 `NUM_GPUS=4`、`FIXED_G_OUTPUT_DIR=results/fixed_global_batch`。

输出：`g256_sweep_{opt}.csv`、`g256_sweep_all.csv`（含 `Benchmark_Max_OK_Micro`、`Benchmark_Avg_Global_Step_ms`）。

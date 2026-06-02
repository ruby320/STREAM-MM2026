# LLaMA-7B Optimizer Benchmark（DeepSpeed ZeRO-2）

在 GPT2-XL sweep 之外，用 **Llama-2-7B 架构** 做 optimizer 显存 / micro-batch / 吞吐对比。

| 项 | 默认 |
|----|------|
| 模型架构 | `LLAMA_MODEL_ID`（默认 `meta-llama/Llama-2-7b-hf` 的 **config**） |
| 权重初始化 | `LLAMA_INIT=random`（不下载 13GB 权重）；`pretrained` 时加载 HF 权重 |
| 并行 | **DeepSpeed ZeRO-2**（`ds_zero2.json` + Accelerate） |
| 数据 | **`/workspace/data/openwebtext_llama7b_1024`**（OpenWebText + Llama tokenizer） |
| MDQ 扩展 | 在父目录 `../` 编译 `setup.py build_ext --inplace` |

## 1. 准备数据（首次必做）

与 GPT2 线共用 **OpenWebText parquet**，仅用 **Llama tokenizer** 重分词：

```bash
cd /workspace/rebuttal/CUDA/7B
bash run_prepare_data.sh

# 调试（1000 条）:
# MAX_SAMPLES=1000 bash run_prepare_data.sh
```

输出目录：`/workspace/data/openwebtext_llama7b_1024`（可用 `LLAMA_DATA_PATH` 覆盖）。

## 2. Adam-mini 式主实验（推荐）

固定 **global batch = 256**，2 卡，五路对照 + MDQ：

| Optimizer | 脚本内名称 |
|-----------|------------|
| AdamW 32bit | `AdamW-32bit` |
| 8bit Adam (bnb) | `8bit-Adam-bnb` |
| Adam-mini | `Adam-mini` |
| GaLore | `GaLore` |
| MDQ | `MDQAdamW-Simple-FusedIO` |

```bash
cd /workspace/rebuttal/CUDA/7B
cd .. && python setup.py build_ext --inplace && cd 7B

bash smoke_zero2.sh

CUDA_VISIBLE_DEVICES=4,5 bash run_adam_mini_style.sh
```

结果：`results/adam_mini_style_g256/g256_sweep_all.csv`  
关键列：`Benchmark_Max_OK_Micro`、`Benchmark_Throughput_tok_s`、`Status`（OOM 为 ✗）。

等价于手动：

```bash
CUDA_VISIBLE_DEVICES=4,5 NUM_GPUS=2 \
  OPTS="AdamW-32bit 8bit-Adam-bnb Adam-mini GaLore MDQAdamW-Simple-FusedIO" \
  bash run_fixed_global_batch_sweep.sh
```

## 3. 其他实验

```bash
# max-micro sweep（默认含 6 个 optimizer，4 卡）
CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 bash run_microbatch_sweep.sh

# 计时 benchmark
CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 bash run_build_and_train.sh
```

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `LLAMA_MODEL_ID` | `meta-llama/Llama-2-7b-hf` | 架构 / tokenizer 来源 |
| `LLAMA_INIT` | `random` | `random` / `pretrained` |
| `LLAMA_DATA_PATH` | `/workspace/data/openwebtext_llama7b_1024` | 训练数据 |
| `LLAMA_SEQ_LEN` | `1024` | 序列长度 |
| `FIXED_GLOBAL_BATCH` | `256` | Adam-mini 式 fixed-G |
| `NUM_GPUS` | fixed-G: 2；sweep: 4 | 数据并行卡数 |
| `OPTS` | 见各脚本 `ALL_OPTS` | 覆盖 optimizer 列表 |

## 文件说明

| 文件 | 作用 |
|------|------|
| `prepare_openwebtext_llama7b.py` | OpenWebText → Llama tokenizer |
| `run_prepare_data.sh` | 数据预处理入口 |
| `run_adam_mini_style.sh` | Adam-mini 式主实验一键脚本 |
| `run_fixed_global_batch_sweep.sh` | fixed-G sweep（默认 5 个 OPTS） |
| `run_microbatch_sweep.sh` | max-micro sweep |
| `llama7b_common.py` | 模型 / ZeRO-2 / optimizer 公共逻辑 |
| `microbatch_sweep.py` | sweep Python 入口 |
| `llama7b_timetest.py` | 计时 benchmark |
| `ds_zero2.json` | DeepSpeed ZeRO-2 配置 |

## 说明

- **OptState_GB** 为 ZeRO-2 **每卡分片后** 的占用，与父目录 GPT2 DDP 全量不可直接比数值。
- `8bit-Adam-bnb` 与 DeepSpeed 组合若失败，可从 `OPTS` 临时去掉。
- 父目录 `rebuttal/CUDA/` GPT2 脚本与数据 **未修改**。

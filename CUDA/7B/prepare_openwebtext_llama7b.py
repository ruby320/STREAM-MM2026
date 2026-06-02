#!/usr/bin/env python3
"""
OpenWebText + Llama tokenizer → save_to_disk（供 7B sweep / timetest 使用）。

与仓库根目录 prepare_openwebtext.py 使用同一 parquet 语料，仅换 Llama 词表。
"""
from __future__ import annotations

import argparse
import glob
import os

from datasets import load_dataset
from transformers import AutoTokenizer

DEFAULT_RAW = "/workspace/data/openwebtext/plain_text/"
DEFAULT_SAVE = "/workspace/data/openwebtext_llama7b_1024"
DEFAULT_TOKENIZER = os.environ.get("LLAMA_MODEL_ID", "meta-llama/Llama-2-7b-hf")


def parse_args():
    p = argparse.ArgumentParser(description="OpenWebText → Llama tokenizer @1024")
    p.add_argument("--raw-path", default=os.environ.get("OWT_RAW_PATH", DEFAULT_RAW))
    p.add_argument("--save-path", default=os.environ.get("LLAMA_DATA_PATH", DEFAULT_SAVE))
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--seq-len", type=int, default=int(os.environ.get("LLAMA_SEQ_LEN", "1024")))
    p.add_argument("--num-proc", type=int, default=int(os.environ.get("TOKENIZE_NUM_PROC", "32")))
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="调试用：只 tokenize 前 N 条",
    )
    return p.parse_args()


def main():
    args = parse_args()
    parquet_files = sorted(glob.glob(os.path.join(args.raw_path, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"未找到 parquet: {args.raw_path}/*.parquet")

    print(f"加载 {len(parquet_files)} 个 parquet ...")
    dataset = load_dataset("parquet", data_files=parquet_files, split="train")
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"调试模式: 仅 {len(dataset)} 条")

    text_column = "text" if "text" in dataset.column_names else dataset.column_names[0]
    print(f"tokenizer={args.tokenizer} seq_len={args.seq_len} text_col={text_column}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=args.seq_len,
            padding="max_length",
        )

    print(f"Tokenize (num_proc={args.num_proc}) ...")
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
    )
    os.makedirs(os.path.dirname(args.save_path.rstrip("/")) or ".", exist_ok=True)
    print(f"保存到 {args.save_path} ...")
    tokenized.save_to_disk(args.save_path)
    print("完成。")


if __name__ == "__main__":
    main()

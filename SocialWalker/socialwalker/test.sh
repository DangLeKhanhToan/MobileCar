#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1

python test.py \
  --test_jsonl out_train10000/test.jsonl \
  --ckpt ckpt_rank.pt \
  --outdir plots_test \
  --num_samples 30 \
  --batch 8 \
  --normalize_cost

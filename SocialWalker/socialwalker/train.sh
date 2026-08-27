#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1

# python train.py \
#   --train_jsonl ../out15/train.jsonl \
#   --epochs 30 --batch 16 --lr 1e-3 \
#   --amp \
#   --save rank_best.pt \
#   --val_jsonl ../out15/val.jsonl

  python train.py \
  --train_jsonl out_train10000/train.jsonl \
  --val_jsonl out_train10000/val.jsonl \
  --test_jsonl out_train10000/test.jsonl \
  --epochs 20 --batch 32 --lr 1e-3 \
  --save ckpt_rank.pt \
  --shuffle_cands --seed 0

#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1

python split_data.py \
    --video_root ../datasets/video_frames_5000 \
    --out_dir data_splits \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --seed 42

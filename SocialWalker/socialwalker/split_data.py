#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create video-level train/val/test split from video_frames_5000 directory.

Example:
    python make_video_level_split.py \
        --video_root video_frames_5000 \
        --out_dir data_splits \
        --train_ratio 0.8 \
        --val_ratio 0.1 \
        --seed 42
"""

import os
import json
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str, required=True,
                        help="Path to video_frames_5000 directory")
    parser.add_argument("--out_dir", type=str, default="data_splits",
                        help="Output directory for split json files")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def extract_video_id(folder_name):
    """
    Example:
        '-yYi7Cp4uQ_0011' -> '-yYi7Cp4uQ'
        '-2nZxI9Pf9U_0041' -> '-2nZxI9Pf9U'
    """
    return folder_name.split("_")[0]


def main():
    args = parse_args()

    assert 0 < args.train_ratio < 1
    assert 0 <= args.val_ratio < 1
    assert args.train_ratio + args.val_ratio < 1

    # Collect all unique video_ids
    all_video_ids = set()

    for name in os.listdir(args.video_root):
        full_path = os.path.join(args.video_root, name)
        if not os.path.isdir(full_path):
            continue
        video_id = extract_video_id(name)
        all_video_ids.add(video_id)

    all_video_ids = sorted(list(all_video_ids))

    print(f"Total unique source videos: {len(all_video_ids)}")

    # Shuffle reproducibly
    random.seed(args.seed)
    random.shuffle(all_video_ids)

    n_total = len(all_video_ids)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    n_test = n_total - n_train - n_val

    train_ids = all_video_ids[:n_train]
    val_ids = all_video_ids[n_train:n_train + n_val]
    test_ids = all_video_ids[n_train + n_val:]

    print(f"Train videos: {len(train_ids)}")
    print(f"Val videos:   {len(val_ids)}")
    print(f"Test videos:  {len(test_ids)}")

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "train_video_ids.json"), "w") as f:
        json.dump(train_ids, f, indent=2)

    with open(os.path.join(args.out_dir, "val_video_ids.json"), "w") as f:
        json.dump(val_ids, f, indent=2)

    with open(os.path.join(args.out_dir, "test_video_ids.json"), "w") as f:
        json.dump(test_ids, f, indent=2)

    print("Split files saved to:", args.out_dir)


if __name__ == "__main__":
    main()
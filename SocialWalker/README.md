# SocialWalker

The Python package, training scripts, checkpoints, and split metadata are consolidated in `socialwalker/`.
Paths in the Python code use `pathlib`, so the package works with Windows and POSIX path separators.

## Windows setup

From this `SocialWalker` directory:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

Run training or evaluation from the package directory so the existing relative data paths resolve correctly:

```powershell
cd socialwalker
py train.py --train_jsonl out_train10000/train.jsonl --val_jsonl out_train10000/val.jsonl --test_jsonl out_train10000/test.jsonl --epochs 20 --batch 32 --lr 1e-3 --save ckpt_rank.pt --shuffle_cands --seed 0
py test.py --test_jsonl out_train10000/test.jsonl --ckpt ckpt_rank.pt --outdir plots_test --num_samples 30 --batch 8 --normalize_cost
```

The generated `out_train*` datasets and `plots_test` images stay local because a training JSONL is larger than GitHub's 100 MB per-file limit. Checkpoints and `data_splits` remain included in the repository.

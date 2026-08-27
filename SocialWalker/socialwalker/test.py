#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test + visualize SocialWalker ranking model.

- Load trained checkpoint (ckpt_rank.pt)
- Run evaluation on test.jsonl
- Save plots: ego history + candidates + expert + predicted best
- Annotate each trajectory with social cost = -score (optionally normalized)

Usage:
  python test.py \
    --test_jsonl out15/test.jsonl \
    --ckpt ckpt_rank.pt \
    --outdir plots_test \
    --num_samples 30 \
    --batch 8 \
    --topk_label 999 \
    --shuffle_cands \
    --seed 0

Notes:
- If you trained with --shuffle_cands, you may also set --shuffle_cands here for consistent candidate order randomization.
- social_cost = -score (lower is better).
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_mkdir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for batch in loader:
            ego = batch["ego"].to(device)
            peds = batch["peds"].to(device)
            ped_mask = batch["ped_mask"].to(device)
            cands = batch["cands"].to(device)
            cand_mask = batch["cand_mask"].to(device)
            y = batch["expert_index"].to(device)

            scores = model(ego, peds, ped_mask, cands, cand_mask)
            loss = F.cross_entropy(scores, y)

            pred = torch.argmax(scores, dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.shape[0])
            loss_sum += float(loss.item()) * int(y.shape[0])

    return {"loss": loss_sum / max(1, total), "acc": correct / max(1, total), "n": total}


def plot_one_sample(out_path, ego_xz, cands_xz, expert_i, pred_i, scores,
                    label_topk=999, normalize_cost=True):
    """
    ego_xz: (Th,2) np
    cands_xz: (K,Tf,2) np
    scores: (K,) np
    """
    K = cands_xz.shape[0]
    cost = -scores.astype(np.float32)  # lower is better

    if normalize_cost:
        cost = cost - float(cost.min())  # shift >=0 for readability

    # sort indices by best (lowest cost) => highest score
    order = np.argsort(cost)

    plt.figure(figsize=(7.5, 6.5))

    # ego history
    plt.plot(ego_xz[:, 0], ego_xz[:, 1], linewidth=2)
    plt.scatter([ego_xz[-1, 0]], [ego_xz[-1, 1]], s=35)

    # candidates (all in light)
    for k in range(K):
        traj = cands_xz[k]
        plt.plot(traj[:, 0], traj[:, 1], linewidth=1, alpha=0.35)

    # highlight expert & predicted best
    ex = cands_xz[int(expert_i)]
    pr = cands_xz[int(pred_i)]
    plt.plot(ex[:, 0], ex[:, 1], linewidth=3)  # expert (real)
    plt.plot(pr[:, 0], pr[:, 1], linewidth=3, linestyle="--")  # predicted best

    # annotate costs (either all, or top-k best)
    to_label = order[:min(label_topk, K)]
    for k in to_label:
        traj = cands_xz[int(k)]
        x_end, y_end = traj[-1, 0], traj[-1, 1]
        tag = f"{cost[int(k)]:.2f}"
        if int(k) == int(expert_i):
            tag = f"E:{tag}"
        if int(k) == int(pred_i):
            tag = f"P:{tag}"
        plt.text(x_end, y_end, tag, fontsize=8)

    plt.title(f"expert={int(expert_i)} pred={int(pred_i)} | score(pred)={scores[int(pred_i)]:.2f}")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=170)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_jsonl", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="plots_test")
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")

    # plotting
    ap.add_argument("--topk_label", type=int, default=999, help="how many trajectories to label with cost (999=all)")
    ap.add_argument("--normalize_cost", action="store_true")

    # keep consistent with training options
    ap.add_argument("--shuffle_cands", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    # data/model fallbacks if ckpt doesn't store args
    ap.add_argument("--Th", type=int, default=8)
    ap.add_argument("--Tf", type=int, default=16)
    ap.add_argument("--max_peds", type=int, default=32)
    ap.add_argument("--use_image_wh", action="store_true")
    ap.add_argument("--img_w", type=int, default=0)
    ap.add_argument("--img_h", type=int, default=0)
    ap.add_argument("--ego_hid", type=int, default=64)
    ap.add_argument("--ped_hid", type=int, default=64)
    ap.add_argument("--traj_hid", type=int, default=96)
    ap.add_argument("--mlp_hid", type=int, default=128)

    args = ap.parse_args()

    # deterministic shuffling (important if you enable shuffle_cands at test time)
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    safe_mkdir(args.outdir)

    # ---- Import from your train.py (same folder) ----
    # Requires: JsonlRankingDataset, collate_rank, RankingModel
    try:
        import train as train_mod
    except Exception as e:
        raise RuntimeError(
            "Cannot import train.py as module. Put test.py in the same folder as train.py "
            "or rename your training script to train.py.\n"
            f"Import error: {e}"
        )

    # load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get("args", None)

    # model dims
    ego_hid = (ckpt_args.get("ego_hid") if isinstance(ckpt_args, dict) and "ego_hid" in ckpt_args else args.ego_hid)
    ped_hid = (ckpt_args.get("ped_hid") if isinstance(ckpt_args, dict) and "ped_hid" in ckpt_args else args.ped_hid)
    traj_hid = (ckpt_args.get("traj_hid") if isinstance(ckpt_args, dict) and "traj_hid" in ckpt_args else args.traj_hid)
    mlp_hid = (ckpt_args.get("mlp_hid") if isinstance(ckpt_args, dict) and "mlp_hid" in ckpt_args else args.mlp_hid)

    model = train_mod.RankingModel(
        ego_hid=ego_hid, ped_hid=ped_hid, traj_hid=traj_hid, ped_feat_dim=4, mlp_hid=mlp_hid
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # dataset params
    Th = (ckpt_args.get("Th") if isinstance(ckpt_args, dict) and "Th" in ckpt_args else args.Th)
    Tf = (ckpt_args.get("Tf") if isinstance(ckpt_args, dict) and "Tf" in ckpt_args else args.Tf)
    max_peds = (ckpt_args.get("max_peds") if isinstance(ckpt_args, dict) and "max_peds" in ckpt_args else args.max_peds)

    use_image_wh = bool(args.use_image_wh)
    img_w, img_h = args.img_w, args.img_h

    test_set = train_mod.JsonlRankingDataset(
        args.test_jsonl,
        Th=Th,
        Tf=Tf,
        max_peds=max_peds,
        use_image_wh=use_image_wh,
        image_wh=(img_w, img_h),
        shuffle_cands=args.shuffle_cands,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=train_mod.collate_rank,
        pin_memory=True,
        drop_last=False,
    )

    # metrics
    m = evaluate(model, test_loader, device)
    print(f"[TEST] loss={m['loss']:.4f} acc={m['acc']*100:.2f}% n={m['n']}")

    # visualize first N samples
    saved = 0
    with torch.no_grad():
        for batch_i, batch in enumerate(test_loader):
            ego = batch["ego"].to(device)              # (B,Th,2)
            peds = batch["peds"].to(device)
            ped_mask = batch["ped_mask"].to(device)
            cands = batch["cands"].to(device)          # (B,K,Tf,2)
            cand_mask = batch["cand_mask"].to(device)  # (B,K)
            y = batch["expert_index"].to(device)       # (B,)

            scores = model(ego, peds, ped_mask, cands, cand_mask)  # (B,K)
            pred = torch.argmax(scores, dim=1)

            B = ego.shape[0]
            for b in range(B):
                Kb = int(cand_mask[b].sum().item())
                ego_np = ego[b].detach().cpu().numpy()
                cands_np = cands[b, :Kb].detach().cpu().numpy()
                scores_np = scores[b, :Kb].detach().cpu().numpy()

                out_path = os.path.join(args.outdir, f"test_{saved:04d}.png")
                plot_one_sample(
                    out_path=out_path,
                    ego_xz=ego_np,
                    cands_xz=cands_np,
                    expert_i=int(y[b].item()),
                    pred_i=int(pred[b].item()),
                    scores=scores_np,
                    label_topk=args.topk_label,
                    normalize_cost=args.normalize_cost,
                )

                # also print a small summary
                cost = -scores_np
                if args.normalize_cost:
                    cost = cost - cost.min()
                print(
                    f"[VIS {saved:04d}] expert={int(y[b].item())} pred={int(pred[b].item())} "
                    f"cost(expert)={float(cost[int(y[b].item())]):.3f} cost(pred)={float(cost[int(pred[b].item())]):.3f}"
                )

                saved += 1
                if saved >= args.num_samples:
                    break
            if saved >= args.num_samples:
                break

    print(f"Saved {saved} plots to: {args.outdir}")


if __name__ == "__main__":
    main()

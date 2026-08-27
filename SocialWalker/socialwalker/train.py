#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a SocialWalker-style trajectory ranking model from train.jsonl.

Expected JSONL schema per line (your current):
  - ego_history: list of [x,z] (Th,2)
  - candidates: list[dict] length K, each contains a trajectory (Tf,2) under some key
  - expert_index: int in [0, K-1]
  - ped_seq: list length Th, each element is dict with:
        - frame_idx: int
        - dets: list[dict], each det has bbox_xyxy and optional depth

This script:
  - reads JSONL
  - builds tensors:
        ego: (Th,2)
        cands: (K,Tf,2)
        peds: (Th,Nmax,F)
  - model outputs scores: (B,K)
  - loss: cross_entropy(scores, expert_index) with candidate padding masked

Run:
  python train_ranking_from_jsonl.py \
      --train_jsonl out15/train.jsonl \
      --val_jsonl out15/val.jsonl \
      --epochs 20 --batch 32 --lr 1e-3 \
      --save ckpt_rank.pt

If you don't have val_jsonl, omit it.
"""

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Utilities
# ----------------------------
def to_np_xy(seq: Any) -> Optional[np.ndarray]:
    """Convert list of [x,z] or [x,y,z] -> (T,2) float32 (x,z)."""
    if seq is None:
        return None
    try:
        a = np.asarray(seq, dtype=np.float32)
        if a.ndim != 2 or a.shape[0] < 1:
            return None
        if a.shape[1] == 2:
            return a
        if a.shape[1] >= 3:
            return a[:, [0, 2]]
        return None
    except Exception:
        return None


def pad_traj(t: np.ndarray, T: int) -> np.ndarray:
    """Pad/truncate trajectory to length T by repeating last point."""
    if t.shape[0] == T:
        return t
    if t.shape[0] > T:
        return t[:T]
    if t.shape[0] < 1:
        return np.zeros((T, 2), dtype=np.float32)
    pad = np.repeat(t[-1:, :], T - t.shape[0], axis=0)
    return np.concatenate([t, pad], axis=0).astype(np.float32)


def finite_float(x, default=None):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


# ----------------------------
# Candidate trajectory extraction
# ----------------------------
CAND_TRAJ_KEYS = [
    # Add/adjust once you confirm your exact key inside each candidate dict
    "traj", "traj_xz", "traj_local", "wp", "waypoints", "poses", "points", "path", "future"
]


def extract_candidate_traj(c: Dict[str, Any]) -> Optional[np.ndarray]:
    """Try multiple keys to find a candidate trajectory."""
    if not isinstance(c, dict):
        return None
    for k in CAND_TRAJ_KEYS:
        if k in c:
            t = to_np_xy(c[k])
            if t is not None:
                return t
    # sometimes stored as separate arrays
    if "x" in c and "z" in c:
        try:
            x = np.asarray(c["x"], dtype=np.float32).reshape(-1)
            z = np.asarray(c["z"], dtype=np.float32).reshape(-1)
            if x.shape[0] == z.shape[0] and x.shape[0] > 0:
                return np.stack([x, z], axis=1).astype(np.float32)
        except Exception:
            pass
    return None


# ----------------------------
# Ped feature extraction
# ----------------------------
def ped_det_to_feat(det: Dict[str, Any], W: Optional[float] = None, H: Optional[float] = None) -> Optional[np.ndarray]:
    """
    Ped feature (lightweight, paper-friendly, resolution-robust if W/H known):
      [cx_norm, h_norm, depth_norm, inv_depth_norm]

    - bbox in pixel -> normalized if W/H provided
    - depth assumed DepthAnything v2 .npy style in your pipeline (already loaded into det["depth"])
    """
    if not isinstance(det, dict):
        return None

    bb = det.get("bbox_xyxy", det.get("bbox", det.get("xyxy", None)))
    if not (isinstance(bb, (list, tuple)) and len(bb) >= 4):
        return None

    try:
        x1, y1, x2, y2 = [float(v) for v in bb[:4]]
    except Exception:
        return None

    cx = 0.5 * (x1 + x2)
    h = max(1.0, (y2 - y1))

    if W is not None and H is not None and W > 1 and H > 1:
        cxn = cx / W
        hn = h / H
    else:
        # fallback: keep pixel scale (not ideal, but won't crash)
        cxn = cx
        hn = h

    d = det.get("depth", None)
    d = finite_float(d, default=0.0)
    # clamp to reduce extreme skew; adjust 10.0 if your depth range differs
    d_clamp = float(np.clip(d, 0.0, 10.0))
    depth_norm = d_clamp / 10.0
    inv_depth = 1.0 / (d_clamp + 1e-3)  # monotonic inverse
    inv_depth = float(np.clip(inv_depth, 0.0, 50.0)) / 50.0

    return np.array([cxn, hn, depth_norm, inv_depth], dtype=np.float32)


# ----------------------------
# JSONL Dataset
# ----------------------------
class JsonlRankingDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        Th: int = 8,
        Tf: int = 16,
        max_peds: int = 32,
        use_image_wh: bool = False,
        image_wh: Tuple[int, int] = (0, 0),
        shuffle_cands: bool = False,
    ):
        self.shuffle_cands = shuffle_cands
        self.path = jsonl_path
        self.Th = Th
        self.Tf = Tf
        self.max_peds = max_peds
        self.use_image_wh = use_image_wh
        self.image_wh = image_wh  # (W,H)

        # Detect format: JSON array vs JSONL
        self._is_json_array = False
        self._json_array: List[Dict[str, Any]] = []
        self.offsets: List[int] = []

        # Try JSON array first (fast check)
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                first = f.read(1)
                f.seek(0)
                if first == "[":
                    obj = json.load(f)
                    if isinstance(obj, list):
                        self._is_json_array = True
                        self._json_array = obj
        except Exception:
            self._is_json_array = False
            self._json_array = []

        if not self._is_json_array:
            # Build byte offsets for JSONL random access
            off = 0
            with open(self.path, "rb") as f:
                for line in f:
                    # skip empty lines
                    if line.strip():
                        self.offsets.append(off)
                    off += len(line)

    def __len__(self):
        return len(self._json_array) if self._is_json_array else len(self.offsets)

    def _read_line(self, idx: int) -> Dict[str, Any]:
        if self._is_json_array:
            return self._json_array[idx]

        with open(self.path, "rb") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
        return json.loads(line.decode("utf-8"))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self._read_line(idx)

        ego = to_np_xy(s.get("ego_history", None))
        if ego is None:
            raise ValueError("Bad ego_history")
        ego = pad_traj(ego, self.Th)

        cands = s.get("candidates", None)
        if not isinstance(cands, list) or len(cands) == 0:
            raise ValueError("Bad candidates")

        exp_i = int(s.get("expert_index", 0))
        exp_i = max(0, min(exp_i, len(cands) - 1))

        # ---- SHUFFLE CANDIDATES TO REMOVE INDEX BIAS ----
        if getattr(self, "shuffle_cands", False):
            K = len(cands)
            perm = np.random.permutation(K)
            cands = [cands[i] for i in perm.tolist()]
            exp_i = int(np.where(perm == exp_i)[0][0])

        trajs: List[np.ndarray] = []
        for c in cands:
            t = extract_candidate_traj(c) if isinstance(c, dict) else None
            if t is None:
                t = np.zeros((self.Tf, 2), dtype=np.float32)
            t = pad_traj(t, self.Tf)
            trajs.append(t)
        trajs = np.stack(trajs, axis=0).astype(np.float32)  # (K,Tf,2)

        ped_seq = s.get("ped_seq", [])
        Fdim = 4
        peds = np.zeros((self.Th, self.max_peds, Fdim), dtype=np.float32)
        ped_mask = np.zeros((self.Th, self.max_peds), dtype=np.float32)

        W, H = None, None
        if self.use_image_wh and self.image_wh[0] > 0 and self.image_wh[1] > 0:
            W, H = float(self.image_wh[0]), float(self.image_wh[1])

        for t in range(min(self.Th, len(ped_seq))):
            ts = ped_seq[t]
            if not isinstance(ts, dict):
                continue
            dets = ts.get("dets", None)
            if not isinstance(dets, list):
                continue

            feats = []
            for det in dets:
                if not isinstance(det, dict):
                    continue
                feat = ped_det_to_feat(det, W=W, H=H)
                if feat is not None:
                    feats.append(feat)
            feats = feats[: self.max_peds]
            if feats:
                feats_np = np.stack(feats, axis=0).astype(np.float32)
                peds[t, : feats_np.shape[0]] = feats_np
                ped_mask[t, : feats_np.shape[0]] = 1.0

        return {
            "ego": torch.from_numpy(ego),                 # (Th,2)
            "cands": torch.from_numpy(trajs),             # (K,Tf,2)
            "cand_mask": torch.ones((trajs.shape[0],), dtype=torch.float32),
            "expert_index": torch.tensor(exp_i, dtype=torch.long),
            "peds": torch.from_numpy(peds),               # (Th,N,F)
            "ped_mask": torch.from_numpy(ped_mask),       # (Th,N)
        }

def collate_rank(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pad variable-K candidates across batch.
    Returns:
      ego: (B,Th,2)
      peds: (B,Th,N,F)
      ped_mask: (B,Th,N)
      cands: (B,Kmax,Tf,2)
      cand_mask: (B,Kmax)
      expert_index: (B,)  (indices in padded Kmax; valid because we keep order)
    """
    B = len(batch)
    Th = batch[0]["ego"].shape[0]
    Tf = batch[0]["cands"].shape[1]
    N = batch[0]["peds"].shape[1]
    Fdim = batch[0]["peds"].shape[2]

    Kmax = max(x["cands"].shape[0] for x in batch)

    ego = torch.stack([x["ego"] for x in batch], dim=0)  # (B,Th,2)
    peds = torch.stack([x["peds"] for x in batch], dim=0)  # (B,Th,N,F)
    ped_mask = torch.stack([x["ped_mask"] for x in batch], dim=0)  # (B,Th,N)

    cands = torch.zeros((B, Kmax, Tf, 2), dtype=torch.float32)
    cand_mask = torch.zeros((B, Kmax), dtype=torch.float32)
    expert_index = torch.zeros((B,), dtype=torch.long)

    for i, x in enumerate(batch):
        Ki = x["cands"].shape[0]
        cands[i, :Ki] = x["cands"]
        cand_mask[i, :Ki] = 1.0
        expert_index[i] = x["expert_index"]

    return {
        "ego": ego,
        "peds": peds,
        "ped_mask": ped_mask,
        "cands": cands,
        "cand_mask": cand_mask,
        "expert_index": expert_index,
    }


# ----------------------------
# Model
# ----------------------------
class GRUEncoder(nn.Module):
    def __init__(self, in_dim: int, hid: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hid,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        _, h = self.gru(x)
        return h[-1]  # (B,hid)


class RankingModel(nn.Module):
    """
    Encode:
      ego_history -> e (B,He)
      ped_seq -> p (B,Hp) via per-timestep pooling + GRU
      candidate traj -> t (B*K,Ht) via GRU

    Score each candidate with MLP([e,p,t]).
    """
    def __init__(self, ego_hid=64, ped_hid=64, traj_hid=96, ped_feat_dim=4, mlp_hid=128):
        super().__init__()
        self.ego_enc = GRUEncoder(in_dim=2, hid=ego_hid)
        self.traj_enc = GRUEncoder(in_dim=2, hid=traj_hid)

        # per-ped MLP then pool per timestep, then GRU over time
        self.ped_mlp = nn.Sequential(
            nn.Linear(ped_feat_dim, ped_hid),
            nn.ReLU(inplace=True),
            nn.Linear(ped_hid, ped_hid),
            nn.ReLU(inplace=True),
        )
        self.ped_time_gru = GRUEncoder(in_dim=ped_hid, hid=ped_hid)

        self.score_mlp = nn.Sequential(
            nn.Linear(ego_hid + ped_hid + traj_hid, mlp_hid),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hid, mlp_hid),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hid, 1),
        )

    def encode_peds(self, peds: torch.Tensor, ped_mask: torch.Tensor) -> torch.Tensor:
        """
        peds: (B,Th,N,F)
        ped_mask: (B,Th,N) in {0,1}
        returns p: (B,ped_hid)
        """
        B, Th, N, Fdim = peds.shape
        x = self.ped_mlp(peds.view(B * Th * N, Fdim)).view(B, Th, N, -1)  # (B,Th,N,H)

        m = ped_mask.unsqueeze(-1)  # (B,Th,N,1)
        x = x * m

        # mean pool over N (avoid div-by-0)
        denom = m.sum(dim=2).clamp_min(1.0)  # (B,Th,1)
        xt = x.sum(dim=2) / denom            # (B,Th,H)

        p = self.ped_time_gru(xt)            # (B,H)
        return p

    def forward(self, ego: torch.Tensor, peds: torch.Tensor, ped_mask: torch.Tensor,
                cands: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        """
        ego: (B,Th,2)
        peds: (B,Th,N,F)
        ped_mask: (B,Th,N)
        cands: (B,K,Tf,2)
        cand_mask: (B,K)
        returns scores: (B,K) (masked padding candidates get -inf)
        """
        B, K, Tf, _ = cands.shape
        e = self.ego_enc(ego)                 # (B,He)
        p = self.encode_peds(peds, ped_mask)  # (B,Hp)

        # encode candidates in one batch
        c = cands.view(B * K, Tf, 2)
        t = self.traj_enc(c)                  # (B*K,Ht)
        t = t.view(B, K, -1)                  # (B,K,Ht)

        ep = torch.cat([e, p], dim=-1)         # (B,He+Hp)
        ep = ep.unsqueeze(1).expand(-1, K, -1) # (B,K,He+Hp)

        feat = torch.cat([ep, t], dim=-1)      # (B,K,He+Hp+Ht)
        s = self.score_mlp(feat).squeeze(-1)   # (B,K)

        # mask padded candidates
        s = s.masked_fill(cand_mask <= 0.0, float("-inf"))
        return s


# ----------------------------
# Train / Eval
# ----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
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

    return {
        "loss": loss_sum / max(1, total),
        "acc": correct / max(1, total),
        "n": float(total),
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_set = JsonlRankingDataset(
        args.train_jsonl,
        Th=args.Th,
        Tf=args.Tf,
        max_peds=args.max_peds,
        use_image_wh=args.use_image_wh,
        image_wh=(args.img_w, args.img_h),
        shuffle_cands=args.shuffle_cands,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_rank,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = None
    if args.val_jsonl and os.path.exists(args.val_jsonl):
        val_set = JsonlRankingDataset(
            args.val_jsonl,
            Th=args.Th,
            Tf=args.Tf,
            max_peds=args.max_peds,
            use_image_wh=args.use_image_wh,
            image_wh=(args.img_w, args.img_h),
            shuffle_cands=args.shuffle_cands,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_rank,
            pin_memory=True,
            drop_last=False,
        )

        test_loader = None
        if args.test_jsonl and os.path.exists(args.test_jsonl):
            test_set = JsonlRankingDataset(
                args.test_jsonl,
                Th=args.Th,
                Tf=args.Tf,
                max_peds=args.max_peds,
                use_image_wh=args.use_image_wh,
                image_wh=(args.img_w, args.img_h),
                shuffle_cands=args.shuffle_cands,
            )
            test_loader = DataLoader(
                test_set,
                batch_size=args.batch,
                shuffle=False,
                num_workers=args.workers,
                collate_fn=collate_rank,
                pin_memory=True,
                drop_last=False,
            )

    model = RankingModel(
        ego_hid=args.ego_hid,
        ped_hid=args.ped_hid,
        traj_hid=args.traj_hid,
        ped_feat_dim=4,
        mlp_hid=args.mlp_hid,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

    best_val = -1.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            ego = batch["ego"].to(device)
            peds = batch["peds"].to(device)
            ped_mask = batch["ped_mask"].to(device)
            cands = batch["cands"].to(device)
            cand_mask = batch["cand_mask"].to(device)
            y = batch["expert_index"].to(device)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp)):
                scores = model(ego, peds, ped_mask, cands, cand_mask)
                loss = F.cross_entropy(scores, y)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            pred = torch.argmax(scores.detach(), dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.shape[0])
            loss_sum += float(loss.item()) * int(y.shape[0])

            global_step += 1

        train_loss = loss_sum / max(1, total)
        train_acc = correct / max(1, total)

        msg = f"[Epoch {epoch:03d}] train loss={train_loss:.4f} acc={train_acc*100:.2f}% n={total}"
        if val_loader is not None:
            val = evaluate(model, val_loader, device)
            msg += f" | val loss={val['loss']:.4f} acc={val['acc']*100:.2f}% n={int(val['n'])}"
            if val["acc"] > best_val:
                best_val = val["acc"]
                if args.save:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "args": vars(args),
                            "best_val_acc": best_val,
                        },
                        args.save,
                    )
                    msg += f"  (saved best -> {args.save})"
        else:
            # save last
            if args.save and (epoch == args.epochs):
                torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "args": vars(args)}, args.save)
                msg += f"  (saved last -> {args.save})"

        print(msg)

    if test_loader is not None:
        if args.save and os.path.exists(args.save):
            ckpt = torch.load(args.save, map_location=device)
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=True)

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                ego = batch["ego"].to(device)
                peds = batch["peds"].to(device)
                ped_mask = batch["ped_mask"].to(device)
                cands = batch["cands"].to(device)
                cand_mask = batch["cand_mask"].to(device)
                y = batch["expert_index"].to(device)

                scores = model(ego, peds, ped_mask, cands, cand_mask)

                # ---- PRINT SOCIAL COST HERE ----
                print(f"\n=== Test Sample {i} ===")
                print("Scores:", scores[0].cpu().numpy())
                print("Social cost:", (-scores[0]).cpu().numpy())
                print("Expert index:", y[0].item())
                print("Pred index:", torch.argmax(scores, dim=1)[0].item())

                if i >= 4:   # chỉ in 5 sample đầu cho đỡ spam
                    break

        test = evaluate(model, test_loader, device)
        print(f"[TEST] loss={test['loss']:.4f} acc={test['acc']*100:.2f}% n={int(test['n'])}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--val_jsonl", type=str, default="")
    ap.add_argument("--test_jsonl", type=str, default="")
    ap.add_argument("--save", type=str, default="")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # data shape
    ap.add_argument("--Th", type=int, default=8)
    ap.add_argument("--Tf", type=int, default=16)
    ap.add_argument("--max_peds", type=int, default=32)

    # optional: if your bbox are in a known resized image resolution, enable normalization
    ap.add_argument("--use_image_wh", action="store_true")
    ap.add_argument("--img_w", type=int, default=0)
    ap.add_argument("--img_h", type=int, default=0)

    # model sizes
    ap.add_argument("--ego_hid", type=int, default=64)
    ap.add_argument("--ped_hid", type=int, default=64)
    ap.add_argument("--traj_hid", type=int, default=96)
    ap.add_argument("--mlp_hid", type=int, default=128)
    ap.add_argument("--shuffle_cands", action="store_true", help="Shuffle candidate order per sample and update expert_index")
    ap.add_argument("--seed", type=int, default=0)

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
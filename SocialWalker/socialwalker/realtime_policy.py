"""Real-time adapter for the trajectory ranker trained by :mod:`train`."""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PolicyResult:
    motion: str = "S"
    speed: str = "0"
    confidence: float = 0.0
    reason: str = "warming up"
    trajectories: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    selected_index: int = -1
    people_xz: list = field(default_factory=list)


class SocialWalkerPolicy:
    """Build the training-time tensors from live person boxes and rank paths."""

    def __init__(self, checkpoint: str):
        import torch
        from .train import RankingModel

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        args = ckpt.get("args") or {}
        self.history_len = int(args.get("Th", 8))
        self.future_len = int(args.get("Tf", 16))
        self.max_peds = int(args.get("max_peds", 32))
        self.model = RankingModel(
            ego_hid=int(args.get("ego_hid", 64)), ped_hid=int(args.get("ped_hid", 64)),
            traj_hid=int(args.get("traj_hid", 96)), ped_feat_dim=4,
            mlp_hid=int(args.get("mlp_hid", 128)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.eval()
        self.frames = deque(maxlen=self.history_len)
        self.epoch = ckpt.get("epoch")
        self.best_val_acc = ckpt.get("best_val_acc")

    def _candidate_paths(self):
        """Seventeen smooth, robot-relative paths matching training candidate count/scale."""
        import numpy as np
        paths = []
        # Dataset futures are roughly eight units long. Preserve that scale and Tf.
        z = np.linspace(0.5, 8.0, self.future_len, dtype=np.float32)
        for curve in np.linspace(-0.10, 0.10, 17, dtype=np.float32):
            x = curve * (z ** 2)
            paths.append(np.stack((x, z), axis=1))
        return np.stack(paths, axis=0)

    def predict(self, people, width: int, height: int) -> PolicyResult:
        import numpy as np
        torch = self.torch
        feats, people_xz = [], []
        for det in people[:self.max_peds]:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            depth = float(det["depth_m"])
            depth_clip = float(np.clip(depth, 0.0, 10.0))
            feats.append([(x1 + x2) * 0.5 / width, max(1.0, y2 - y1) / height,
                          depth_clip / 10.0, min(50.0, 1.0 / (depth_clip + 1e-3)) / 50.0])
            # Approximate lateral location for the diagnostic top-down plot.
            people_xz.append([((x1 + x2) * 0.5 / width - 0.5) * 2.0 * depth, depth])
        self.frames.append(feats)
        paths = self._candidate_paths()
        if len(self.frames) < self.history_len:
            return PolicyResult(reason=f"warming up {len(self.frames)}/{self.history_len}",
                                trajectories=paths.tolist(), people_xz=people_xz)

        peds = np.zeros((1, self.history_len, self.max_peds, 4), np.float32)
        mask = np.zeros((1, self.history_len, self.max_peds), np.float32)
        for t, frame in enumerate(self.frames):
            count = min(len(frame), self.max_peds)
            if count:
                peds[0, t, :count] = frame[:count]
                mask[0, t, :count] = 1.0
        ego = np.zeros((1, self.history_len, 2), np.float32)
        cand_mask = np.ones((1, len(paths)), np.float32)
        with torch.no_grad():
            scores = self.model(torch.from_numpy(ego).to(self.device),
                                torch.from_numpy(peds).to(self.device),
                                torch.from_numpy(mask).to(self.device),
                                torch.from_numpy(paths[None]).to(self.device),
                                torch.from_numpy(cand_mask).to(self.device))[0]
            probabilities = torch.softmax(scores, dim=0)
            selected = int(scores.argmax().item())
        endpoint_x = float(paths[selected, -1, 0])
        motion = "G" if endpoint_x < -0.8 else "I" if endpoint_x > 0.8 else "F"
        return PolicyResult(motion=motion, speed="6", confidence=float(probabilities[selected].item()),
                            reason=f"ranked path {selected + 1}/{len(paths)}",
                            trajectories=paths.tolist(), scores=scores.cpu().tolist(),
                            selected_index=selected, people_xz=people_xz)

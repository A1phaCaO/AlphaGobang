"""自对弈数据缓冲。每轮训练写一个 npz，训练时取最近若干轮拼成内存窗口。
human/ 子目录存放人机对局记录（冷启动数据），不受窗口轮数限制。"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from .config import FEATURE_PLANES

_D4: dict[int, list] = {}


def d4_transforms(size: int):
    """8 个 D4 对称变换。返回 [(rows, cols, move_perm)]，
    rows/cols 为 (size, size) 网格的 gather 索引，move_perm 为 (size*size,)。"""
    if size not in _D4:
        out = []
        base = np.arange(size * size).reshape(size, size)
        for k in range(4):
            for flip in (False, True):
                idx = np.rot90(base, k)
                if flip:
                    idx = np.fliplr(idx)
                idx = np.ascontiguousarray(idx)
                out.append((idx // size, idx % size, idx.reshape(-1)))
        _D4[size] = out
    return _D4[size]


class Buffer:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def append_iter(self, it: int, feats: np.ndarray, pols: np.ndarray, zs: np.ndarray):
        np.savez_compressed(self.root / f"iter_{it:03d}.npz",
                            feats=feats, pols=pols, zs=zs)

    def load_window(self, last_iter: int, n_iters: int,
                    with_human: bool = True) -> dict | None:
        fs, ps, zs = [], [], []
        if with_human:
            h = self.load_human()
            if h is not None:
                fs.append(h["feats"])
                ps.append(h["pols"])
                zs.append(h["zs"])
        for i in range(max(0, last_iter - n_iters + 1), last_iter + 1):
            p = self.root / f"iter_{i:03d}.npz"
            if not p.exists():
                continue
            with np.load(p) as d:
                fs.append(d["feats"])
                ps.append(d["pols"])
                zs.append(d["zs"])
        if not fs:
            return None
        return {"feats": np.concatenate(fs), "pols": np.concatenate(ps),
                "zs": np.concatenate(zs)}

    # ---------- 人机对局（冷启动）数据 ----------
    def append_human(self, feats: np.ndarray, pols: np.ndarray, zs: np.ndarray):
        d = self.root / "human"
        d.mkdir(exist_ok=True)
        idx = len(list(d.glob("h_*.npz")))
        np.savez_compressed(d / f"h_{idx:04d}.npz",
                            feats=feats, pols=pols, zs=zs)

    def load_human(self) -> dict | None:
        d = self.root / "human"
        files = sorted(d.glob("h_*.npz"))
        if not files:
            return None
        fs, ps, zs = [], [], []
        for p in files:
            with np.load(p) as x:
                fs.append(x["feats"])
                ps.append(x["pols"])
                zs.append(x["zs"])
        return {"feats": np.concatenate(fs), "pols": np.concatenate(ps),
                "zs": np.concatenate(zs)}


class GameRecorder:
    """记录一盘对局：每手记（落子前局面、行棋方、落子、可选的访问分布）。
    policy 缺省用落子 one-hot（人类着法），AI 着法可传 MCTS 访问分布。"""

    def __init__(self):
        self.samples = []

    def reset(self):
        self.samples = []

    @property
    def empty(self) -> bool:
        return not self.samples

    def add(self, game_before, pos: int, policy=None):
        self.samples.append((game_before.encode(), pos, game_before.turn, policy))

    def finish(self, winner: int):
        """winner: 0/1 胜方，2 和。样本太少（<6 手）视为误触不保存。"""
        if len(self.samples) < 6 or winner not in (0, 1, 2):
            self.reset()
            return None
        n = len(self.samples)
        size = self.samples[0][0].shape[1]
        A = size * size
        feats = np.stack([s[0] for s in self.samples]).astype(np.int8)
        pols = np.zeros((n, A), np.float32)
        zs = np.zeros(n, np.float32)
        for i, (_, pos, turn, policy) in enumerate(self.samples):
            if policy is not None and policy.sum() > 0.99:
                pols[i] = policy
            else:
                pols[i, pos] = 1.0
            zs[i] = 0.0 if winner == 2 else (1.0 if turn == winner else -1.0)
        self.reset()
        return feats, pols, zs

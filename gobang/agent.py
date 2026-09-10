"""网络智能体：封装批量推理与 MCTS，输出落子与访问分布。"""
from __future__ import annotations

import numpy as np
import torch

from .mcts import mcts_search
from .model import load_ckpt


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=1, keepdims=True)
    return x


class Agent:
    def __init__(self, ckpt_path=None, net=None, device: str = "cpu",
                 c_puct: float = 1.6, batch: int = 32, sim: int = 400):
        if net is None:
            net, _ = load_ckpt(ckpt_path, device)
        self.net = net
        self.size = net.size
        self.device = device
        self.c_puct = c_puct
        self.batch = batch
        self.sim = sim

    def evaluator(self, games):
        x = np.stack([g.encode() for g in games])
        t = torch.from_numpy(x).to(self.device)
        with torch.inference_mode():  # 比 no_grad 少一层版本计数，推理更快
            logits, value = self.net(t)
        probs = _softmax(logits.float().cpu().numpy())
        return probs, value.float().cpu().numpy()

    def act(self, game, temperature: float = 0.0, noise_eps: float = 0.0,
            noise_alpha: float = 0.12, rng: np.random.Generator | None = None,
            sim: int | None = None, tactic: bool = True):
        force = None
        if tactic and not game.is_terminal():
            mw = game.winning_moves(game.turn)
            if mw:
                force = set(mw)
            else:
                ow = game.winning_moves(1 - game.turn)
                if ow:
                    force = set(ow)  # 对方一手可成五：只允许堵
        root = mcts_search(game, self.evaluator, sim or self.sim, self.c_puct,
                           self.batch, noise_eps, noise_alpha, rng, force)
        visits = {m: ch.N for m, ch in root.children.items()}
        total = float(sum(visits.values()))
        policy = np.zeros(self.size * self.size, np.float32)
        for m, n in visits.items():
            policy[m] = n / total
        if temperature <= 1e-9:
            mx = max(visits.values())
            cands = [m for m, n in visits.items() if n == mx]
            action = int(rng.choice(cands)) if rng is not None else cands[0]
        else:
            keys = np.fromiter(visits, np.int64)
            w = np.fromiter(visits.values(), np.float64) ** (1.0 / temperature)
            w /= w.sum()
            action = int(rng.choice(keys, p=w))
        stats = {"value": root.Q, "visits": root.N, "policy": policy,
                 "move": action, "force": force}
        return action, policy, stats


class RandomPlayer:
    """均匀随机对手，用于基准测试。"""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, game, **kwargs):
        moves = game.legal_moves()
        action = int(self.rng.choice(moves))
        policy = np.zeros(game.size * game.size, np.float32)
        policy[action] = 1.0
        return action, policy, {"value": 0.0, "visits": 0, "policy": policy,
                                "move": action, "force": None}

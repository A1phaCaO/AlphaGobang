"""启发式老师：五子棋模式打分引擎，生成冷启动数据。

与「人机对局记录」的区别：人机局里 AI 一侧的 π 是当时（可能很弱的）网络的
访问分布，会把噪声当老师喂进训练；本模块的策略来自手写窗口评估——对每个
空点统计穿过它的所有 5 格窗口（成五 > 双四 > 成四 > 活三…，攻防加权），
无禁手 9×9 上明显强于未训练网络，自对弈的 z 全部来自真实终局。

用法（训练前生成一次即可）：
    uv run python -m gobang.teacher --games 600 --out runs/5060c/buffer
然后 train.json 设 human_epochs: 10~20 蒸馏若干轮；老师数据永久留在训练窗口
里，但会被自对弈数据稀释，之后 human_epochs 归零与否均可。
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .config import BOARD_SIZE
from .dataset import Buffer
from .game import Gomoku

DEVAL_W = 0.9          # 防分权重：同型的进攻略优先于防守
DOUBLE_FOUR = 5e6      # 一手同时造出两个「四」= 对方堵不完，视同必胜点
DOUBLE_FOUR_DEF = 4.5e6
_TABLE = {5: 1e7, 4: 2e5, 3: 2e3, 2: 50.0, 1: 1.0}
_WIN_CACHE: dict[int, tuple[list[np.ndarray], list[list[int]]]] = {}


def _windows(size: int):
    """全部 5 连窗口 + 每格所属窗口索引表。"""
    if size not in _WIN_CACHE:
        wins: list[list[int]] = []
        for r in range(size):
            for c in range(size - 4):
                wins.append([r * size + c + k for k in range(5)])
        for c in range(size):
            for r in range(size - 4):
                wins.append([(r + k) * size + c for k in range(5)])
        for r in range(size - 4):
            for c in range(size - 4):
                wins.append([(r + k) * size + c + k for k in range(5)])
        for r in range(size - 4):
            for c in range(4, size):
                wins.append([(r + k) * size + c - k for k in range(5)])
        cell2w: list[list[int]] = [[] for _ in range(size * size)]
        for i, w in enumerate(wins):
            for p in w:
                cell2w[p].append(i)
        _WIN_CACHE[size] = ([np.array(w) for w in wins], cell2w)
    return _WIN_CACHE[size]


class Teacher:
    def __init__(self, size: int = BOARD_SIZE,
                 rng: np.random.Generator | None = None):
        self.size = size
        self.rng = rng or np.random.default_rng()

    def move_scores(self, game: Gomoku) -> np.ndarray:
        """(size*size,) 每空点打分；远离棋子的点为 0。"""
        s = self.size
        wins, cell2w = _windows(s)
        flat = np.full(s * s, -1, np.int8)
        flat[game.p0] = 0
        flat[game.p1] = 1
        me, opp = game.turn, 1 - game.turn
        near = np.zeros(s * s, bool)
        for p in game.p0.tolist() + game.p1.tolist():
            r, c = divmod(p, s)
            for rr in range(max(r - 2, 0), min(r + 2, s - 1) + 1):
                lo, hi = max(c - 2, 0), min(c + 2, s - 1)
                near[rr * s + lo:rr * s + hi + 1] = True
        empty = flat < 0
        cand = np.flatnonzero(empty & (near if near.any() else empty))
        sc = np.zeros(s * s)
        for x in cand:
            own = defv = 0.0
            c4own = c4def = 0
            for wi in cell2w[x]:
                v = flat[wins[wi]]
                # 我占 x：该窗口的己方连子数（窗口内无对方子才有价值）
                mo = int(np.count_nonzero(v == me)) + 1
                if int(np.count_nonzero(v == opp)) == 0:
                    own += _TABLE[min(mo, 5)]
                    c4own += mo == 4
                # 对手占 x 的威胁 = 防守价值
                mp = int(np.count_nonzero(v == opp)) + 1
                if int(np.count_nonzero(v == me)) == 0:
                    defv += _TABLE[min(mp, 5)]
                    c4def += mp == 4
            total = own + DEVAL_W * defv
            if c4own >= 2:
                total += DOUBLE_FOUR
            if c4def >= 2:
                total += DOUBLE_FOUR_DEF
            sc[x] = total
        return sc

    def policy(self, game: Gomoku, temp: float) -> np.ndarray:
        ls = np.log1p(self.move_scores(game))
        ls -= ls.max()
        w = np.exp(ls / max(temp, 1e-6))
        w /= w.sum()
        return w.astype(np.float32)

    def play(self, tau_early: float = 0.6, tau_moves: int = 8,
             tau_late: float = 0.2):
        """一局师生自对弈：前 tau_moves 手大温度保证开局多样，之后近贪心。"""
        g = Gomoku(self.size)
        feats, pols, turns = [], [], []
        while not g.is_terminal():
            temp = tau_early if g.n_moves < tau_moves else tau_late
            pi = self.policy(g, temp)
            pos = int(self.rng.choice(pi.size, p=pi))
            feats.append(g.encode())
            pols.append(pi)
            turns.append(g.turn)
            g = g.placed(pos)
        w = g.winner
        zs = np.asarray(
            [0.0 if w == 2 else (1.0 if t == w else -1.0) for t in turns],
            np.float32)
        return (np.stack(feats).astype(np.int8),
                np.stack(pols).astype(np.float32), zs, w)


def main():
    ap = argparse.ArgumentParser(description="启发式老师生成冷启动数据")
    ap.add_argument("--out", required=True,
                    help="目标 buffer 目录（如 runs/5060c/buffer），写入其 human/ 子目录")
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--size", type=int, default=BOARD_SIZE)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    buf = Buffer(a.out)
    rng = np.random.default_rng(a.seed)
    th = Teacher(a.size, rng)
    t0 = time.time()
    n_black = n_draw = n_moves = 0
    for _ in range(a.games):
        f, p, z, w = th.play()
        buf.append_human(f, p, z)
        n_moves += len(z)
        n_draw += w == 2
        n_black += w == 0
    print(f"{a.games} 局完成：样本 {n_moves}，平均局长 {n_moves/a.games:.0f}，"
          f"先手胜 {n_black/a.games:.0%}，和 {n_draw}，"
          f"用时 {time.time()-t0:.0f}s → {buf.root}/human")


if __name__ == "__main__":
    main()

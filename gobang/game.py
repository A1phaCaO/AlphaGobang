"""五子棋规则引擎。

位棋盘表示，每行左右各留 WIN_LEN-1 个哨兵空位，保证跨行不会误判五连。
棋盘尺寸由 config.BOARD_SIZE 决定，可实例化任意 N×N。
"""
from __future__ import annotations

import numpy as np

from .config import BOARD_SIZE, FEATURE_PLANES, WIN_LEN

_META: dict[int, dict] = {}
_EMPTY = np.empty(0, np.int64)


def _meta(size: int) -> dict:
    m = _META.get(size)
    if m is None:
        pad = WIN_LEN - 1
        stride = size + 2 * pad
        bi_list = [r * stride + pad + c for r in range(size) for c in range(size)]
        bit = [1 << bi for bi in bi_list]
        all_mask = 0
        for b in bit:
            all_mask |= b
        b2p = {bi: p for p, bi in enumerate(bi_list)}
        dirs = (1, stride, stride - 1, stride + 1)
        R = np.array([p // size for p in range(size * size)], dtype=np.intp)
        C = np.array([p % size for p in range(size * size)], dtype=np.intp)
        m = {"pad": pad, "stride": stride, "bit": bit, "all_mask": all_mask,
             "b2p": b2p, "dirs": dirs, "R": R, "C": C}
        _META[size] = m
    return m


class Gomoku:
    """不可变风格的棋局对象，placed() 返回新状态，适合 MCTS 树复用。

    p0/p1 为双方落子位置的增量数组，使 encode() 可整体向量化。
    """

    __slots__ = ("size", "m", "stones", "turn", "last", "n_moves", "winner",
                 "p0", "p1")

    def __init__(self, size: int = BOARD_SIZE, state: tuple | None = None):
        self.size = size
        self.m = _meta(size)
        if state is None:
            self.stones = (0, 0)
            self.turn = 0
            self.last = -1
            self.n_moves = 0
            self.winner = -1
            self.p0 = _EMPTY
            self.p1 = _EMPTY
        else:
            self.stones, self.turn, self.last, self.n_moves, self.winner = \
                state[:5]
            if len(state) > 5:
                self.p0, self.p1 = state[5]
            else:
                self._rebuild_lists()

    def _rebuild_lists(self):
        self.p0 = np.fromiter(
            (p for p, b in enumerate(self.m["bit"]) if self.stones[0] & b),
            np.int64)
        self.p1 = np.fromiter(
            (p for p, b in enumerate(self.m["bit"]) if self.stones[1] & b),
            np.int64)

    def is_terminal(self) -> bool:
        return self.winner >= 0

    def _wins(self, mask: int) -> bool:
        for d in self.m["dirs"]:
            t = mask
            for k in range(1, WIN_LEN):
                t &= mask >> (k * d)
            if t:
                return True
        return False

    def placed(self, pos: int) -> "Gomoku":
        bit = self.m["bit"][pos]
        s0, s1 = self.stones
        if self.turn == 0:
            stones = (s0 | bit, s1)
            line = s0 | bit
            pls = (np.append(self.p0, pos), self.p1)
        else:
            stones = (s0, s1 | bit)
            line = s1 | bit
            pls = (self.p0, np.append(self.p1, pos))
        winner = -1
        if self._wins(line):
            winner = self.turn
        elif self.n_moves + 1 >= self.size * self.size:
            winner = 2
        return Gomoku(self.size, (stones, 1 - self.turn, pos, self.n_moves + 1,
                                  winner, pls))

    def forfeit(self) -> "Gomoku":
        """判当前行棋方负（用于认输）。"""
        return Gomoku(self.size, (self.stones, self.turn, self.last,
                                  self.n_moves, 1 - self.turn, (self.p0, self.p1)))

    def is_legal(self, pos: int) -> bool:
        if self.winner >= 0 or not 0 <= pos < self.size * self.size:
            return False
        return not (self.stones[0] | self.stones[1]) & self.m["bit"][pos]

    def legal_moves(self) -> list[int]:
        if self.winner >= 0:
            return []
        occ = self.stones[0] | self.stones[1]
        return [i for i, b in enumerate(self.m["bit"]) if not occ & b]

    def winning_moves(self, player: int) -> list[int]:
        """player 可立即成五的所有空点（战术检测，供搜索层强制掩码用）。"""
        if self.winner >= 0:
            return []
        occ = self.stones[0] | self.stones[1]
        m = self.m
        base = self.stones[player]
        out = []
        for i, b in enumerate(m["bit"]):
            if occ & b:
                continue
            t = base | b
            for d in m["dirs"]:
                if t & (t >> d) & (t >> (2 * d)) & (t >> (3 * d)) \
                        & (t >> (4 * d)):
                    out.append(i)
                    break
        return out

    def terminal_value(self) -> float:
        """终局时轮到走棋一方的得分：负方 -1，和棋 0。"""
        if self.winner == 2:
            return 0.0
        return -1.0

    def encode(self) -> np.ndarray:
        """(FEATURE_PLANES, S, S) 特征平面，以当前行棋方视角：己方、对方、上一手、先手恒置平面。"""
        s = self.size
        m = self.m
        e = np.zeros((FEATURE_PLANES, s, s), np.float32)
        mine, opp = (self.p0, self.p1) if self.turn == 0 else (self.p1, self.p0)
        if len(mine):
            e[0, m["R"][mine], m["C"][mine]] = 1.0
        if len(opp):
            e[1, m["R"][opp], m["C"][opp]] = 1.0
        if self.last >= 0:
            e[2, self.last // s, self.last % s] = 1.0
        if self.turn == 0:
            e[3] = 1.0
        return e

    def move_name(self, pos: int) -> str:
        s = self.size
        return f"{chr(ord('A') + pos % s)}{pos // s + 1}"

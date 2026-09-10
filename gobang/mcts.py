"""PUCT 蒙特卡洛树搜索。

批量推理：串行下降到若干新叶子（立即挂树并用虚拟损失 Q=-2 防止重复占用），
攒够一批后统一送网络评估、展开并回传，降低推理开销。
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

Evaluator = Callable[[list], tuple[np.ndarray, np.ndarray]]

# Q 存的是「该节点行棋方」视角的值；父节点选择时必须取 -ch.Q 换算成本方视角。
# 虚拟损失因此取 +2：取负后 u 极低，才能阻止同批下降重复进入同一待扩展节点。
VirtualLossQ = 2.0


class Node:
    __slots__ = ("game", "N", "W", "Q", "prior", "children")

    def __init__(self, game):
        self.game = game
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.prior = None
        self.children: dict[int, "Node"] = {}


def _expand(node: Node, policy_row: np.ndarray):
    legal = node.game.legal_moves()
    probs = {m: float(policy_row[m]) for m in legal}
    tot = sum(probs.values())
    if tot <= 0.0:
        u = 1.0 / len(probs)
        probs = {k: u for k in probs}
    else:
        probs = {k: v / tot for k, v in probs.items()}
    node.prior = probs


def _backup(path: list[Node], v: float):
    for node in reversed(path):
        node.N += 1
        node.W += v
        node.Q = node.W / node.N
        v = -v


def _select(root: Node, c_puct: float):
    """下降到新叶子。终局路径当场回传并返回 None，否则返回待评估的 (leaf, path)。"""
    node = root
    path = [root]
    while True:
        g = node.game
        if g.winner >= 0:
            _backup(path, g.terminal_value())
            return None
        if node.prior is None:
            return node, path
        spn = math.sqrt(node.N)
        best, bu = -1, float("-inf")
        for m, p in node.prior.items():
            ch = node.children.get(m)
            if ch is None:
                u = c_puct * p * spn
            else:
                u = -ch.Q + c_puct * p * spn / (1.0 + ch.N)
            if u > bu:
                bu, best = u, m
        ch = node.children.get(best)
        if ch is None:
            ch = Node(g.placed(best))
            node.children[best] = ch
            path.append(ch)
            if ch.game.winner >= 0:
                _backup(path, ch.game.terminal_value())
                return None
            ch.Q = VirtualLossQ
            return ch, path
        node = ch
        path.append(node)


def mcts_search(game, evaluator: Evaluator, sim: int, c_puct: float = 1.6,
                batch: int = 32, noise_eps: float = 0.0, noise_alpha: float = 0.12,
                rng: np.random.Generator | None = None,
                force: set | None = None) -> Node:
    root = Node(game)
    probs, values = evaluator([game])
    _expand(root, probs[0])
    if force:
        keep = {m: p for m, p in root.prior.items() if m in force}
        if keep:
            tot = sum(keep.values())
            root.prior = {k: v / tot for k, v in keep.items()}
    if noise_eps > 0.0 and rng is not None and len(root.prior) > 1:
        moves = list(root.prior)
        d = rng.dirichlet(np.full(len(moves), noise_alpha))
        pr = root.prior
        for m, di in zip(moves, d):
            pr[m] = (1.0 - noise_eps) * pr[m] + noise_eps * float(di)
    root.N = 1
    root.W = root.Q = float(values[0])
    done = 1
    pending: list[tuple[Node, list[Node]]] = []
    while done < sim:
        sel = _select(root, c_puct)
        done += 1
        if sel is not None:
            pending.append(sel)
        if len(pending) >= batch or (done >= sim and pending):
            outs = evaluator([lf.game for lf, _ in pending])
            for (lf, pa), pol, val in zip(pending, outs[0], outs[1]):
                _expand(lf, pol)
                lf.N = 1
                lf.W = lf.Q = float(val)
                _backup(pa[:-1], -float(val))
            pending = []
    return root

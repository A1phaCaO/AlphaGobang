"""PUCT 蒙特卡洛树搜索。

批量推理：串行下降到若干新叶子（挂树后打 pend 标记，选择时跳过），攒够一批统一
送网络评估、展开并回传，降低推理开销。

两条防退化的硬约束（违反过其中一条就会「训练不收敛」，详见 docs/训练细节.md）：
1. 待评估的叶子不许重复入队。sim 大于本层可走点数时，同批下降会再次落到同一个
   还没有 prior 的节点上，白白多花一次前向，还会把它的 N/Q 算重。
2. mcts_batch 只表示「一次前向最多塞几个叶子」，不是搜索预算。batch >= sim 时整步
   搜索只有一次回传，树里完全没有价值反馈：PUCT 退化成「按先验顺序枚举根节点的
   孩子」，每个点各 1 次访问，访问分布约等于均匀分布。自对弈数据于是近似随机落子，
   策略损失只能收敛到均匀分布的熵（9x9 上约 ln(54)=4.0）。这里按
   SEARCH_BATCH_ROUNDS 自动收紧 batch 作为兜底。
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .config import SEARCH_BATCH_ROUNDS

Evaluator = Callable[[list], tuple[np.ndarray, np.ndarray]]

# Q 存的是「该节点行棋方」视角的值；父节点选择时必须取 -ch.Q 换算成本方视角。
# 待评估的叶子用 pend 标记排除，比虚拟损失 Q=+2 更可靠（不进队列就不会被算重）。
Blocked = object()   # 本层可用孩子全在待评估队列里：先回传一批才能继续下降


class Node:
    __slots__ = ("game", "N", "W", "Q", "prior", "children", "pend")

    def __init__(self, game):
        self.game = game
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.prior = None
        self.children: dict[int, "Node"] = {}
        self.pend = False        # 已进待评估队列、尚未拿到网络输出


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
    """下降到新叶子并打上 pend。返回 (leaf, path)、None（终局已回传）或 Blocked。"""
    node = root
    path = [root]
    while True:
        g = node.game
        if g.winner >= 0:
            _backup(path, g.terminal_value())
            return None
        if node.prior is None:
            if node.pend:
                return Blocked
            node.pend = True
            return node, path
        spn = math.sqrt(node.N)
        best, bu, blocked = -1, float("-inf"), False
        for m, p in node.prior.items():
            ch = node.children.get(m)
            if ch is None:
                u = c_puct * p * spn
            elif ch.pend:
                blocked = True
                continue
            else:
                u = -ch.Q + c_puct * p * spn / (1.0 + ch.N)
            if u > bu:
                bu, best = u, m
        if best < 0:
            return Blocked if blocked else None
        ch = node.children.get(best)
        if ch is None:
            ch = Node(g.placed(best))
            node.children[best] = ch
            path.append(ch)
            if ch.game.winner >= 0:
                _backup(path, ch.game.terminal_value())
                return None
            ch.pend = True
            return ch, path
        node = ch
        path.append(node)


def clamp_batch(batch: int, sim: int) -> int:
    """保证一步搜索至少有 SEARCH_BATCH_ROUNDS 次「批量评估->回传」。"""
    return max(1, min(batch, sim // SEARCH_BATCH_ROUNDS))


def mcts_search(game, evaluator: Evaluator, sim: int, c_puct: float = 1.6,
                batch: int = 32, noise_eps: float = 0.0, noise_alpha: float = 0.12,
                rng: np.random.Generator | None = None,
                force: set | None = None) -> Node:
    batch = clamp_batch(batch, sim)
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

    def flush():
        """把攒下的叶子送网络并回传，同时清掉 pend（价值由此进入树）。"""
        if not pending:
            return
        outs = evaluator([lf.game for lf, _ in pending])
        for (lf, pa), pol, val in zip(pending, outs[0], outs[1]):
            lf.pend = False
            _expand(lf, pol)
            lf.N = 1
            lf.W = lf.Q = float(val)
            _backup(pa[:-1], -float(val))
        pending.clear()

    while done < sim:
        if len(pending) >= batch:
            flush()
            continue
        sel = _select(root, c_puct)
        if sel is Blocked:
            if not pending:      # 理论上到不了（Blocked 必有在途叶子），兜底防死循环
                break
            flush()
            continue
        done += 1
        if sel is not None:
            pending.append(sel)
    flush()
    return root

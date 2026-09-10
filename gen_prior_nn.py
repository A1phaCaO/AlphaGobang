"""显式用已训好的 9x9 网络，为大棋盘“造先验”：滑窗推理生成软策略标签数据集。

为什么需要它：改了模型架构后主干/头的形状与 9x9 权重不再兼容，无法像 gen_prior.py
那样整块搬卷积权重。但 9x9 网络的“棋感”仍可当一个推理 oracle 来用——五子棋的落子
判断本质是局部的（一个点的好坏只取决于它四方向 ±4 格内的连子），而一个以它为中心的
9x9 窗口正好覆盖 ±4，所以把 9x9 网络在 13x13 盘上滑窗推理，能近似出合法分辨率的
逐点策略先验。

做法：
  对 13x13 每个局面，铺若干 9x9 窗口（默认步幅 1，全部内嵌），每个窗口按训练期的
  “当前行棋方视角”4 平面编码（己/对方/上一手/先手恒置），整批喂进 9x9 网络 -> 每窗口
  81 点 logits；对已占点屏蔽后 softmax，再用以窗口中心为峰的高斯核做 overlap-add
  融合成一张 13x13 先验（越靠近真实落子点、被越多窗口以“中心身份”覆盖的点权重越高）。
  策略标签 = 该先验；价值标签 z = 该局真实终局（由启发式老师驱动对局产生，攻防合理、
  局长正常）。写入 buffer 的 human/ 子目录，之后由 distill_nn.py 蒸馏，或被
  main_train 的训练窗口自动并入（dataset.load_window 默认含 human）。

诚实边界：只有局部战术可信迁移；窗口边界会被网络当成棋盘边界，故贴边点先验偏软，
全局形势判断（超 ±4）不体现。它迁移的是 9x9 的“输出判断”，与能整块搬的“主干特征”
是互补的两回事（后者改架构后搬不动了，才更凸显这个显式先验的价值）。

用法（默认从 13x13、teacher 用 runs/5060/best.pt）：
    uv run python gen_prior_nn.py --games 800 --out runs/13_scratch/buffer
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# 与其它入口一致：import torch 前收紧 BLAS 线程，防 Windows 提交内存被多进程撑爆。
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

try:                       # Windows 控制台默认 cp936，遇不可编码字符会崩；兜底替换而非中止
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

from gobang.config import FEATURE_PLANES
from gobang.dataset import Buffer
from gobang.game import Gomoku
from gobang.model import load_ckpt
from gobang.teacher import Teacher

_DEV = torch.device("cpu")        # 模块级默认；main() 里按 --device 覆盖


def _center_kernel(win: int, sigma: float) -> np.ndarray:
    c = (win - 1) / 2.0
    g = np.exp(-(((np.arange(win)[:, None] - c) ** 2 +
                  (np.arange(win)[None, :] - c) ** 2) / (2 * sigma * sigma)))
    return g.astype(np.float32)


def _softmax_masked(logits_row: np.ndarray, occ_row: np.ndarray) -> np.ndarray:
    x = logits_row - logits_row.max()
    x = np.where(occ_row, -1e9, x)          # 已占点屏蔽，与 mcts._expand / 训练一致
    e = np.exp(x, dtype=np.float64)
    s = e.sum()
    return (e / s).astype(np.float32) if s > 0 else e.astype(np.float32)


def crop_prior(teacher: torch.nn.Module, game: Gomoku, size: int, win: int,
               stride: int, kernel: np.ndarray) -> tuple[np.ndarray, float]:
    """把一个局面经 9x9 网络滑窗推理，融合成 size^2 维策略先验 + 平均价值。"""
    # 当前行棋方视角的全局占位/棋子网格
    mine = game.p0 if game.turn == 0 else game.p1
    opp = game.p1 if game.turn == 0 else game.p0
    gm = np.zeros((size, size), bool)
    go = np.zeros((size, size), bool)
    if len(mine):
        gm[mine // size, mine % size] = True
    if len(opp):
        go[opp // size, opp % size] = True
    occ = gm | go
    first = float(game.turn == 0)           # 先手恒置平面（与 encode 对齐）

    tops = list(range(0, size - win + 1, stride))
    feats, origins = [], []
    for r0 in tops:
        for c0 in tops:
            e = np.zeros((FEATURE_PLANES, win, win), np.float32)
            sl = (slice(r0, r0 + win), slice(c0, c0 + win))
            e[0][gm[sl]] = 1.0
            e[1][go[sl]] = 1.0
            if game.last >= 0:
                lr, lc = game.last // size, game.last % size
                if r0 <= lr < r0 + win and c0 <= lc < c0 + win:
                    e[2, lr - r0, lc - c0] = 1.0
            if first:
                e[3] = 1.0
            feats.append(e)
            origins.append((r0, c0))
    F = torch.from_numpy(np.stack(feats)).to(_DEV)
    with torch.inference_mode():
        logits, value = teacher(F)
    logits = logits.float().cpu().numpy()    # (Wn, win*win)
    value = value.float().cpu().numpy()      # (Wn,)

    acc = np.zeros((size, size), np.float64)
    wsum = np.zeros((size, size), np.float64)
    for k, (r0, c0) in enumerate(origins):
        sl = (slice(r0, r0 + win), slice(c0, c0 + win))
        wocc = occ[sl]
        pi_w = _softmax_masked(logits[k].reshape(win, win), wocc)
        acc[sl] += kernel * pi_w
        wsum[sl] += kernel
    prior = np.where(wsum > 1e-8, acc / np.maximum(wsum, 1e-8), 0.0)
    prior[occ] = 0.0                          # 非法点归零
    total = prior.sum()
    if total <= 0:                            # 极端兜底：无先验质量则均匀落在合法点
        legal = ~occ
        n = int(legal.sum()) or 1
        prior = np.where(legal, 1.0 / n, 0.0)
        total = 1.0
    prior = (prior / total).astype(np.float32)
    return prior.reshape(size * size), float(value.mean())


def generate(teacher, size, games, rng, win, stride, sigma):
    """启发式老师驱动对局；每个局面记录 (特征, 9x9网络先验pi, 真终局z)。"""
    kernel = _center_kernel(win, sigma)
    th = Teacher(size, rng)
    feats, pols, zs = [], [], []
    for _ in range(games):
        g = Gomoku(size)
        gf, gp, gt = [], [], []
        while not g.is_terminal():
            temp = 0.6 if g.n_moves < 8 else 0.2      # 老师驱动，保证攻防合理、局长正常
            hpi = th.policy(g, temp)
            pos = int(rng.choice(hpi.size, p=hpi))
            pi_nn, _v = crop_prior(teacher, g, size, win, stride, kernel)
            gf.append(g.encode())
            gp.append(pi_nn)
            gt.append(g.turn)
            g = g.placed(pos)
        w = g.winner
        zs.append(np.asarray([0.0 if w == 2 else (1.0 if t == w else -1.0)
                              for t in gt], np.float32))
        feats.append(np.stack(gf))
        pols.append(np.stack(gp))
    return (np.concatenate(feats).astype(np.int8),
            np.concatenate(pols).astype(np.float32),
            np.concatenate(zs).astype(np.float32))


def main():
    ap = argparse.ArgumentParser(
        description="滑窗用 9x9 网络为大棋盘显式生成策略先验数据集")
    ap.add_argument("--teacher", default="runs/5060/best.pt", help="源 9x9 权重")
    ap.add_argument("--size", type=int, default=13, help="目标（学生）棋盘边长")
    ap.add_argument("--win", type=int, default=9, help="裁窗边长（=teacher.size 才可信）")
    ap.add_argument("--stride", type=int, default=1, help="滑窗步幅（1 最稠密）")
    ap.add_argument("--sigma", type=float, default=2.0, help="中心高斯核标准差（格）")
    ap.add_argument("--games", type=int, default=800, help="老师驱动的对局数")
    ap.add_argument("--out", required=True,
                    help="目标 buffer 目录（如 runs/13_scratch/buffer），写其 human/")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    global _DEV
    _DEV = torch.device(a.device)
    teacher, tmeta = load_ckpt(a.teacher, a.device)
    ta = teacher.arch
    if a.win != ta["size"]:
        raise SystemExit(f"裁窗边长 {a.win} 与 teacher size {ta['size']} 不一致，"
                         f"teacher 头只会输出 {ta['size']}x{ta['size']} 个点，请确认 teacher。")
    if a.size <= a.win:
        raise SystemExit(f"目标 size({a.size}) 需大于裁窗 size({a.win}) 才有滑窗意义")
    print(f"teacher size={ta['size']} {ta['channels']}ch x {ta['blocks']}bl "
          f"(iter={tmeta.get('iter','?')}) | 目标 size={a.size} | "
          f"窗口 {a.win}x{a.win} stride={a.stride} sigma={a.sigma}")

    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    feats, pols, zs = generate(teacher, a.size, a.games, rng, a.win, a.stride, a.sigma)
    buf = Buffer(a.out)
    buf.append_human(feats, pols, zs)
    print(f"{a.games} 局完成：样本 {len(zs)}，平均局长 {len(zs)/a.games:.0f}，"
          f"用时 {time.time()-t0:.0f}s -> {buf.root}/human")
    print("下一步蒸馏：uv run python distill_nn.py --config <你的新架构配置> "
          f"--data {a.out}")


if __name__ == "__main__":
    main()

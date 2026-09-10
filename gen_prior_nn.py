"""显式用已训好的 9x9 网络，为大棋盘“造先验”：滑窗推理生成软策略标签数据集。

为什么需要它：改了模型架构后主干/头的形状与 9x9 权重不再兼容，无法像 gen_prior.py
那样整块搬卷积权重。但 9x9 网络的“棋感”仍可当一个推理 oracle 来用——五子棋的落子
判断本质是局部的（一个点的好坏只取决于它四方向 ±4 格内的连子），而一个以它为中心的
9x9 窗口正好覆盖 ±4，所以把 9x9 网络在目标棋盘上滑窗推理，能近似出合法分辨率的逐点
策略先验。

做法（两阶段，为把 GPU 吃满）：
  阶段A（CPU）：启发式老师驱动对局，记录每个局面的全分辨率 4 平面编码 + 真终局 z。
  阶段B（GPU）：把所有局面的 9x9 窗口一次性攒成大 batch，分块送 9x9 网络前向；每个
    窗口 logits 对已占点屏蔽后 softmax，再以“窗口中心为峰”的高斯核 overlap-add 融合
    成整盘先验。批量化后 GPU 不再被逐步的 Python 循环饿着。
  策略标签 = 融合先验；价值标签 z = 该局真实终局。写入 buffer 的 human/ 子目录，之后
  由 distill_nn.py 蒸馏，或被 main_train 的训练窗口自动并入（load_window 默认含 human）。

诚实边界：只有局部战术可信迁移；窗口边界会被网络当成棋盘边界，故贴边点先验偏软，
全局形势判断（超 ±4）不体现。它迁移的是 9x9 的“输出判断”，与能整块搬的“主干特征”
是互补的两回事（后者改架构后搬不动了，才更凸显这个显式先验的价值）。

用法（默认 13x13、teacher 用 runs/5060/best.pt；GPU 上跑，--chunk 越大显存越吃满）：
    uv run python gen_prior_nn.py --games 800 --out runs/13_scratch/buffer --device cuda
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

from tqdm import tqdm

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


def _window_index(size: int, win: int, stride: int):
    """滑窗左上角原点集合，及 (窗口, 局部行/列) -> 全局行/列 的 gather 索引。"""
    tops = list(range(0, size - win + 1, stride))
    origins = np.array([(r0, c0) for r0 in tops for c0 in tops], np.intp)  # (Wn,2)
    ii = np.arange(win)
    R = np.broadcast_to(origins[:, 0][:, None, None] + ii[None, :, None],
                        (len(origins), win, win))
    C = np.broadcast_to(origins[:, 1][:, None, None] + ii[None, None, :],
                        (len(origins), win, win))
    return origins, R, C


def prior_batch(teacher, feats: np.ndarray, win: int, stride: int,
                kernel: np.ndarray, idx=None) -> np.ndarray:
    """一批局面的滑窗融合先验。feats:(P,FEATURE_PLANES,S,S) float -> 返回 (P,S*S)。"""
    P, _, S, _ = feats.shape
    origins, R, C = idx if idx is not None else _window_index(S, win, stride)
    Wn = len(origins)
    W = feats[:, :, R, C]                                   # (P,4,Wn,win,win)
    Wf = np.ascontiguousarray(W.transpose(0, 2, 1, 3, 4)).reshape(P * Wn, 4, win, win)
    with torch.inference_mode():
        logits, _val = teacher(torch.from_numpy(Wf).to(_DEV))
    logits = logits.float().cpu().numpy().reshape(P, Wn, win, win)

    occ = (feats[:, 0] + feats[:, 1]) > 0.5                 # (P,S,S)
    win_occ = occ[:, R, C]                                  # (P,Wn,win,win)
    x = logits - logits.max((2, 3), keepdims=True)
    x = np.where(win_occ, -1e9, x)                          # 已占点屏蔽，与训练/搜索一致
    e = np.exp(x, dtype=np.float64)
    pi = (e / np.maximum(e.sum((2, 3), keepdims=True), 1e-30)).astype(np.float64)

    acc = np.zeros((P, S, S), np.float64)
    Pi = np.broadcast_to(np.arange(P, dtype=np.intp)[:, None, None, None],
                         (P, Wn, win, win))
    Rp = np.broadcast_to(R[None], (P, Wn, win, win))
    Cp = np.broadcast_to(C[None], (P, Wn, win, win))
    K = kernel[None, None, :, :]
    np.add.at(acc, (Pi, Rp, Cp), pi * K)                    # overlap-add 融合（按位置分别累加）
    ws = np.zeros((S, S), np.float64)                       # 权重和与位置无关，算一次
    np.add.at(ws, (R, C), np.broadcast_to(K[0], (Wn, win, win)))
    prior = acc / np.maximum(ws[None], 1e-8)
    prior[occ] = 0.0                                        # 非法点归零
    tot = prior.sum((1, 2))
    legal = ~occ
    n_legal = np.maximum(legal.sum((1, 2)), 1)             # 每位置合法点数 -> (P,)
    uniform = np.where(legal, 1.0 / n_legal[:, None, None], 0.0)
    bad = (tot <= 0)[:, None, None]                         # 无质量则兜底均匀
    prior = np.where(bad, uniform, prior / np.maximum(tot[:, None, None], 1e-30))
    return prior.reshape(P, S * S).astype(np.float32)


def generate(teacher, size, games, rng, win, stride, sigma, chunk):
    """阶段A 收集局面+z，阶段B 批量算先验。返回 (feats int8, pols, zs)。"""
    kernel = _center_kernel(win, sigma)
    idx = _window_index(size, win, stride)
    th = Teacher(size, rng)
    feats_l, z_l = [], []
    for _ in tqdm(range(games), total=games, desc="阶段A 启发式对局(CPU)", unit="局"):
        g = Gomoku(size)
        gf, gt = [], []
        while not g.is_terminal():
            temp = 0.6 if g.n_moves < 8 else 0.2            # 老师驱动：攻防合理、局长正常
            hpi = th.policy(g, temp)
            pos = int(rng.choice(hpi.size, p=hpi))
            gf.append(g.encode())
            gt.append(g.turn)
            g = g.placed(pos)
        w = g.winner
        feats_l.append(np.stack(gf))                        # (L,4,S,S) float32
        z_l.append(np.asarray([0.0 if w == 2 else (1.0 if t == w else -1.0)
                               for t in gt], np.float32))
    F = np.concatenate(feats_l)                             # (P,4,S,S)
    Z = np.concatenate(z_l)
    P = len(Z)
    pols = np.empty((P, size * size), np.float32)
    for s in tqdm(range(0, P, chunk), total=(P + chunk - 1) // chunk,
                  desc="阶段B 滑窗先验(GPU批量)", unit="批"):
        e = min(s + chunk, P)
        pols[s:e] = prior_batch(teacher, F[s:e], win, stride, kernel, idx)
    return F.astype(np.int8), pols, Z


def main():
    ap = argparse.ArgumentParser(
        description="滑窗用 9x9 网络为大棋盘显式生成策略先验数据集")
    ap.add_argument("--teacher", default="runs/5060/best.pt", help="源 9x9 权重")
    ap.add_argument("--size", type=int, default=13, help="目标（学生）棋盘边长")
    ap.add_argument("--win", type=int, default=9, help="裁窗边长（=teacher.size 才可信）")
    ap.add_argument("--stride", type=int, default=1, help="滑窗步幅（1 最稠密）")
    ap.add_argument("--sigma", type=float, default=2.0, help="中心高斯核标准差（格）")
    ap.add_argument("--games", type=int, default=800, help="老师驱动的对局数")
    ap.add_argument("--chunk", type=int, default=4096,
                    help="每次 GPU 前向的窗口批大小（越大显存/算力吃越满，5060 可加大）")
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
          f"窗口 {a.win}x{a.win} stride={a.stride} sigma={a.sigma} device={a.device}")

    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    feats, pols, zs = generate(teacher, a.size, a.games, rng,
                               a.win, a.stride, a.sigma, a.chunk)
    buf = Buffer(a.out)
    buf.append_human(feats, pols, zs)
    print(f"{a.games} 局完成：样本 {len(zs)}，平均局长 {len(zs)/a.games:.0f}，"
          f"用时 {time.time()-t0:.0f}s -> {buf.root}/human")
    print("下一步蒸馏：uv run python distill_nn.py --config <你的新架构配置> "
          f"--data {a.out}")


if __name__ == "__main__":
    main()

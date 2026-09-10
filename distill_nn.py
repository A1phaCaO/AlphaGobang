"""把 9x9 显式先验（gen_prior_nn.py 造的数据集）蒸馏进新架构学生网络。

改架构后主干卷积形状对不上，gen_prior.py 那套“整块搬权重”失效；于是退而求其次——
不搬权重，搬“判断”：用现成的 trainer.train_steps 在先验数据集上做 KL(先验‖学生)
+ MSE(z)，把随机初始化的学生拟合到 9x9 网络的逐点落子品味上，产出一份蒸馏后的
权重当作新 run 的 latest.pt（iter=0）。之后再 main_train --resume 接着自对弈提升。

这样新架构学生一上来就不是乱走（省掉随机冷启动那段“局长很长、自对弈很慢”的时间），
但它学的是 teacher 的“局部输出判断”，不是 teacher 的特征，故蒸馏一轮后仍要靠自对弈
把全局与更大棋盘的适配练出来。

用法：
    uv run python gen_prior_nn.py --games 800 --out runs/13_scratch/buffer
    uv run python distill_nn.py --config train_13_scratch.json --data runs/13_scratch/buffer
    uv run python main_train.py --config train_13_scratch.json --resume
默认 out 用配置里的 run_dir/latest.pt；不覆盖已有进度（已存在会提示，除非 --force）。
"""
from __future__ import annotations

import argparse
import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import torch

from gobang.config import Config, load_param_file
from gobang.dataset import Buffer
from gobang.game import Gomoku
from gobang.model import build_net, load_ckpt, save_ckpt
from gobang.trainer import train_steps


def load_cfg(path: str) -> Config:
    cfg = Config()
    for k, v in load_param_file(path).items():
        if not k.startswith("_") and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def main():
    ap = argparse.ArgumentParser(
        description="用 9x9 显式先验数据蒸馏新架构学生网络")
    ap.add_argument("--config", required=True, help="学生（新架构）参数文件")
    ap.add_argument("--data", default=None,
                    help="先验数据 buffer 目录（默认取配置 run_dir/buffer）")
    ap.add_argument("--out", default=None,
                    help="蒸馏后权重输出（默认 <run_dir>/latest.pt）")
    ap.add_argument("--epochs", type=float, default=8.0,
                    help="先验数据集上的等效训练轮数（steps = 样本/批 * epochs）")
    ap.add_argument("--steps", type=int, default=0,
                    help="直接指定训练步数，>0 时忽略 --epochs")
    ap.add_argument("--lr", type=float, default=None, help="蒸馏学习率，默认取配置 lr_init")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true",
                    help="即使输出已存在也覆盖（默认拒绝，避免抹掉自对弈进度）")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    data_dir = a.data or os.path.join(cfg.run_dir, "buffer")
    out = a.out or os.path.join(cfg.run_dir, "latest.pt")

    if os.path.exists(out) and not a.force:
        raise SystemExit(f"{out} 已存在：可能已训练过。确认要覆盖再加 --force")

    buf = Buffer(data_dir).load_human()
    if buf is None:
        raise SystemExit(f"未找到先验数据 {data_dir}/human，先跑 "
                         f"uv run python gen_prior_nn.py --out {data_dir}")
    dsize = int(buf["feats"].shape[-1])
    if dsize != cfg.size:
        raise SystemExit(f"先验数据是 {dsize}x{dsize}，与配置 size={cfg.size} 不符；"
                         f"用同一 size 重新 gen_prior_nn")

    dev = a.device
    net = build_net(cfg)
    n = int(buf["zs"].shape[0])
    steps = a.steps or max(1, int(n / cfg.batch_size * a.epochs))
    lr = a.lr if a.lr is not None else cfg.lr_init
    print(f"学生新架构 size={cfg.size} {cfg.channels}ch×{cfg.blocks}bl "
          f"参数量 {sum(p.numel() for p in net.parameters()):,}")
    print(f"先验样本 {n}（{dsize}x{dsize}）| 蒸馏 {steps} 步 (lr={lr}, "
          f"batch={cfg.batch_size}) -> {out}")

    r = train_steps(net, buf, cfg, steps, lr, dev, log=None, seed_mix=7,
                    progress=True)
    print(f"蒸馏完成：ce {r['ce']:.3f}  mse {r['mse']:.3f}")

    save_ckpt(out, net, {"iter": 0, "init": "nn-distill",
                         "prior_samples": n, "distill_steps": steps,
                         "ce": round(r["ce"], 4), "mse": round(r["mse"], 4)})
    # 冒烟：重载确认形状与前向可用
    rl, _ = load_ckpt(out, "cpu")
    with torch.inference_mode():
        lg, vl = rl(torch.from_numpy(Gomoku(cfg.size).encode())[None])
    assert lg.shape == (1, cfg.size * cfg.size)
    print(f"已写蒸馏权重 -> {out}（冒烟 policy{tuple(lg.shape)} value {float(vl):+.2f}）")
    print(f"接着自对弈续训：uv run python main_train.py --config {a.config} --resume")


if __name__ == "__main__":
    main()

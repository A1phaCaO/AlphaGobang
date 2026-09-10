"""把已训好的小棋盘权重迁移成大棋盘的“热启动”初始权重（只生成、不开训）。

与 `main_train.py --warmfrom` 共用 gobang.model.build_warm_net，是同一条迁移逻辑
的独立入口：当你只想拿到热启动权重、或想先检查搬运/重学了哪些张量、暂不启动
训练时用这个；正式开训直接一条命令 `main_train --warmfrom` 更省事。

原理见 model.py：主干是全卷积 + BatchNorm，权重形状不含 H/W，换分辨率能整块搬；
只有吃 Flatten 的两个 Linear（policy_head.4 / value_head.4）被 size² 锁死拷不动，
重新随机初始化。故继承的是棋形特征（trunk），落子决策映射与价值头需靠后续自对弈
重学——刚迁移完的模型尚不会下棋，收益是省掉从零摸索主干的冷启动。

用法：
    uv run python gen_prior.py                               # 默认 9x9 best -> 13x13
    uv run python gen_prior.py --teacher runs/5060/best.pt \
        --size 13 --out runs/5060_13/latest.pt
"""
from __future__ import annotations

import argparse
import os

# 与 main_train.py 一致：import torch 前收紧 BLAS 线程，防 Windows 提交内存被撑爆。
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import torch

from gobang.game import Gomoku
from gobang.model import build_warm_net, load_ckpt, save_ckpt


def main():
    ap = argparse.ArgumentParser(description="小棋盘权重 -> 大棋盘热启动初始权重")
    ap.add_argument("--teacher", default="runs/5060/best.pt",
                    help="源权重 checkpoint（9x9 那份）")
    ap.add_argument("--size", type=int, default=13, help="目标棋盘边长")
    ap.add_argument("--out", default="runs/5060_13/latest.pt",
                    help="输出的热启动 ckpt（应落在新 run 的 latest.pt）")
    a = ap.parse_args()

    if not os.path.exists(a.teacher):
        raise SystemExit(f"找不到源权重：{a.teacher}")
    teacher, meta = load_ckpt(a.teacher, "cpu")
    arch = teacher.arch
    if arch["size"] == a.size:
        raise SystemExit(f"目标 size 与 teacher 相同（都是 {a.size}），无需迁移")

    print(f"teacher：size={arch['size']} channels={arch['channels']} "
          f"blocks={arch['blocks']}（iter={meta.get('iter', '?')}）")

    student, copied, reinit = build_warm_net(a.size, arch["channels"],
                                             arch["blocks"], teacher)
    print(f"student：size={a.size} channels={arch['channels']} "
          f"blocks={arch['blocks']}（主干同构才能整块搬运）")

    full = student.state_dict()
    nc = sum(int(full[k].numel()) for k in copied)
    nr = sum(int(full[k].numel()) for k in reinit)
    print(f"\n搬运 {len(copied)} 个张量（{nc:,} 参数），重学 {len(reinit)} 个"
          f"（{nr:,} 参数）：")
    for k in reinit:
        print(f"  重新随机初始化：{k}  {tuple(full[k].shape)}")

    # 冒烟：确保目标棋盘上能真跑一次前向，且 policy 维度 = size²。
    student.eval()
    with torch.inference_mode():
        logits, value = student(torch.from_numpy(Gomoku(a.size).encode())[None])
    assert logits.shape == (1, a.size * a.size), logits.shape
    assert value.shape == (1,), value.shape
    print(f"\n冒烟通过：{a.size}x{a.size} 前向 policy {tuple(logits.shape)} "
          f"value {float(value):+.3f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    save_ckpt(a.out, student, {"iter": 0, "init_from": a.teacher,
                               "init_arch": dict(arch, size=a.size)})
    print(f"已写热启动权重 -> {a.out}")
    print(f"续训：uv run python main_train.py --config train_13_5060.json --resume "
          f"（train_13_5060.json 的 run_dir 需与 {a.out} 同目录）")


if __name__ == "__main__":
    main()

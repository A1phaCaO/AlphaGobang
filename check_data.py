"""自对弈数据体检：先确认标签里有没有内容，再谈调参。

训练不收敛时别急着动 lr——九成是标签本身没信息。两个指标要分开看：

  gap = ln(合法点数) - H(策略目标)   搜索质量。gap 接近 0 表示访问分布是均匀的，
                                  MCTS 退化成「每个可走点各 1 次访问」（典型成因是
                                  mcts_batch >= sim_selfplay），自对弈等于随机落子。
  ce - H(窗口)                      学习能力。ce 取 meta.json 当轮均值，H 按
                                  buffer_window 拼同一批数据算。ce 贴着 H 说明网络
                                  已经把标签学干了，再加训练步数只是拟合噪声。

注意 gap 只看全盘均值会被深盘骗过去：残局本来就没几个可走点，战术强制掩码还会把
目标压成一两点，所以判定用开局档（前 20 手）——那才是搜索该发力的地方。

用法：
  uv run python check_data.py                        # 默认 runs/default
  uv run python check_data.py runs/5060 --iters 1-8  # 指定 run 与轮次
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BANDS = ((0, 6), (6, 20), (20, 40), (40, 60), (60, 99))


def stat(feats: np.ndarray, pols: np.ndarray) -> dict:
    """一个数据块的目标熵/均匀度。feats 平面 0、1 是己方与对方棋子。"""
    size = int(round(pols.shape[1] ** 0.5))
    ply = feats[:, 0].sum((1, 2)) + feats[:, 1].sum((1, 2))
    legal = np.maximum(size * size - ply, 1)
    ent = -(pols * np.log(np.maximum(pols, 1e-12))).sum(1)
    flat = np.log(legal)
    maxp = pols.max(1)
    out = {"n": len(ent), "ent": float(ent.mean()), "flat": float(flat.mean()),
           "maxp": float(maxp.mean()), "uniform": float((maxp < 0.05).mean())}
    out["band"] = {}
    for lo, hi in BANDS:
        m = (ply >= lo) & (ply < hi)
        k = int(m.sum())
        out["band"][(lo, hi)] = {
            "n": k, "ent": float(ent[m].mean()) if k else 0.0,
            "flat": float(flat[m].mean()) if k else 0.0,
            "maxp": float(maxp[m].mean()) if k else 0.0}
    return out


def pick(spec: str | None, have: list[int]) -> list[int]:
    if not spec:
        return have
    want = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        try:
            want.update(range(int(lo), int(hi or lo) + 1))
        except ValueError:
            raise SystemExit(f"--iters 看不懂：{part}（例子：1,3,5-8）")
    return [i for i in have if i in want]


def wmean(rows: list[dict], key: str) -> float:
    n = sum(r["n"] for r in rows)
    return sum(r[key] * r["n"] for r in rows) / max(1, n)


def main():
    p = argparse.ArgumentParser(description="自对弈数据体检")
    p.add_argument("run_dir", nargs="?", default="runs/default")
    p.add_argument("--iters", help="轮次，如 1,3,5-8（默认全部）")
    args = p.parse_args()

    buf = Path(args.run_dir) / "buffer"
    files = sorted(buf.glob("iter_*.npz"))
    if not files:
        raise SystemExit(f"{buf} 下没有 iter_*.npz")
    have = [int(f.stem.split("_")[1]) for f in files]
    todo = pick(args.iters, have)
    if not todo:
        raise SystemExit(f"--iters 与现有轮次 {have} 无交集")

    hist, window = {}, 10
    meta_path = Path(args.run_dir) / "meta.json"
    ce_ok = True
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            hist = {e["iter"]: e for e in meta.get("history", [])}
            window = int(meta.get("config", {}).get("buffer_window", 10)) or 10
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
            ce_ok = False
            print(f"meta.json 解析失败（{e}）：跳过 ce 对比。多半是跨机拷贝时截断的；"
                  "--resume 时主循环会把它备份成 meta.bad 并重记历史，权重与轮次不受影响")
    else:
        ce_ok = False
    hdir = Path(args.run_dir) / "buffer" / "human"
    if any(hdir.glob("h_*.npz")):
        # dataset.load_window 会把 human 池永久拼进训练窗口，这里只统计自对弈部分，
        # 所以 ce 会比 H_tgt 低一截（人类 one-hot 更好学），不是 bug。
        print("注意：该 run 有 human/ 冷启动数据，它也参与训练窗口，"
              "但不在此表的 H_tgt 里，ce-Hw 会偏负")

    # 逐个文件算完就丢，150 轮的 run 全读进内存要好几 GB
    stats = {}
    for it, f in zip(have, files):
        if it < max(have[0], min(todo) - window + 1):
            continue
        with np.load(f) as d:
            stats[it] = stat(d["feats"], d["pols"])
            if it in todo:
                zs = d["zs"]
                stats[it]["draw"] = float((np.abs(zs) < 0.1).mean())

    # 表头用 ASCII：中文占两格宽，按字符数对齐会错位
    print(f"{'iter':>5} {'n':>8} {'H_tgt':>7} {'ln_leg':>8} {'gap':>6} "
          f"{'maxP':>6} {'unif':>6} {'draw':>6} {'ce':>7} {'ce-Hw':>7}")
    print("      H_tgt=策略目标熵 ln_leg=ln(合法点数) gap=两者之差(搜索有没有思考)"
          " unif=最大概率<0.05 占比 draw=z=0 占比 ce-Hw=损失减标签熵(还剩多少可学)")
    for it in todo:
        s = stats[it]
        rows = [stats[j] for j in range(max(have[0], it - window + 1), it + 1)
                if j in stats]
        win_h = wmean(rows, "ent")
        ce = hist.get(it, {}).get("ce")
        tail = (f"{ce:7.3f} {ce - win_h:+8.3f}" if ce is not None and ce_ok
                else "      -        -")
        print(f"{it:5d} {s['n']:8d} {s['ent']:7.3f} {s['flat']:8.3f} "
              f"{s['flat'] - s['ent']:6.2f} {s['maxp']:6.3f} "
              f"{s['uniform']:6.0%} {s.get('draw', 0):5.0%} {tail}")

    last = stats[todo[-1]]
    print(f"\niter {todo[-1]} 手数分档（gap≈0 的档就是随机落子）")
    for lo, hi in BANDS:
        b = last["band"][(lo, hi)]
        if b["n"]:
            print(f"  ply {lo:2d}-{hi - 1:2d} n={b['n']:6d}  H {b['ent']:5.2f}  "
                  f"ln(legal) {b['flat']:5.2f}  gap {b['flat'] - b['ent']:5.2f}  "
                  f"maxP {b['maxp']:.3f}")
    early = [last["band"][k] for k in BANDS[:2]]
    e_n = sum(b["n"] for b in early)
    e_gap = (sum((b["flat"] - b["ent"]) * b["n"] for b in early) / e_n
             if e_n else 0.0)
    print(f"\n判定：开局档（前 20 手，占样本 {e_n / max(1, last['n']):.0%}）"
          f"gap={e_gap:.2f} nats，{last['uniform']:.0%} 的样本最大概率 < 0.05")
    if e_gap < 0.2:
        print("  搜索退化：开局目标接近均匀分布 = 随机落子。先查 mcts_batch 是不是"
              " >= sim_selfplay（现在由 mcts.clamp_batch 自动收紧），调 lr 没用。")
    elif e_gap < 0.6:
        print("  搜索偏弱：目标有区分度但不锐。看 c_puct 是否偏大、sim_selfplay 是否"
              "偏低、噪声 noise_eps/noise_alpha 是否盖过了先验。")
    else:
        print("  搜索有区分度：策略目标带真实选择，ce 才是有意义的学习信号。")
    ce = hist.get(todo[-1], {}).get("ce")
    if ce is not None and ce_ok and ce - wmean(
            [stats[j] for j in range(max(have[0], todo[-1] - window + 1),
                                     todo[-1] + 1) if j in stats], "ent") < 0.05:
        print("  且 ce 已贴着标签熵（差 <0.05）：网络把这批标签学干了，卡住的是数据"
              "质量，不是优化器。")


if __name__ == "__main__":
    main()
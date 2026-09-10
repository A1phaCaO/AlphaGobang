"""主训练脚本：AlphaGo Zero 式自对弈迭代训练，支持断点续训。

细节参数不堆在命令行上：默认加载工作目录的 train.json（--config 可换文件），
或用 --preset 一键选用代码内置参数组（见 gobang/config.py 的 PRESETS）。
优先级：命令行单项 > --preset > 参数文件 > 内置默认。

每轮收尾时同时投放 本轮评估(独立 CPU 进程池) 与 下一轮自对弈(GPU 进程池)，
评估结果延后一轮收集：训练吃 GPU、评估吃 CPU，完全重叠，训练结束后的串行
等待基本归零（日志「评估等待 Xs」即实际阻塞时长）。GPU 评估与训练抢同一块
限功率显卡会双双变慢（1650 实测训练 83s→142s+），因此提交内存不足、CPU
评估池建不起来时宁可退回串行。

用法示例：
    uv run python main_train.py                      # 标准参数（改 train.json 即可）
    uv run python main_train.py --resume             # 中断后继续
    uv run python main_train.py --config runs/a.json # 换一份参数文件
    uv run python main_train.py --preset smoke       # 1~2 分钟冒烟
    uv run python main_train.py --preset strong --games 1000   # 预设上再覆盖单项
    uv run python main_train.py --amp                # 开启混合精度（默认关，见 README）
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
import time
from pathlib import Path

# spawn 子进程 import torch 时 OpenBLAS 会按核数预分配线程 arena，十来个进程
# 就能撑爆 Windows 提交限制（页面文件太小 / 随机分配失败）。必须在 import
# torch 之前收紧，子进程经环境继承同样生效；训练本身跑在 GPU 上不受影响。
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import torch

from gobang.config import (DEFAULT_PARAM_FILE, PRESETS, Config,
                           load_param_file)
from gobang.dataset import Buffer
from gobang.model import build_net, load_ckpt, save_ckpt
from gobang.selfplay import (collect_ev, collect_sp, eval_games,
                             eval_parallel, make_eval_pool, make_ev_jobs,
                             make_pool, make_sp_jobs, play_one_game,
                             resolve_workers, run_selfplay, submit)
from gobang.trainer import train_steps

_BASE = Config()


class ParamFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """各项 argparse 默认值留 None 作“未显式指定”哨兵，帮助文本改显示 Config 真实默认。"""

    def add_argument(self, action):
        if action.dest != "help" and action.default is None:
            v = getattr(_BASE, action.dest, None)
            if v is not None:
                action.help = ((action.help or "") + f"（默认 {v}）").strip()
        super().add_argument(action)


def parse_args() -> tuple[Config, list[str]]:
    p = argparse.ArgumentParser(
        description="五子棋自对弈强化学习训练",
        formatter_class=ParamFormatter)
    p.add_argument("--config", default=DEFAULT_PARAM_FILE,
                   help="JSON 参数文件，键名与 Config 字段一致；文件不存在则跳过")
    p.add_argument("--preset", choices=sorted(PRESETS),
                   help=f"内置参数组（{', '.join(sorted(PRESETS))}），"
                        "优先于参数文件、低于命令行单项")
    p.add_argument("--resume", action="store_true", help="从 latest.pt 断点续训")
    p.add_argument("--size", type=int, help="棋盘边长")
    p.add_argument("--channels", type=int, help="CNN 通道数")
    p.add_argument("--blocks", type=int, help="残差块数")
    p.add_argument("--iterations", type=int, help="总迭代轮数")
    p.add_argument("--games", dest="games_per_iter", type=int,
                   help="每轮自对弈局数")
    p.add_argument("--sim", dest="sim_selfplay", type=int,
                   help="自对弈每步 MCTS 模拟数")
    p.add_argument("--steps", dest="train_steps", type=int, help="每轮训练步数")
    p.add_argument("--batch", dest="batch_size", type=int, help="训练批大小")
    p.add_argument("--lr", dest="lr_init", type=float, help="初始学习率")
    p.add_argument("--c-puct", type=float, help="PUCT 探索系数")
    p.add_argument("--mcts-batch", type=int, help="MCTS 每批网络评估的叶子数")
    p.add_argument("--eval-games", type=int, help="评估对局数")
    p.add_argument("--eval-sim", dest="sim_eval", type=int,
                   help="评估 MCTS 模拟数")
    p.add_argument("--eval-threshold", type=float, help="晋升所需得分率")
    p.add_argument("--eval-device", choices=("cpu", "cuda"),
                   help="晋升评估进程设备：cpu=独立进程池与训练重叠（默认），"
                        "cuda=与自对弈共享 GPU 进程池")
    p.add_argument("--eval-workers", type=int,
                   help="评估 CPU 进程数，0 为自动（核数-4）")
    p.add_argument("--no-eval", action="store_true",
                   help="跳过评估，每轮直接将新模型设为最佳")
    p.add_argument("--buffer-window", type=int, help="训练取最近多少轮数据")
    p.add_argument("--human-epochs", type=int,
                   help=">0 时每轮额外用人机对局数据训练 N 个 epoch（冷启动）")
    p.add_argument("--workers", type=int, help="自对弈进程数，0 为自动")
    p.add_argument("--device", help="训练设备 auto/cuda/cpu")
    p.add_argument("--sp-device", help="自对弈推理设备 auto/cuda/cpu")
    p.add_argument("--amp", dest="use_amp", action=argparse.BooleanOptionalAction,
                   help="混合精度训练（仅 GPU 生效，--no-amp 关闭）")
    p.add_argument("--run-dir", help="输出目录")
    p.add_argument("--seed", type=int)
    args = vars(p.parse_args())

    cfg = Config()
    cfg_path = args.pop("config")
    preset_name = args.pop("preset") or "default"
    if preset_name not in PRESETS:
        raise SystemExit(f"内部错误：PRESETS 缺少预设 '{preset_name}'")
    preset = PRESETS[preset_name]
    file_cfg = load_param_file(cfg_path)
    field_names = {f.name for f in dataclasses.fields(cfg)}
    stray = sorted(set(args) - field_names)
    if stray:
        raise SystemExit(f"内部错误：命令行参数未映射到 Config 字段：{stray}")
    for k in field_names:
        for src in (args, preset, file_cfg):   # 命令行 > 预设 > 参数文件
            v = src.get(k)
            if v is not None:
                setattr(cfg, k, v)
                break
    notes = [f"参数文件 {cfg_path}：已加载 {len(file_cfg)} 项" if file_cfg
             else f"参数文件 {cfg_path} 不存在，使用内置默认参数"]
    if preset_name != "default":
        notes.append(f"预设 {preset_name}：{len(preset)} 项参数生效")
    return cfg, notes


class Tee:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        try:
            from tqdm import tqdm
            tqdm.write(str(msg))
        except Exception:
            print(str(msg), flush=True)
        self.f.write(str(msg) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def load_meta(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"iter": 0, "history": []}


def save_meta(path: Path, meta: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def main():
    cfg, notes = parse_args()
    torch.manual_seed(cfg.seed)
    run = Path(cfg.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    log = Tee(run / "train.log")
    sys.stdout = sys.stderr  # 让 tqdm 与 print 输出流统一，日志顺序不串
    for n in notes:
        log(n)

    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif cfg.device == "cuda" and not torch.cuda.is_available():
        log("CUDA 不可用，训练设备回退到 cpu")
        cfg.device = "cpu"
    if cfg.use_amp and cfg.device == "cpu":
        cfg.use_amp = False
        log("训练设备为 cpu，AMP 已自动关闭")
    if cfg.sp_device == "auto":
        cfg.sp_device = cfg.device
    cfg.workers = resolve_workers(cfg)
    if cfg.device != "cuda":
        cfg.eval_device = cfg.sp_device   # 纯 CPU 机器不另开评估进程池，避免抢核

    buffer = Buffer(run / "buffer")
    latest, best = run / "latest.pt", run / "best.pt"
    meta_path = run / "meta.json"
    meta = load_meta(meta_path)

    start = 1
    if cfg.resume and latest.exists():
        net, m = load_ckpt(latest, cfg.device)
        cfg.size = net.size
        cfg.channels = net.arch["channels"]
        cfg.blocks = net.arch["blocks"]
        start = int(m.get("iter", 0)) + 1
        log(f"续训：从第 {start} 轮开始（架构 size={cfg.size} "
            f"channels={cfg.channels} blocks={cfg.blocks}）")
    else:
        net = build_net(cfg)
        if cfg.resume:
            log("未找到 latest.pt，从头开始训练")
        save_ckpt(latest, net, {"iter": 0})
        log(f"新建网络 size={cfg.size} 参数量 "
            f"{sum(p.numel() for p in net.parameters()):,}")

    if not best.exists():
        shutil.copyfile(latest, best)
        log("best.pt 不存在，已用当前模型初始化")
    meta["best_iter"] = meta.get("best_iter", start - 1)
    meta["config"] = cfg.to_dict()

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(run / "tb"))
    except Exception:
        tb = None

    if cfg.eval_device == "auto":
        cfg.eval_device = cfg.device
    pool = make_pool(cfg) if cfg.workers > 1 else None
    want_cpu_eval = (not cfg.no_eval and cfg.device == "cuda"
                     and cfg.eval_device == "cpu")
    ev_pool = pool
    if want_cpu_eval:
        # 评估 CPU 池延迟到首轮训练结束后再建：避免与 6 个 CUDA 自对弈进程
        # 同时 import torch 撑爆 Windows 提交内存（页面文件太小 1455）。
        ev_pool = None

    def ensure_ev_pool():
        nonlocal ev_pool
        if want_cpu_eval and ev_pool is None:
            ev_pool = make_eval_pool(cfg) or pool
            if ev_pool is None:
                log("提交内存不足，评估退回串行阻塞")
            elif ev_pool is pool:
                log("提交内存不足，CPU 评估池放弃；GPU 评估与训练抢限功率"
                    "显卡反而拖慢双方，评估退回串行")
        return ev_pool

    if cfg.no_eval:
        eval_txt = "评估 跳过"
    elif want_cpu_eval:
        eval_txt = (f"评估 {cfg.eval_games} 局 @{cfg.eval_threshold}"
                    "（CPU 池自动收缩进程数，与训练重叠；建不起来则串行）")
    else:
        eval_txt = f"评估 {cfg.eval_games} 局 @{cfg.eval_threshold}（串行）"
    log(f"训练设备 {cfg.device}（AMP {'开' if cfg.use_amp else '关'}）| "
        f"自对弈 {cfg.workers} 进程 x {cfg.sp_device}（进程池跨轮复用）| "
        f"轮数 {start}..{cfg.iterations} | 每轮 {cfg.games_per_iter} 局 x "
        f"sim{cfg.sim_selfplay} | 训练 {cfg.train_steps} 步 | {eval_txt}")
    log("Ctrl+C 随时中断（当前轮的自对弈/训练会作废），之后 --resume 续训\n")

    def emit(base, sp, tl, lr, wr, promoted, sec):
        """记档+落盘+日志。base 含 iter/t_selfplay/t_train/t_eval。"""
        it = base["iter"]
        entry = dict(base)
        entry.update({
            "games": sp["games"], "avg_len": round(sp["avg_len"], 1),
            "black_win_rate": round(sp["black_wins"] / max(1, sp["games"]), 3),
            "draws": sp["draws"], "resigns": sp["resigns"],
            "ce": round(tl["ce"], 4), "mse": round(tl["mse"], 4),
            "samples": tl["samples"], "lr": lr,
            "eval_wr": None if cfg.no_eval else round(wr, 3),
            "promoted": promoted, "sec": sec})
        meta["iter"] = it
        meta["history"].append(entry)
        save_meta(meta_path, meta)
        if tb:
            tb.add_scalar("train/ce", entry["ce"], it)
            tb.add_scalar("train/mse", entry["mse"], it)
            tb.add_scalar("train/lr", lr, it)
            tb.add_scalar("selfplay/avg_len", sp["avg_len"], it)
            tb.add_scalar("selfplay/black_win_rate", entry["black_win_rate"], it)
            tb.add_scalar("selfplay/samples", tl["samples"], it)
            if not cfg.no_eval:
                tb.add_scalar("eval/score_vs_best", wr, it)
        ev_s = "auto" if cfg.no_eval else (
            f"{wr:.2f} {'晋升' if promoted else '保持'}")
        h_txt = (f" 人机{tl['h_steps']}步ce{tl['h_ce']}"
                 if "h_steps" in tl else "")
        eta = sec * (cfg.iterations - it) / 60
        log(f"[{it}/{cfg.iterations}] 样本 {tl['samples']} "
            f"ce {entry['ce']:.3f} mse {entry['mse']:.3f}{h_txt} | "
            f"局长 {sp['avg_len']:.0f} 先手胜 {entry['black_win_rate']:.0%} "
            f"投子 {sp['resigns']} | 评估 {ev_s} 等待{entry['t_eval']}s | "
            f"本轮 {sec}s 自对弈{entry['t_selfplay']}s 训练{entry['t_train']}s "
            f"预计剩余 {eta:.0f}min\n")

    def take_pending(pending):
        """收上一轮评估并晋升记档。此刻 latest.pt 仍是被评那轮的权重。
        评估本就在后台跑完，这里的等待通常 ≈ 0s（记入 t_eval）。"""
        pit, pf, pbase, psp, ptl, plr, psec = pending
        te = time.time()
        wr, _ev = collect_ev(pf)
        pbase["t_eval"] = round(time.time() - te)
        promoted = wr >= cfg.eval_threshold
        if promoted:
            shutil.copyfile(latest, best)
            meta["best_iter"] = pit
        emit(pbase, psp, ptl, plr, wr, promoted, psec)

    pending = None          # (it, ev_futs, base, sp, tl, lr, t0)
    sp_futs = None
    interrupted = False
    try:
        if pool is not None:
            sp_futs = submit(pool, play_one_game,
                             make_sp_jobs(cfg, str(latest), start))
        for it in range(start, cfg.iterations + 1):
            t0 = time.time()
            lr = cfg.lr_init * cfg.lr_decay ** ((it - 1) // cfg.lr_decay_every)

            if pool is not None:
                feats, pols, zs, sp = collect_sp(sp_futs)
            else:
                feats, pols, zs, sp = run_selfplay(cfg, str(latest),
                                                   pool=None, ver=it)
            buffer.append_iter(it, feats, pols, zs)
            del feats, pols, zs
            t_sp = time.time() - t0

            buf = buffer.load_window(it, cfg.buffer_window)
            tl = train_steps(net, buf, cfg, cfg.train_steps, lr, cfg.device,
                             seed_mix=it * 4 + 1)
            del buf
            t_tr = time.time() - t0 - t_sp
            if cfg.human_epochs:
                hbuf = buffer.load_human()
                if hbuf is not None:
                    hn = int(hbuf["zs"].shape[0])
                    h_steps = max(1, int(hn / cfg.batch_size * cfg.human_epochs))
                    th = train_steps(net, hbuf, cfg, h_steps, lr, cfg.device,
                                     seed_mix=it * 4 + 2)
                    tl["h_steps"] = h_steps
                    tl["h_ce"] = round(th["ce"], 3)
                    del hbuf

            if pending is not None:
                # 评估在「收自对弈+训练」期间并行完成，这里通常零等待
                take_pending(pending)
                pending = None

            save_ckpt(latest, net, {"iter": it, "loss": tl})
            if pool is not None and it < cfg.iterations:
                # 先投下一轮自对弈：与评估共享池时不被评估任务堵在后面
                sp_futs = submit(pool, play_one_game,
                                 make_sp_jobs(cfg, str(latest), it + 1))
            base = {"iter": it, "t_selfplay": round(t_sp),
                    "t_train": round(t_tr), "t_eval": 0}
            if cfg.no_eval:
                shutil.copyfile(latest, best)
                meta["best_iter"] = it
                emit(base, sp, tl, lr, 1.0, True, round(time.time() - t0))
            else:
                tgt = ensure_ev_pool() if want_cpu_eval else None
                if tgt is not None and tgt is not pool:
                    # 只有 CPU 评估池才值得跨轮重叠：GPU 与训练抢同一块
                    # 限功率显卡会双双变慢（1650 实测训练 83s→142s+）
                    jobs = make_ev_jobs(cfg, str(latest), str(best),
                                        cfg.eval_games, cfg.seed * 10000 + it,
                                        tgt.gobang_workers)
                    # sec 按投放时刻结算：评估延到下一轮收但不占本轮节拍；
                    # 收集时刻的真实阻塞另记 t_eval
                    pending = (it, submit(tgt, eval_games, jobs), base,
                               sp, tl, lr, round(time.time() - t0))
                else:               # GPU 评估：串行跑完再走下一轮（不与训练抢卡）
                    te = time.time()
                    wr, _ev = eval_parallel(cfg, str(latest), str(best),
                                            games=cfg.eval_games,
                                            seed=cfg.seed * 10000 + it,
                                            pool=pool)
                    base["t_eval"] = round(time.time() - te)
                    promoted = wr >= cfg.eval_threshold
                    if promoted:
                        shutil.copyfile(latest, best)
                        meta["best_iter"] = it
                    emit(base, sp, tl, lr, wr, promoted,
                         round(time.time() - t0))
        if pending is not None:
            take_pending(pending)
    except KeyboardInterrupt:
        interrupted = True
        log("\n已中断。上一轮完整进度已保存，用 --resume 继续。")
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        if ev_pool is not None and ev_pool is not pool:
            ev_pool.terminate()
            ev_pool.join()
    if tb:
        tb.close()
    log.close()
    if not interrupted:
        print("训练完成。对弈：uv run python play.py")


if __name__ == "__main__":
    main()

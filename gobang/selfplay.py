"""多进程自我对弈与并行评估。Pool 跨轮复用（省掉每轮 spawn/CUDA 初始化）。

任务采用"提交/收集"两段式：一轮收尾时把 本轮评估 与 下一轮自对弈 一起投进
进程池，主进程先去忙别的，稍后再收结果——评估与自对弈/训练重叠，训练后的
串行等待基本归零。Agent 缓存按模型文件 (路径, mtime) 索引：文件更新后自动
重载，两类任务并发投放也不会拿到旧权重。
"""
from __future__ import annotations

import multiprocessing as mp
import os
from collections import OrderedDict

import numpy as np
import torch

from .agent import Agent
from .config import (CPU_WORKER_RESERVE, CPU_WORKER_COMMIT_MB,
                     CUDA_WORKER_CAP, EVAL_COMMIT_HEADROOM_MB,
                     EVAL_WORKER_RESERVE)
from .game import Gomoku

_CACHE: OrderedDict[tuple, Agent] = OrderedDict()
_CACHE_MAX = 4          # 4GB 显存下同时驻留几份小网络推理副本没有问题
_DEVICE = "cpu"


def _init_worker(device: str):
    global _DEVICE
    _DEVICE = device
    torch.set_num_threads(1)


def _get_agent(path: str, c_puct: float, batch: int) -> Agent:
    key = (path, os.path.getmtime(path), _DEVICE, c_puct, batch)
    agent = _CACHE.get(key)
    if agent is None:
        agent = Agent(path, device=_DEVICE, c_puct=c_puct, batch=batch)
        _CACHE[key] = agent
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)  # 优先淘汰最旧权重，显存及时归还
    else:
        _CACHE.move_to_end(key)
    return agent


def play_one_game(args):
    path, cfg, seed = args
    agent = _get_agent(path, cfg["c_puct"], cfg["mcts_batch"])
    rng = np.random.default_rng(seed)
    game = Gomoku(cfg["size"])
    feats, pols, turns = [], [], []
    winner = -1
    resigned = False
    while not game.is_terminal():
        ply = game.n_moves
        temp = 1.0 if ply < cfg["tau_moves"] else 0.0
        action, pi, stats = agent.act(
            game, temperature=temp, noise_eps=cfg["noise_eps"],
            noise_alpha=cfg["noise_alpha"], rng=rng, sim=cfg["sim_selfplay"])
        if ply >= cfg["resign_moves"] and stats["value"] < cfg["resign_value"]:
            winner = 1 - game.turn
            resigned = True
            break
        feats.append(game.encode())
        pols.append(pi)
        turns.append(game.turn)
        game = game.placed(action)
    if winner < 0:
        winner = game.winner
    zs = np.asarray(
        [0.0 if winner == 2 else (1.0 if t == winner else -1.0) for t in turns],
        np.float32)
    x = (np.stack(feats).astype(np.int8) if feats
         else np.zeros((0, 4, cfg["size"], cfg["size"]), np.int8))
    p = (np.stack(pols) if pols
         else np.zeros((0, cfg["size"] * cfg["size"]), np.float32))
    return x, p, zs, {"length": len(turns), "winner": int(winner), "resigned": resigned}


def eval_games(args):
    """进程内串行若干局评估对局。args: (path_a, path_b, cfg, game_indices, seed)。
    返回 (a胜, a负, 和, 总手数, 局数)。"""
    path_a, path_b, cfg, idxs, seed = args
    a = _get_agent(path_a, cfg["c_puct"], cfg["mcts_batch"])
    b = _get_agent(path_b, cfg["c_puct"], cfg["mcts_batch"])
    rng = np.random.default_rng(seed)
    wins = losses = draws = 0
    total_moves = 0
    for i in idxs:
        players = (a, b) if i % 2 == 0 else (b, a)
        game = Gomoku(cfg["size"])
        while not game.is_terminal():
            act, _, _ = players[game.turn].act(
                game, temperature=0.0, noise_eps=cfg["eval_noise"],
                noise_alpha=cfg["noise_alpha"], rng=rng, sim=cfg["sim_eval"])
            game = game.placed(act)
        total_moves += game.n_moves
        a_player = 0 if i % 2 == 0 else 1
        if game.winner == 2:
            draws += 1
        elif game.winner == a_player:
            wins += 1
        else:
            losses += 1
    return wins, losses, draws, total_moves, len(idxs)


def _avail_commit_mb() -> float:
    """Windows 剩余可提交内存（MB）；非 Windows 或查询失败视为无限。"""
    if os.name != "nt":
        return float("inf")
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_uint),
                    ("dwMemoryLoad", ctypes.c_uint),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)]

    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPageFile / 1048576 if ok else float("inf")


def resolve_workers(cfg):
    if cfg.workers > 0:
        return cfg.workers
    w = max(1, (os.cpu_count() or 4) - CPU_WORKER_RESERVE)
    if cfg.sp_device == "cuda":
        w = min(w, CUDA_WORKER_CAP)  # 限制 CUDA 上下文数量，保护显存
    return w


def resolve_eval_workers(cfg):
    """评估 CPU 进程数：显式指定最优先；自动值还会被当前剩余提交内存封顶，
    防止浏览器等大内存应用在场时 spawn 风暴直接 WinError 1455。"""
    if cfg.eval_workers > 0:
        return cfg.eval_workers
    w = max(2, (os.cpu_count() or 8) - EVAL_WORKER_RESERVE)
    fit = int((_avail_commit_mb() - EVAL_COMMIT_HEADROOM_MB)
              // CPU_WORKER_COMMIT_MB)
    return max(0, min(w, fit))


def make_pool(cfg):
    """创建可跨轮复用的自对弈进程池。"""
    workers = resolve_workers(cfg)
    if workers <= 1:
        return None
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(workers, initializer=_init_worker,
                    initargs=(cfg.sp_device,))
    pool.gobang_workers = workers
    return pool


def make_eval_pool(cfg):
    """评估专用 CPU 进程池：训练吃 GPU 时 CPU 正空闲，评估与其完全重叠。
    提交内存不足时返回 None，调用方回退共享池/串行。"""
    workers = resolve_eval_workers(cfg)
    if workers <= 1:
        return None
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(workers, initializer=_init_worker, initargs=("cpu",))
    pool.gobang_workers = workers
    return pool


def make_sp_jobs(cfg, model_path: str, ver: int):
    """ver 参与种子派生：不同轮次的对局种子互不重复。"""
    rng = np.random.default_rng(cfg.seed + 7919 * (ver + 1))
    return [(model_path, cfg.to_dict(), int(rng.integers(1 << 30)))
            for _ in range(cfg.games_per_iter)]


def make_ev_jobs(cfg, path_a: str, path_b: str, games: int, seed: int,
                 n_chunks: int):
    """评估对局按交错分片切成 n_chunks 个任务。"""
    nw = max(1, min(n_chunks, games))
    chunks = [list(range(games))[k::nw] for k in range(nw)]
    return [(path_a, path_b, cfg.to_dict(), ch, seed + k)
            for k, ch in enumerate(chunks) if ch]


def submit(pool, fn, jobs):
    """把任务全部投进池子立即返回，结果稍后用 collect_* 收。"""
    return [pool.apply_async(fn, (j,)) for j in jobs]


def _sp_concat(results):
    feats = np.concatenate([r[0] for r in results])
    pols = np.concatenate([r[1] for r in results])
    zs = np.concatenate([r[2] for r in results])
    stat = {"games": len(results),
            "avg_len": float(np.mean([r[3]["length"] for r in results])),
            "black_wins": sum(1 for r in results if r[3]["winner"] == 0),
            "white_wins": sum(1 for r in results if r[3]["winner"] == 1),
            "draws": sum(1 for r in results if r[3]["winner"] == 2),
            "resigns": sum(1 for r in results if r[3]["resigned"])}
    return feats, pols, zs, stat


def collect_sp(futs, desc="selfplay"):
    from tqdm import tqdm
    results = [f.get() for f in tqdm(futs, total=len(futs), desc=desc,
                                     unit="局", smoothing=0.05)]
    return _sp_concat(results)


def _ev_agg(outs):
    wins = sum(o[0] for o in outs)
    losses = sum(o[1] for o in outs)
    draws = sum(o[2] for o in outs)
    moves = sum(o[3] for o in outs)
    total = sum(o[4] for o in outs)
    score = (wins + 0.5 * draws) / max(1, total)
    return score, {"wins": wins, "losses": losses, "draws": draws,
                   "avg_len": moves / max(1, total)}


def collect_ev(futs, desc="eval"):
    from tqdm import tqdm
    outs = [f.get() for f in tqdm(futs, total=len(futs), desc=desc,
                                  unit="批", smoothing=0.05)]
    return _ev_agg(outs)


def run_selfplay(cfg, model_path: str, log=print, pool=None, ver: int = 0):
    jobs = make_sp_jobs(cfg, model_path, ver)
    if pool is not None and len(jobs) > 1:
        return collect_sp(submit(pool, play_one_game, jobs))
    _init_worker(cfg.sp_device)
    from tqdm import tqdm
    results = [play_one_game(j) for j in tqdm(jobs, total=len(jobs),
                                              desc="selfplay", unit="局",
                                              smoothing=0.05)]
    return _sp_concat(results)


def eval_parallel(cfg, path_a: str, path_b: str, games: int, seed: int,
                  pool=None, ver: int = 0):
    jobs = make_ev_jobs(cfg, path_a, path_b, games, seed,
                        resolve_workers(cfg))
    if pool is not None and len(jobs) > 1:
        return collect_ev(submit(pool, eval_games, jobs))
    _init_worker(cfg.sp_device)
    return _ev_agg([eval_games(j) for j in jobs])

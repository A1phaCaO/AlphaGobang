"""网络训练与棋盘评估（新模型 vs 当前最佳）。"""
from __future__ import annotations

import numpy as np
import torch

from .dataset import d4_transforms
from .game import Gomoku


def train_steps(net, buf, cfg, steps: int, lr: float, device: str, log=print,
                seed_mix: int = 0, progress: bool = False):
    """训练循环。GPU 可用时把整个数据窗口常驻显存、增强与采样全在 GPU 上做，
    消除每步 numpy/拷贝开销；显存不足自动回退 CPU 流水线。
    progress=True 时给步循环套一个 tqdm 进度条（默认关，main_train 不受影响）。
    cfg.use_amp 且设备为 GPU 时启用 FP16 混合精度（autocast + GradScaler），
    损失计算保持在 FP32，数值行为与纯 FP32 基本一致。
    seed_mix：跨调用打散采样/增强序列（同一参数文件连训多轮时避免每轮
    重复同一批索引与 D4 变换序列）。"""
    net.to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    dev = torch.device(device)
    use_cuda = dev.type == "cuda"
    amp = use_cuda and bool(getattr(cfg, "use_amp", False))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rng = np.random.default_rng(abs(cfg.seed) * 1000003 + seed_mix)
    tr = d4_transforms(net.size)
    n = int(buf["zs"].shape[0])
    B = cfg.batch_size

    gpu_data = None
    if use_cuda:
        try:
            F = torch.from_numpy(buf["feats"].astype(np.float32)).to(dev)
            P = torch.from_numpy(buf["pols"].astype(np.float32)).to(dev)
            Z = torch.from_numpy(buf["zs"]).to(dev)
            TR = [(torch.from_numpy(rr).to(dev), torch.from_numpy(cc).to(dev),
                   torch.from_numpy(pm).to(dev)) for rr, cc, pm in tr]
            gpu_data = (F, P, Z, TR)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            gpu_data = F = P = Z = TR = None
            torch.cuda.empty_cache()

    def batches():
        if gpu_data is not None:
            F, P, Z, TR = gpu_data
            for _ in range(steps):
                idx = torch.randint(n, (B,), device=dev)
                rr, cc, pm = TR[int(rng.integers(0, 8))]
                yield F[idx][:, :, rr, cc], P[idx][:, pm], Z[idx]
        else:
            feats, pols, zs = buf["feats"], buf["pols"], buf["zs"]
            for _ in range(steps):
                idx = rng.integers(0, n, B)
                f = feats[idx].astype(np.float32)
                p = pols[idx].astype(np.float32)
                rr, cc, pm = tr[int(rng.integers(0, 8))]
                f = f[:, :, rr, cc]
                p = p[:, pm]
                yield (torch.from_numpy(f).to(dev), torch.from_numpy(p).to(dev),
                       torch.from_numpy(zs[idx]).to(dev))

    tot = torch.zeros(2, device=dev)
    it = batches()
    if progress:
        from tqdm import tqdm
        it = tqdm(it, total=steps, desc="训练", unit="步")
    for f, p, z in it:
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(dev.type, enabled=amp):
            logits, value = net(f)
            # 非法点（已落子）屏蔽掉再算 softmax：推理侧 mcts._expand 是在合法
            # 点上重归一化的，等价于先屏蔽再 softmax。训练不屏蔽的话，网络得把
            # 概率质量挪到已占点上、目标里那些位置又恒为 0，梯度和推理语义不一致，
            # 策略头收敛明显变慢（交叉验证 CE 下不去）。
            occ = ((f[:, 0] + f[:, 1]) > 0.5).flatten(1)
            logits = logits.masked_fill(occ, -1e4)
            logp = torch.log_softmax(logits, dim=1)
            ce = -(p * logp).sum(1).mean()
            mse = ((value - z) ** 2).mean()
            loss = ce + mse
        if amp:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        tot += torch.stack([ce.detach().float(), mse.detach().float()])
    if use_cuda:
        gpu_data = None  # 生成器已耗尽，掐掉最后的引用让常驻显存尽早释放
    tot_ce, tot_v = float(tot[0]) / steps, float(tot[1]) / steps
    net.eval()
    return {"ce": tot_ce, "mse": tot_v, "samples": n}


def match(cfg, player_a, player_b, games: int, seed: int = 0, log=print,
          desc="eval"):
    """两个玩家交替先后手对弈，返回 (A 得分率, 统计)。玩家需实现 act()。"""
    from tqdm import tqdm
    rng = np.random.default_rng(seed)
    wins = losses = draws = 0
    lengths = []
    for i in tqdm(range(games), total=games, desc=desc, unit="局"):
        players = (player_a, player_b) if i % 2 == 0 else (player_b, player_a)
        game = Gomoku(cfg.size)
        while not game.is_terminal():
            act, _, _ = players[game.turn].act(
                game, temperature=0.0, noise_eps=cfg.eval_noise,
                noise_alpha=cfg.noise_alpha, rng=rng)
            game = game.placed(act)
        w = game.winner
        lengths.append(game.n_moves)
        a_player = 0 if i % 2 == 0 else 1
        if w == 2:
            draws += 1
        elif w == a_player:
            wins += 1
        else:
            losses += 1
        if log:
            log(f"eval {i + 1}/{games} score={_score(wins, draws, i + 1):.3f}")
    return _score(wins, draws, games), {"wins": wins, "losses": losses,
                                        "draws": draws,
                                        "avg_len": float(np.mean(lengths))}


def _score(wins: int, draws: int, games: int) -> float:
    return (wins + 0.5 * draws) / games

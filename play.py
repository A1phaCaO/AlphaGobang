"""对战脚本：人机 / 机器 vs 机器。默认打开图形窗口（tkinter）。

用法示例：
    uv run python play.py                          # 图形界面人机对战
    uv run python play.py --you white --sim 600    # 执白，AI 更深
    uv run python play.py -o runs/default/latest.pt            # 图形界面观战两个 AI
    uv run python play.py -o random --games 20                 # 控制台快速统计
    uv run python play.py --cli                                # 强制控制台字符棋盘
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gobang import ui
from gobang.agent import Agent, RandomPlayer
from gobang.config import (DEFAULT_MODEL, DEFAULT_PARAM_FILE, LIGHT_SIM_DIV,
                           LIGHT_SIM_MIN, Config, load_param_file)
from gobang.game import Gomoku
from gobang.model import load_ckpt
from gobang.trainer import match


def parse_args():
    p = argparse.ArgumentParser(description="五子棋人机/机器对战",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-m", "--model", default=DEFAULT_MODEL, help="主模型路径")
    p.add_argument("--config", default=DEFAULT_PARAM_FILE,
                   help="JSON 参数文件（c_puct/mcts_batch 等细节参数）")
    p.add_argument("-o", "--opponent", default=None,
                   help="对手：模型路径或 random；省略则为人机对战")
    p.add_argument("--you", choices=["black", "white"], default="black",
                   help="人机模式下人类执子")
    p.add_argument("--sim", type=int, default=None,
                   help="主模型 MCTS 模拟数，默认取参数文件的 sim_play")
    p.add_argument("--sim-b", type=int, default=None, help="对手 MCTS 模拟数，默认同 --sim")
    p.add_argument("--games", type=int, default=1, help="机器对战局数，>1 进入统计模式")
    p.add_argument("--self", dest="selfplay", action="store_true",
                   help="人人对弈录制：双方都由人输入，局末存入冷启动数据"
                        "（不加载任何模型）；配合 --cli 用控制台版")
    p.add_argument("--view", action="store_true", help="控制台逐手展示（配合 --cli）")
    p.add_argument("--cli", action="store_true", help="使用控制台字符界面而非图形窗口")
    p.add_argument("--delay", type=float, default=0.0, help="逐手展示时每步停顿秒数")
    p.add_argument("--device", default=None, help="推理设备，默认自动")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make_agent(path: str, device: str, sim: int, cfg: Config):
    net, _ = load_ckpt(path, device)
    agent = Agent(net=net, device=device, c_puct=cfg.c_puct,
                  batch=cfg.mcts_batch, sim=sim)
    return agent, net.size


def human_match(args, cfg, agent: Agent, size: int, buffer_dir: str | None = None):
    from gobang.dataset import Buffer, GameRecorder
    human = 1 if args.you == "white" else 0
    game = Gomoku(size)
    stack = [game]
    rng = np.random.default_rng(args.seed)
    buffer = Buffer(buffer_dir) if buffer_dir else None
    rec = GameRecorder()

    def save_record(winner):
        if buffer is None:
            return
        arr = rec.finish(winner)
        if arr is not None:
            buffer.append_human(*arr)
            print(f"本局 {arr[0].shape[0]} 手已存入冷启动数据 {buffer.root}/human")

    print(ui.LEGEND)
    print(ui.HELP)
    print()
    print(ui.render(game))
    result = None
    while result is None:
        if game.turn == human:
            while True:
                try:
                    line = input(f"\n轮到你（{ui.SYMBOLS[human]}）: ").strip()
                except EOFError:
                    print("退出")
                    return
                c = line.lower()
                if c in ("q", "quit", "exit"):
                    print("退出")
                    return
                if c == "undo":
                    i = len(stack) - 2
                    while i >= 0 and stack[i].turn != human:
                        i -= 1
                    if i >= 0:
                        del stack[i + 1:]
                        game = stack[-1]
                        rec.reset()
                        print(ui.render(game))
                    else:
                        print("没有可悔的棋")
                    break
                if c == "resign":
                    result = ("resign",)
                    save_record(1 - human)
                    break
                if c == "hint":
                    _, _, st = agent.act(game, temperature=0.0, rng=rng,
                                         sim=max(LIGHT_SIM_MIN,
                                                 args.sim // LIGHT_SIM_DIV))
                    top = int(st["policy"].argmax())
                    print(f"AI 推荐: {ui.pos_name(top, size)}  "
                          f"胜率评估 {st['value']:+.2f}")
                    continue
                pos = ui.parse_cell(c, size)
                if pos is None:
                    print("格式不对，参考 E5 / 5e / 5,5")
                    continue
                if not game.is_legal(pos):
                    print("该点已有子或不合法")
                    continue
                rec.add(game, pos)
                game = game.placed(pos)
                stack.append(game)
                print()
                print(ui.render(game))
                break
        else:
            t0 = time.time()
            a, _, st = agent.act(game, temperature=0.0, rng=rng, sim=args.sim)
            dt = time.time() - t0
            rec.add(game, a, st["policy"])
            game = game.placed(a)
            stack.append(game)
            print(f"\nAI（{ui.SYMBOLS[1 - human]}）落子 {ui.pos_name(a, size)}  "
                  f"用时 {dt:.1f}s  评估 {st['value']:+.2f}  访问 {st['visits']}")
            print(ui.render(game))
        if result is None and game.is_terminal():
            result = ("over", game.winner, game.n_moves)
            save_record(game.winner)
    if result[0] == "resign":
        print(f"你认输，{'黑' if human == 1 else '白'}棋胜")
    elif result[1] == 2:
        print("棋盘走满，和棋")
    else:
        side = "黑" if result[1] == 0 else "白"
        who = "AI 获胜" if result[1] != human else "你获胜"
        print(f"{result[2]} 手后{side}棋胜，{who}")


def self_match(args, size: int, buffer_dir: str | None = None):
    """人人对弈（控制台）：双方依次输入着法，局末整局存入冷启动数据。"""
    from gobang.dataset import Buffer, GameRecorder
    game = Gomoku(size)
    stack = [game]
    rec = GameRecorder()
    buffer = Buffer(buffer_dir) if buffer_dir else None
    print(ui.LEGEND)
    print(ui.HELP)
    print()
    print(ui.render(game))
    result = None

    def save(winner):
        if buffer is None:
            return
        arr = rec.finish(winner)
        if arr is not None:
            buffer.append_human(*arr)
            print(f"本局 {arr[0].shape[0]} 手已存入冷启动数据 {buffer.root}/human")

    while result is None:
        side = "黑" if game.turn == 0 else "白"
        while True:
            try:
                line = input(f"\n{side}方走棋: ").strip()
            except EOFError:
                print("退出")
                return
            c = line.lower()
            if c in ("q", "quit", "exit"):
                print("退出")
                return
            if c == "undo":
                if len(stack) > 1:
                    del stack[-1:]
                    game = stack[-1]
                    rec.reset()
                    print("（悔棋后本局不再作为教学样本）")
                    print(ui.render(game))
                else:
                    print("没有可悔的棋")
                break
            if c == "resign":
                result = ("resign", 1 - game.turn)
                save(1 - game.turn)
                break
            pos = ui.parse_cell(c, size)
            if pos is None:
                print("格式不对，参考 E5 / 5e / 5,5")
                continue
            if not game.is_legal(pos):
                print("该点已有子或不合法")
                continue
            rec.add(game, pos)
            game = game.placed(pos)
            stack.append(game)
            print()
            print(ui.render(game))
            break
        if result is None and game.is_terminal():
            result = ("over", game.winner)
            save(game.winner)
    if result[1] == 2:
        print("棋盘走满，和棋")
    else:
        print(f"{'黑' if result[1] == 0 else '白'}棋胜")


def view_match(args, cfg, pa, pb, size: int):
    game = Gomoku(size)
    rng = np.random.default_rng(args.seed)
    players = (pa, pb)
    print(ui.LEGEND)
    print()
    print(ui.render(game))
    while not game.is_terminal():
        t = game.turn
        a, _, st = players[t].act(game, temperature=0.0, noise_eps=cfg.eval_noise,
                                  noise_alpha=cfg.noise_alpha, rng=rng)
        game = game.placed(a)
        name = "黑" if t == 0 else "白"
        print(f"\n第 {game.n_moves} 手 {name} {ui.pos_name(a, size)}  "
              f"评估 {st['value']:+.2f}  访问 {st['visits']}")
        print(ui.render(game))
        if args.delay > 0:
            time.sleep(args.delay)
    if game.winner == 2:
        print("棋盘走满，和棋")
    else:
        print(f"{game.n_moves} 手后{'黑' if game.winner == 0 else '白'}棋胜")


def main():
    args = parse_args()
    cfg = Config()
    for k, v in load_param_file(args.config).items():
        setattr(cfg, k, v)
    if args.sim is None:
        args.sim = cfg.sim_play
    if args.selfplay:
        buffer_dir = str(Path(args.model).parent / "buffer")
        if args.cli:
            self_match(args, cfg.size, buffer_dir)
        else:
            from gobang import gui
            gui.run(cfg, "cpu", args.model, mode="hh", buffer_dir=buffer_dir)
        return
    if not Path(args.model).exists():
        print(f"找不到模型 {args.model}，请先运行: python main_train.py")
        sys.exit(1)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.set_num_threads(2)
    main_agent, size = make_agent(args.model, device, args.sim, cfg)
    cfg.size = size
    opp = None
    if args.opponent is not None:
        if args.opponent == "random":
            opp = RandomPlayer(seed=args.seed)
        else:
            if not Path(args.opponent).exists():
                print(f"找不到对手模型 {args.opponent}")
                sys.exit(1)
            opp, size_b = make_agent(args.opponent, device,
                                     args.sim_b or args.sim, cfg)
            if size_b != size:
                print(f"两个模型棋盘尺寸不一致: {size} vs {size_b}")
                sys.exit(1)
    games = max(1, args.games)
    stats_mode = opp is not None and games > 1 and not args.view
    if not args.cli and not stats_mode:
        try:
            from gobang import gui
        except ImportError:
            print("tkinter 不可用，退回控制台模式")
        else:
            mode = ("pvp" if args.opponent is None
                    else "view_random" if args.opponent == "random"
                    else "view_model")
            gui.run(cfg, device, args.model, mode=mode,
                    human=1 if args.you == "white" else 0, sim=args.sim,
                    opp_path=None if args.opponent in (None, "random")
                    else args.opponent,
                    buffer_dir=str(Path(args.model).parent / "buffer"))
            return
    if opp is None:
        human_match(args, cfg, main_agent, size,
                    buffer_dir=str(Path(args.model).parent / "buffer"))
    elif games == 1:
        view_match(args, cfg, main_agent, opp, size)
    else:
        print(f"{games} 局统计模式（交替先后手）...")
        wr, st = match(cfg, main_agent, opp, games, seed=args.seed,
                       log=lambda m: None if games <= 10 else print(m))
        print(f"\n主模型得分 {wr:.3f}   胜 {st['wins']} 负 {st['losses']} "
              f"和 {st['draws']}   平均 {st['avg_len']:.0f} 手")


if __name__ == "__main__":
    main()

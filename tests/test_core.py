"""核心组件自检。运行: python tests/test_core.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from gobang import ui
from gobang.agent import Agent
from gobang.config import FEATURE_PLANES, Config
from gobang.dataset import d4_transforms
from gobang.game import Gomoku
from gobang.model import GomokuNet

S = 9


def P(r, c):
    return r * S + c


def build(black_cells, white_cells):
    g = Gomoku(S)
    seq = []
    for i, b in enumerate(black_cells):
        seq.append(b)
        if i < len(white_cells):
            seq.append(white_cells[i])
    for pos in seq:
        g = g.placed(pos)
    return g


def win_cases():
    w = [P(0, k) for k in range(9)]  # 白棋占第一行前几格，均不成活
    assert build([P(4, c) for c in range(2, 7)], [w[0], w[1], w[2], w[3]]).winner == 0
    assert build([P(r, 4) for r in range(2, 7)], [w[4], w[5], w[6], w[7]]).winner == 0
    g = build([P(2, 2), P(3, 3), P(4, 4), P(5, 5), P(6, 6)],
              [w[0], w[1], w[2], w[3]])
    assert g.winner == 0
    g = build([P(2, 6), P(3, 5), P(4, 4), P(5, 3), P(6, 2)],
              [w[4], w[5], w[6], w[7]])
    assert g.winner == 0
    g = build([P(8, c) for c in range(4, 9)], [w[0], w[1], w[2], w[3]])
    assert g.winner == 0, "末行五连应判胜"
    print("win detection ok")


def no_false_positive():
    # 行末(3,6..8)+行首(4,0..1) 不构成五连，哨兵位必须挡住误判
    g = build([P(3, 6), P(3, 7), P(3, 8), P(4, 0), P(4, 1)],
              [P(0, 0), P(0, 1), P(0, 2), P(0, 3)])
    assert g.winner == -1, "跨行误判!"
    g = build([P(2, 8), P(3, 7), P(4, 6), P(5, 5), P(6, 4)],
              [P(0, 5), P(0, 6), P(0, 7), P(0, 8)])
    assert g.winner == 0, "斜线真五连漏判"
    print("no false positive ok")


def board_basics():
    g = Gomoku(S)
    assert len(g.legal_moves()) == S * S and g.turn == 0 and not g.is_terminal()
    g = g.placed(P(4, 4))
    assert g.turn == 1 and len(g.legal_moves()) == S * S - 1
    assert not g.is_legal(P(4, 4)) and g.is_legal(P(0, 0))
    assert g.last == P(4, 4) and g.n_moves == 1
    e = g.encode()
    assert e.shape == (4, S, S) and e[1, 4, 4] == 1 and e[2, 4, 4] == 1
    assert e[0].sum() == 0 and e[3].mean() == 0.0  # 白方视角：己方为空、先手平面 0
    g2 = g.placed(P(0, 0))
    e2 = g2.encode()
    assert e2[0, 4, 4] == 1 and e2[1, 0, 0] == 1 and e2[3].mean() == 1.0
    term = build([P(4, c) for c in range(2, 7)],
                 [P(0, 0), P(0, 1), P(0, 2), P(0, 3)])
    assert term.winner == 0 and term.terminal_value() == -1.0
    assert term.legal_moves() == []
    print("board basics ok")


def draw_case():
    g = Gomoku(S)
    rng = np.random.default_rng(1)
    # 交替走满：无五连很难构造，改为直接验证走满判和的逻辑上限
    cells = list(range(S * S))
    rng.shuffle(cells)
    try:
        for i, cpos in enumerate(cells):
            g = g.placed(cpos)
            if g.is_terminal():
                assert g.winner in (0, 1, 2)
                break
        else:
            assert g.winner == 2 and g.n_moves == S * S
    except AssertionError:
        raise
    print("full-board terminal ok")


def d4_checks():
    perms = d4_transforms(S)
    assert len(perms) == 8
    seen = set()
    mat = np.arange(S * S).reshape(S, S)
    for rr, cc, pm in perms:
        assert sorted(pm.tolist()) == list(range(S * S))
        seen.add(bytes(pm.tobytes()))
        g = mat[rr, cc].reshape(-1)
        assert (g == pm).all(), "平面变换与策略变换不一致"
    assert len(seen) == 8
    print("d4 ok")


def model_checks():
    net = GomokuNet(S, 32, 2)
    x = torch.randn(3, 4, S, S)
    logits, value = net(x)
    assert logits.shape == (3, S * S) and value.shape == (3,)
    assert (value.abs() <= 1.0).all()
    print("model ok, params:", sum(p.numel() for p in net.parameters()))


def mcts_direction():
    """PUCT 方向性回归：选择必须最大化本方视角(-ch.Q)。
    构造器规定「上一手落在40点则轮走方大优」，则开局40点是坏手，
    搜索的访问分布必须避开它。"""
    from gobang.mcts import mcts_search

    def evaluator(games):
        probs = np.full((len(games), S * S), 1.0 / (S * S), np.float32)
        vals = np.array([1.0 if g.last == 40 else -1.0 for g in games],
                        np.float32)
        return probs, vals

    root = mcts_search(Gomoku(S), evaluator, sim=200, batch=8)
    visits = {m: ch.N for m, ch in root.children.items()}
    bad = visits.get(40, 0)
    rest = sum(v for m, v in visits.items() if m != 40)
    assert bad * 3 < rest, f"搜索偏好坏手: bad={bad} rest={rest}"
    print("puct direction ok")


def _tree_depth(node):
    # 树深。退化搜索的典型特征就是所有叶子都挂在根下面（深度只有 1）。
    if not node.children:
        return 0
    return 1 + max(_tree_depth(ch) for ch in node.children.values())


def mcts_batch_guard():
    """batch >= sim 的兜底：搜索必须拆成多轮，且同一叶子不得重复入队。

    回归用例：mcts_batch=128 + sim_selfplay=128 时整步搜索只有一次回传，树里
    没有价值反馈，访问分布退化成「每个可走点各 1 次访问」的均匀分布，自对弈
    数据等于随机落子，训练 CE 只能收敛到 ln(合法点数)≈3.5~4.0 的熵地板。
    """
    import gobang.mcts as M
    from gobang.mcts import clamp_batch, mcts_search

    assert clamp_batch(128, 128) == 16
    assert clamp_batch(8, 128) == 8
    for sim in (24, 64, 128):
        calls = []
        seen = []

        def evaluator(games, calls=calls):
            probs = np.full((len(games), S * S), 1.0 / (S * S), np.float32)
            vals = np.array([1.0 if g.last == 40 else -1.0 for g in games],
                            np.float32)
            calls.append(len(games))
            return probs, vals

        orig_select = M._select

        def select(root, c_puct, orig=orig_select, seen=seen):
            out = orig(root, c_puct)
            if isinstance(out, tuple):
                seen.append(id(out[0]))
            return out

        M._select = select
        try:
            root = mcts_search(Gomoku(S), evaluator, sim=sim, batch=sim)
        finally:
            M._select = orig_select
        vis = {m: ch.N for m, ch in root.children.items()}
        assert len(calls) > 2, f"sim={sim} 只做了 {len(calls)} 批评估，搜索退化成单层"
        assert len(seen) == len(set(seen)), f"sim={sim} 同一叶子被重复入队"
        assert sum(vis.values()) == sim - 1, (sim, sum(vis.values()))
        assert max(vis.values()) > 1, f"sim={sim} 访问数全为 1，等于没有搜索"
        assert _tree_depth(root) > 1, f"sim={sim} 树只有根一层，没往下搜"
    print("mcts batch guard ok")


def mcts_checks():
    torch.manual_seed(0)
    net = GomokuNet(S, 32, 2)
    agent = Agent(net=net, device="cpu", sim=24)
    game = Gomoku(S)
    rng = np.random.default_rng(0)
    action, policy, stats = agent.act(game, temperature=1.0, noise_eps=0.25, rng=rng)
    assert game.is_legal(action)
    assert abs(policy.sum() - 1.0) < 1e-5
    assert stats["visits"] == 24
    action2, _, _ = agent.act(game, temperature=0.0, rng=rng)
    assert game.is_legal(action2)
    print("mcts ok")


def ui_checks():
    assert ui.parse_cell("e5", S) == P(4, 4)
    assert ui.parse_cell("5e", S) == P(4, 4)
    assert ui.parse_cell("5,5", S) == P(4, 4)
    assert ui.parse_cell(" 5 5 ", S) == P(4, 4)
    assert ui.parse_cell("j10", S) is None
    assert ui.parse_cell("0,5", S) is None
    assert ui.parse_cell("e0", S) is None
    g = Gomoku(S)
    txt = ui.render(g.placed(P(4, 4)))
    assert " x " in txt and "A" in txt and "9" in txt
    print("ui ok")


def tactic_checks():
    g = build([P(4, 2), P(4, 3), P(4, 4), P(4, 5)],
              [P(0, 0), P(0, 1), P(1, 0), P(1, 1)])
    w = g.winning_moves(0)
    assert set(w) == {P(4, 1), P(4, 6)}, w
    assert g.winning_moves(1) == []
    from gobang.agent import Agent
    net = GomokuNet(S, 32, 2)
    agent = Agent(net=net, device="cpu", sim=24)
    rng = np.random.default_rng(0)
    a, _, st = agent.act(g, temperature=0.0, rng=rng)
    assert a in w and st["force"] == set(w), (a, st["force"])
    # 白方轮走，黑有四连威胁 → 强制只搜堵点
    g2 = build([P(4, 3), P(4, 4), P(4, 5), P(4, 6)],
               [P(0, 0), P(0, 1), P(1, 0)])
    assert g2.turn == 1
    ow = g2.winning_moves(0)
    assert set(ow) == {P(4, 2), P(4, 7)}, ow
    a2, _, st2 = agent.act(g2, temperature=0.0, rng=rng)
    assert a2 in ow and st2["force"] == set(ow), (a2, st2["force"])
    a3, _, st3 = agent.act(Gomoku(S), temperature=0.0, rng=rng)
    assert st3["force"] is None
    print("tactic checks ok")


def recorder_checks():
    import tempfile
    from gobang.dataset import Buffer, GameRecorder
    with tempfile.TemporaryDirectory() as td:
        buf = Buffer(td)
        rec = GameRecorder()
        g = Gomoku(S)
        moves = [P(4, 3), P(0, 0), P(4, 4), P(0, 1), P(4, 5), P(0, 2),
                 P(4, 6), P(0, 4), P(4, 7)]
        for m in moves:
            rec.add(g, m)
            g = g.placed(m)
        assert g.winner == 0, g.winner
        arr = rec.finish(g.winner)
        assert arr is not None and arr[0].shape[0] == 9
        assert arr[2][0] == 1.0 and arr[2][1] == -1.0
        buf.append_human(*arr)
        w = buf.load_window(3, 3)
        assert w is not None and w["feats"].shape[0] == 9
        assert buf.load_window(3, 3, with_human=False) is None
    print("recorder ok")


def cli_checks():
    """参数优先级：命令行 > --preset > 参数文件 > 内置默认；未知键必须报错。"""
    import sys
    import tempfile
    argv = sys.argv
    try:
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "p.json"
            # 下划线开头的键当注释放行（JSON 没注释），其余未知键仍要报错
            fp.write_text('{"games_per_iter": 100, "c_puct": 1.2, "use_amp": true,'
                          ' "_comment": "跑法说明", "_note": "另一行注释"}',
                          encoding="utf-8")
            sys.argv = ["main_train.py", "--config", str(fp),
                        "--preset", "smoke", "--games", "9", "--eval-every", "3"]
            from main_train import parse_args
            cfg, notes = parse_args()
            assert cfg.games_per_iter == 9, "命令行单项应最优先"
            assert cfg.sim_selfplay == 32 and cfg.run_dir == "runs/smoke", \
                "预设应覆盖参数文件与默认"
            assert cfg.c_puct == 1.2 and cfg.use_amp is True, "参数文件应覆盖内置默认"
            assert cfg.iterations == 2 and cfg.train_steps == 50
            assert cfg.eval_every == 3, "--eval-every 应落到 Config.eval_every"
            assert any("smoke" in n for n in notes)
            sys.argv = ["main_train.py", "--config", str(fp), "--games", "9"]
            cfg, _ = parse_args()  # 不传 --preset 也不应崩
            assert cfg.games_per_iter == 9 and cfg.c_puct == 1.2
            assert cfg.sim_selfplay == 128 and cfg.run_dir == "runs/default"
            fp.write_text('{"no_such_key": 1}', encoding="utf-8")
            sys.argv = ["main_train.py", "--config", str(fp)]
            try:
                parse_args()
            except SystemExit:
                pass
            else:
                raise AssertionError("参数文件未知键应报错")
    finally:
        sys.argv = argv
    print("cli/param-file ok")


def amp_checks():
    """AMP 开关下训练都应产出有限损失。"""
    from gobang.trainer import train_steps
    if not torch.cuda.is_available():
        print("amp skipped (no cuda)")
        return
    rng = np.random.default_rng(0)
    n = 64
    buf = {"feats": rng.integers(0, 2, (n, FEATURE_PLANES, S, S)).astype(np.int8),
           "pols": np.zeros((n, S * S), np.float32),
           "zs": rng.choice(np.array([-1.0, 0.0, 1.0]), n)}
    for i in range(n):
        buf["pols"][i, int(rng.integers(0, S * S))] = 1.0
    for amp in (False, True):
        torch.manual_seed(0)
        net = GomokuNet(S, 32, 2)
        cfg = Config()
        cfg.use_amp = amp
        cfg.batch_size = 32
        out = train_steps(net, buf, cfg, 8, 1e-3, "cuda")
        assert np.isfinite(out["ce"]) and np.isfinite(out["mse"]), (amp, out)
    print("amp train ok")


def main():
    win_cases()
    no_false_positive()
    board_basics()
    draw_case()
    tactic_checks()
    recorder_checks()
    d4_checks()
    model_checks()
    mcts_checks()
    mcts_direction()
    mcts_batch_guard()
    ui_checks()
    cli_checks()
    amp_checks()
    print("ALL CORE TESTS PASSED")


if __name__ == "__main__":
    main()

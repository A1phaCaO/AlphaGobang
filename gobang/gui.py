"""tkinter 图形界面：点击落子，人机对战与机器观战共用。

所有对战参数（模式、模型、执子、深度、速度）均可在面板内调节；
AI 推理与胜率预判在后台线程运行，结果通过 after() 回投主线程。
"""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .config import LIGHT_SIM_DIV, LIGHT_SIM_MIN, STATS_GAMES
from .game import Gomoku
from .model import load_ckpt

MARGIN = 38
BG_COLOR = "#d9a066"
LINE_COLOR = "#4a2f14"
LABELS = "ABCDEFGHIJKLMN"
SIM_MIN, SIM_MAX = 32, 30000

MODES = [("人机对战", "pvp"), ("观战 vs 随机", "view_random"),
         ("观战 vs 模型", "view_model")]


def _stars(size: int) -> list[tuple[int, int]]:
    if size < 7:
        return []
    pad = 2
    pts = [(pad, pad), (pad, size - 1 - pad), (size - 1 - pad, pad),
           (size - 1 - pad, size - 1 - pad)]
    if size % 2 == 1:
        pts.append(((size - 1) // 2, (size - 1) // 2))
    return pts


class App(tk.Tk):
    def __init__(self, cfg, device: str, model_path: str, mode: str = "pvp",
                 human: int = 0, sim: int = 400, opp_path: str | None = None,
                 buffer_dir: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.model_path = model_path
        self.opp_path = opp_path
        self.buffer = None
        if buffer_dir:
            from .dataset import Buffer, GameRecorder
            self.buffer = Buffer(buffer_dir)
            self.recorder = GameRecorder()
        self._recorded = True
        self.agent = None
        self.opp = None
        self.busy = False
        self.dead = False
        self.seq = 0
        self.auto = False
        self.auto_ms = 400
        self.rng = np.random.default_rng()
        self.title("五子棋")
        self.resizable(False, False)
        self._build()
        self.game = Gomoku(self.size)
        self.stack = [self.game]
        self.mode_var.set(mode)
        self.color_var.set("黑" if human == 0 else "白")
        if mode == "view_model" and opp_path:
            self.opp_name_var.set(opp_path)
        self._apply_mode()
        self._set_sim_slider(sim)
        self._update_sim_label()
        self._reload_players()
        self.new_game()

    # ---------- 构建 ----------
    def _build(self):
        self.size = self.cfg.size
        self._fit_canvas()
        self.canvas = tk.Canvas(self, width=self.board_px, height=self.board_px,
                                bg=BG_COLOR, highlightthickness=0)
        self.canvas.pack(side="left")
        panel = ttk.Frame(self, padding=8)
        panel.pack(side="left", fill="y")
        panel.columnconfigure(1, weight=1)

        ttk.Label(panel, text="模式").grid(row=0, column=0, sticky="w")
        self.mode_var = tk.StringVar(value="pvp")
        frame_m = ttk.Frame(panel)
        frame_m.grid(row=0, column=1, sticky="w")
        for i, (text, val) in enumerate(MODES):
            ttk.Radiobutton(frame_m, text=text, value=val, variable=self.mode_var,
                            command=self._on_mode).grid(row=i, column=0, sticky="w")

        self.opp_name_var = tk.StringVar(value="runs/default/latest.pt")
        self.opp_row = ttk.Frame(panel)
        self.opp_row.grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Button(self.opp_row, text="对手模型…", width=10,
                   command=self._pick_opp).pack(side="left")
        ttk.Label(self.opp_row, textvariable=self.opp_name_var,
                  width=22).pack(side="left")

        row = ttk.Frame(panel)
        row.grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Button(row, text="加载模型…", width=10, command=self._pick_model)\
            .pack(side="left")
        self.model_name_var = tk.StringVar(value=self.model_path)
        ttk.Label(row, textvariable=self.model_name_var, width=22).pack(side="left")

        ttk.Label(panel, text="你执子").grid(row=3, column=0, sticky="w")
        self.color_var = tk.StringVar(value="黑")
        self.color_cb = ttk.Combobox(panel, textvariable=self.color_var,
                                     values=["黑", "白"], state="readonly", width=6)
        self.color_cb.grid(row=3, column=1, sticky="w")
        self.color_cb.bind("<<ComboboxSelected>>",
                           lambda e: self.new_game() if not self.view else None)

        ttk.Label(panel, text="搜索深度").grid(row=4, column=0, sticky="w")
        fr = ttk.Frame(panel)
        fr.grid(row=4, column=1, sticky="we")
        self.sim_scale = ttk.Scale(fr, from_=0, to=100,
                                   command=self._update_sim_label)
        self.sim_scale.pack(side="left", fill="x", expand=True)
        self.sim_var = tk.StringVar(value="400")
        ttk.Label(fr, textvariable=self.sim_var, width=6,
                  anchor="e").pack(side="right")

        ttk.Label(panel, text="胜率预判").grid(row=5, column=0, sticky="w")
        self.pred_frame = ttk.Frame(panel)
        self.pred_frame.grid(row=5, column=1, sticky="we")
        self.pred_bar = ttk.Progressbar(self.pred_frame, maximum=100, length=110)
        self.pred_bar.pack(side="left")
        self.pred_var = tk.StringVar(value="--")
        ttk.Label(self.pred_frame, textvariable=self.pred_var,
                  width=12).pack(side="left")

        self.rec_on = tk.BooleanVar(value=True)
        self.rec_chk = ttk.Checkbutton(panel, text="本局存入人类数据集",
                                       variable=self.rec_on)
        self.rec_chk.grid(row=6, column=0, columnspan=2, sticky="w")

        self.status = tk.Label(panel, text="", width=26, anchor="w",
                               font=("Microsoft YaHei UI", 10, "bold"))
        self.status.grid(row=7, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.last_info = ttk.Label(panel, text="")
        self.last_info.grid(row=8, column=0, columnspan=2, sticky="w")

        btns = ttk.Frame(panel)
        btns.grid(row=9, column=0, columnspan=2, sticky="w", pady=6)
        self.b_new = ttk.Button(btns, text="新局", width=7, command=self.new_game)
        self.b_undo = ttk.Button(btns, text="悔棋", width=7, command=self.undo)
        self.b_hint = ttk.Button(btns, text="提示", width=7, command=self.hint)
        self.b_resign = ttk.Button(btns, text="认输", width=7, command=self.resign)
        self.play_text = tk.StringVar(value="▶ 自动")
        self.b_play = ttk.Button(btns, textvariable=self.play_text, width=7,
                                 command=self.toggle_auto)
        self.b_step = ttk.Button(btns, text="单步", width=7, command=self.step_once)
        self.b_stat = ttk.Button(btns, text=f"{STATS_GAMES}局统计", width=7,
                                 command=self.stats)
        self.view_only = (self.b_play, self.b_step)
        self.pvp_only = (self.b_undo, self.b_hint, self.b_resign)
        for i, b in enumerate((self.b_new, self.b_stat, self.b_undo,
                               self.b_hint, self.b_resign, self.b_play, self.b_step)):
            b.grid(row=i // 2, column=i % 2, padx=2, pady=2, sticky="we")

        self.delay_row = ttk.Frame(panel)
        ttk.Label(self.delay_row, text="出手间隔").pack(side="left")
        sc = ttk.Scale(self.delay_row, from_=80, to=2000, orient="horizontal",
                       command=lambda v: setattr(self, "auto_ms", int(float(v))))
        sc.set(self.auto_ms)
        sc.pack(side="left", fill="x")
        self.delay_row.grid(row=10, column=0, columnspan=2, sticky="we")

        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda e: self._draw_hover(None))
        self.canvas.bind("<Button-1>", self._on_click)
        self.protocol("WM_DELETE_WINDOW", self._quit)

    def _fit_canvas(self):
        self.cell = 46 if self.size <= 11 else 34
        self.board_px = 2 * MARGIN + (self.size - 1) * self.cell + 16

    def _on_mode(self):
        self._apply_mode()
        self._reload_players()
        self.new_game()

    def _apply_mode(self):
        m = self.mode_var.get()
        self.view = m != "pvp"
        self.human = 1 if self.color_var.get() == "白" else 0
        for b in self.view_only:
            b.grid() if self.view else b.grid_remove()
        for b in self.pvp_only:
            b.grid() if not self.view else b.grid_remove()
        self.delay_row.grid() if self.view else self.delay_row.grid_remove()
        if self.buffer is not None:
            self.rec_chk.grid() if not self.view else self.rec_chk.grid_remove()
        self.color_cb.config(state="readonly" if not self.view else "disabled")
        if self.view:
            self.auto = False
            self.play_text.set("▶ 自动")

    def _default_dir(self):
        from pathlib import Path
        d = str(Path(self.model_path).parent)
        return d if Path(d).is_dir() else "."

    def _pick_model(self):
        p = filedialog.askopenfilename(title="选择模型",
                                       filetypes=[("模型", "*.pt"), ("全部", "*.*")],
                                       initialdir=self._default_dir())
        if p:
            self.model_path = p
            self.model_name_var.set(p)
            self._reload_players()
            self.new_game()

    def _pick_opp(self):
        p = filedialog.askopenfilename(title="选择对手模型",
                                       filetypes=[("模型", "*.pt"), ("全部", "*.*")],
                                       initialdir=self._default_dir())
        if p:
            self.opp_name_var.set(p)
            self._reload_players()
            self.new_game()

    def _reload_players(self):
        from .agent import RandomPlayer
        self.agent = None
        try:
            net, _ = load_ckpt(self.model_path, self.device)
            if net.size != self.size:
                self._resize_board(net.size)
            from .agent import Agent
            self.agent = Agent(net=net, device=self.device,
                               c_puct=self.cfg.c_puct, batch=self.cfg.mcts_batch,
                               sim=self.sim)
        except Exception as e:
            self._set_status(f"模型加载失败: {e}")
            return
        m = self.mode_var.get()
        if m == "pvp":
            self.opp = None
        elif m == "view_random":
            self.opp = RandomPlayer()
        else:
            try:
                net2, _ = load_ckpt(self.opp_name_var.get().strip(), self.device)
                if net2.size != self.size:
                    self._resize_board(net2.size)
                from .agent import Agent
                self.opp = Agent(net=net2, device=self.device,
                                 c_puct=self.cfg.c_puct, batch=self.cfg.mcts_batch,
                                 sim=self.sim)
            except Exception as e:
                self._set_status(f"对手模型加载失败: {e}")
                self.opp = None
        self.title(f"{'人机' if m == 'pvp' else '观战'}  {self.size}x{self.size}")

    def _resize_board(self, size):
        self.size = size
        self.cfg.size = size
        self._fit_canvas()
        self.canvas.config(width=self.board_px, height=self.board_px)

    def _update_sim_label(self, _v=None):
        frac = float(self.sim_scale.get()) / 100.0
        self.sim = max(16, int(round(SIM_MIN * (SIM_MAX / SIM_MIN) ** frac)))
        self.sim_var.set(str(self.sim))

    def _set_sim_slider(self, sim):
        import math
        sim = min(SIM_MAX, max(SIM_MIN, int(sim)))
        self.sim_scale.set(
            100.0 * math.log(sim / SIM_MIN) / math.log(SIM_MAX / SIM_MIN))

    # ---------- 绘制 ----------
    def _xy(self, i):
        return MARGIN + i * self.cell

    def _draw_base(self):
        c = self.canvas
        c.delete("all")
        s = self.size
        lo, hi = self._xy(0), self._xy(s - 1)
        for i in range(s):
            p = self._xy(i)
            c.create_line(lo, p, hi, p, fill=LINE_COLOR)
            c.create_line(p, lo, p, hi, fill=LINE_COLOR)
        for (r, col) in _stars(s):
            x, y = self._xy(col), self._xy(r)
            c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=LINE_COLOR, outline="")
        for i in range(s):
            c.create_text(self._xy(i), 14, text=LABELS[i], font=("Segoe UI", 10))
            c.create_text(14, self._xy(i), text=str(i + 1), font=("Segoe UI", 10))

    def _draw_stones(self):
        c = self.canvas
        c.delete("stone")
        c.delete("hint")
        s, r = self.size, self.cell * 0.42
        for pid, color, edge in ((0, "#1b1b1b", "gray40"), (1, "#f5f5f5", "gray45")):
            mask = self.game.stones[pid]
            while mask:
                lsb = mask & -mask
                mask ^= lsb
                pos = self.game.m["b2p"][lsb.bit_length() - 1]
                x, y = self._xy(pos % s), self._xy(pos // s)
                c.create_oval(x - r, y - r, x + r, y + r, fill=color,
                              outline=edge, tags="stone")
        if self.game.last >= 0:
            x, y = self._xy(self.game.last % s), self._xy(self.game.last // s)
            c.create_oval(x - 4, y - 4, x + 4, y + 4, fill="red", outline="",
                          tags="stone")

    def _draw_hover(self, pos):
        self.canvas.delete("hover")
        if pos is None or self.game.is_terminal() or not self.game.is_legal(pos):
            return
        s = self.size
        x, y = self._xy(pos % s), self._xy(pos // s)
        r = self.cell * 0.42
        self.canvas.create_oval(x - r, y - r, x + r, y + r, outline="red",
                                dash=(3, 2), tags="hover")

    def _refresh(self):
        self._draw_stones()
        self._draw_hover(None)
        g = self.game
        if g.is_terminal():
            if g.winner == 2:
                text = "和棋（棋盘走满）"
            else:
                side = "黑" if g.winner == 0 else "白"
                if self.view:
                    text = f"{g.n_moves} 手后{side}棋胜"
                else:
                    text = "你赢了！" if g.winner == self.human else "AI 获胜"
            self._set_status(text)
            self.pred_var.set("终局")
            self._save_record(g.winner)
            if self.view:
                self.auto = False
                self.play_text.set("▶ 自动")
        elif self.view:
            self._set_status(f"第 {g.n_moves + 1} 手"
                             f"（{'黑' if g.turn == 0 else '白'}方行棋）")
        elif g.turn == self.human:
            self._set_status(f"轮到你（{'黑' if self.human == 0 else '白'}）点击落子")
        else:
            self._set_status("AI 思考中…")

    def _set_status(self, text):
        if not self.dead:
            self.status.config(text=text)

    def pos_name(self, pos):
        return f"{LABELS[pos % self.size]}{pos // self.size + 1}"

    # ---------- 输入 ----------
    def _pick(self, event):
        s = self.size
        col = int(round((event.x - MARGIN) / self.cell))
        row = int(round((event.y - MARGIN) / self.cell))
        if not (0 <= row < s and 0 <= col < s):
            return None
        if max(abs(event.x - self._xy(col)), abs(event.y - self._xy(row))) \
                > self.cell * 0.45:
            return None
        return row * s + col

    def _on_motion(self, event):
        if self.view or self.busy:
            return
        pos = self._pick(event)
        self._draw_hover(pos if self.game.turn == self.human else None)

    def _on_click(self, event):
        if self.view or self.busy or self.game.is_terminal():
            return
        pos = self._pick(event)
        if pos is None or self.game.turn != self.human or not self.game.is_legal(pos):
            return
        self._move(pos)
        if not self.game.is_terminal():
            self._predict_async()
            self._ai_step()

    def _move(self, pos, policy=None):
        if (self.buffer is not None and not self.view
                and self.rec_on.get()):
            self.recorder.add(self.game, pos, policy)
        self.game = self.game.placed(pos)
        self.stack.append(self.game)
        self._refresh()

    def _save_record(self, winner):
        if (self.buffer is None or self._recorded or self.view
                or not self.rec_on.get()):
            return
        self._recorded = True
        arr = self.recorder.finish(winner)
        if arr is not None:
            self.buffer.append_human(*arr)
            n = arr[0].shape[0]
            self.last_info.config(text=f"本局 {n} 手已存入冷启动数据 "
                                       f"({self.buffer.root}/human)")

    # ---------- 对局调度 ----------
    def new_game(self):
        if self.agent is None:
            return
        self.seq += 1
        self.game = Gomoku(self.size)
        self.stack = [self.game]
        if self.buffer is not None:
            self.recorder.reset()
            self._recorded = False
        self._draw_base()
        if not self.view:
            self.human = 0 if self.color_var.get() == "黑" else 1
        self.last_info.config(text="")
        self.pred_bar["value"] = 50
        self.pred_var.set("--")
        self._refresh()
        if self.view:
            self.auto = False
            self.play_text.set("▶ 自动")
        elif self.game.turn != self.human:
            self._ai_step()

    def _bg(self, fn, on_done):
        self.busy = True
        seq = self.seq
        self._refresh()

        def work():
            try:
                res = fn()
            except Exception as e:
                def fail():
                    if self.dead:
                        return
                    self.busy = False
                    if seq == self.seq:
                        self._set_status(f"错误: {e}")
                self._post(fail)
                return
            self._post(lambda: self._bg_done(seq, res, on_done))

        threading.Thread(target=work, daemon=True).start()

    def _post(self, fn):
        if self.dead:
            return
        try:
            self.after(0, fn)
        except tk.TclError:
            pass

    def _bg_done(self, seq, res, on_done):
        if self.dead:
            return
        self.busy = False
        if seq == self.seq:
            on_done(res)

    def _next_player(self):
        """当前行棋方对应的玩家对象。"""
        if self.view:
            if self.game.turn == 0:
                return self.agent
            return self.opp or self.agent
        return self.agent

    def _ai_step(self):
        g = self.game
        player = self._next_player()
        if player is None or g.is_terminal():
            return
        rng = self.rng
        sim = self.sim
        label = ("黑" if g.turn == 0 else "白") if self.view else "AI"

        def think():
            # 观战保留少量噪声让每局不同；人机不给 AI 注入随机
            return player.act(g, temperature=0.0,
                              noise_eps=self.cfg.eval_noise if self.view else 0.0,
                              noise_alpha=self.cfg.noise_alpha, rng=rng, sim=sim)

        def done(res):
            action, _, stats = res
            if self.game is not g:
                return
            v = stats["value"]
            self.last_info.config(
                text=f"{label} {self.pos_name(action)}  评估 "
                     f"{v:+.2f}  访问 {stats['visits']}")
            self._move(action, stats["policy"])
            if self.view:
                nxt = "黑" if self.game.turn == 0 else "白"
                self._show_pred(-v, f"{nxt}方胜率")
                if self.auto:
                    self._schedule()
            else:
                self._show_pred(v, "AI胜率")
                if not self.game.is_terminal() and self.game.turn != self.human:
                    self._ai_step()

        self._bg(think, done)

    # ---------- 胜率预判 ----------
    def _show_pred(self, v, label):
        """v ∈ [-1,1]，label 方的预期得分率。"""
        pct = (float(v) + 1.0) / 2.0 * 100.0
        self.pred_bar["value"] = pct
        self.pred_var.set(f"{label} {pct:.0f}%")

    def _predict_async(self):
        """人类刚走完一手：用当前 AI 快速推演一次给出胜率（不阻塞输入）。"""
        if self.view or self.agent is None or self.game.is_terminal():
            return
        g = self.game
        agent = self.agent
        rng = self.rng
        sim = max(LIGHT_SIM_MIN, self.sim // LIGHT_SIM_DIV)
        seq = self.seq

        def work():
            try:
                _, _, stats = agent.act(g, temperature=0.0, rng=rng, sim=sim)
            except Exception:
                return
            v = stats["value"]
            def show():
                if self.dead or seq != self.seq or self.game is not g:
                    return
                # stats["value"] 是 g 行棋方视角；常规时行棋方=AI，
                # 悔棋后回到人类行棋，需取负才是 AI 胜率
                self._show_pred(v if g.turn != self.human else -v, "AI胜率")
            self._post(show)

        threading.Thread(target=work, daemon=True).start()

    # ---------- 观战 ----------
    def toggle_auto(self):
        if self.game.is_terminal():
            return
        self.auto = not self.auto
        self.play_text.set("⏸ 暂停" if self.auto else "▶ 自动")
        if self.auto and not self.busy:
            self._ai_step()

    def _schedule(self):
        if self.auto and not self.dead and not self.game.is_terminal():
            self.after(self.auto_ms, self._auto_tick)

    def _auto_tick(self):
        if not self.auto or self.dead or self.game.is_terminal():
            return
        if self.busy:
            self._schedule()
        else:
            self._ai_step()

    def step_once(self):
        if self.busy or self.game.is_terminal():
            return
        self._ai_step()

    # ---------- 人机功能 ----------
    def undo(self):
        if self.busy or self.view or len(self.stack) < 2:
            return
        i = len(self.stack) - 2
        while i >= 0 and self.stack[i].turn != self.human:
            i -= 1
        if i < 0:
            return
        self.seq += 1
        del self.stack[i + 1:]
        self.game = self.stack[-1]
        if self.buffer is not None:
            self.recorder.reset()
            self._recorded = True  # 悔棋后本局不再作为教学样本
        self._refresh()
        self._predict_async()

    def hint(self):
        if self.busy or self.view or self.game.is_terminal():
            return
        g = self.game
        rng = self.rng
        sim = max(LIGHT_SIM_MIN, self.sim // LIGHT_SIM_DIV)

        def think():
            return self.agent.act(g, temperature=0.0, rng=rng, sim=sim)

        def done(res):
            action, _, stats = res
            x, y = self._xy(action % self.size), self._xy(action // self.size)
            r = self.cell * 0.55
            self.canvas.delete("hint")
            self.canvas.create_oval(x - r, y - r, x + r, y + r, outline="#1565c0",
                                    width=3, tags="hint")
            self._set_status(f"提示 {self.pos_name(action)} 评估 {stats['value']:+.2f}")

        self._bg(think, done)

    def resign(self):
        if self.busy or self.view or self.game.is_terminal():
            return
        self.seq += 1
        self.game = self.game.forfeit()
        self._refresh()

    def stats(self):
        """当前模型连下 5 局（交替先后手）对当前对手，弹窗显示成绩。"""
        if self.busy or self.agent is None:
            return
        from .agent import RandomPlayer
        opp = self.opp if self.opp is not None else RandomPlayer()
        cfg = self.cfg
        seq = self.seq
        self.agent.sim = self.sim
        if hasattr(opp, "sim"):
            opp.sim = self.sim
        auto_was, self.auto = self.auto, False
        self.play_text.set("▶ 自动")

        def run():
            try:
                from .trainer import match as _match
                wr, st = _match(cfg, self.agent, opp, STATS_GAMES,
                                seed=int(self.rng.integers(1 << 20)), log=None)
                txt = (f"{STATS_GAMES} 局统计（交替先后手，深度 {self.sim}）\n"
                       f"主模型得分 {wr:.1%}    胜 {st['wins']} 负 {st['losses']} "
                       f"和 {st['draws']}    平均 {st['avg_len']:.0f} 手")
            except Exception as e:
                txt = f"统计失败: {e}"

            def show():
                if self.dead:
                    return
                self.busy = False
                if seq == self.seq:
                    self._set_status("统计失败" if txt.startswith("统计失败")
                                     else "统计完成")
                    messagebox.showinfo("统计", txt)
            self._post(show)

        self.busy = True
        threading.Thread(target=run, daemon=True).start()

    def _quit(self):
        self.dead = True
        self.destroy()


def run(cfg, device, model_path, mode="pvp", human=0, sim=400, opp_path=None,
        buffer_dir=None):
    app = App(cfg, device, model_path, mode=mode, human=human, sim=sim,
              opp_path=opp_path, buffer_dir=buffer_dir)
    app.mainloop()

"""控制台字符棋盘渲染与坐标解析。全部使用 ASCII 半角字符，避免宽度错位。"""
from __future__ import annotations

import re

from .game import Gomoku

SYMBOLS = {0: "X", 1: "O"}
EMPTY = "."

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def pos_name(pos: int, size: int) -> str:
    return f"{LETTERS[pos % size]}{pos // size + 1}"


def render(game: Gomoku, last: int | None = None) -> str:
    s = game.size
    if last is None:
        last = game.last
    head = "    " + " ".join(LETTERS[:s])
    lines = [head, "    " + "--" * (s - 1) + "-"]
    for r in range(s):
        cells = []
        for c in range(s):
            pos = r * s + c
            sym = EMPTY
            for pid in (0, 1):
                if game.stones[pid] & game.m["bit"][pos]:
                    sym = SYMBOLS[pid]
                    if pos == last:
                        sym = sym.lower()
                    break
            cells.append(sym)
        lines.append(f"{r + 1:>3}  " + " ".join(cells))
    return "\n".join(lines)


LEGEND = ("图例: X 黑棋  O 白棋  x/o 小写表示最后一手  坐标 = 列字母+行号，"
          "例如 'E5'；也接受 '5,5' 或 '5e'")


def parse_cell(text: str, size: int) -> int | None:
    """解析 'e5' / '5e' / '5,5' / '5 5'（行号 1 起），返回 pos 或 None。"""
    t = text.strip().lower().replace("，", ",").replace("；", ",").replace(";", ",")
    t2 = t.replace(" ", "").replace("\t", "")
    m = re.fullmatch(r"([a-z])(\d{1,2})", t2)
    if m:
        col = ord(m.group(1)) - ord("a")
        row = int(m.group(2)) - 1
    else:
        m = re.fullmatch(r"(\d{1,2})([a-z])", t2)
        if m:
            row = int(m.group(1)) - 1
            col = ord(m.group(2)) - ord("a")
        else:
            m = re.fullmatch(r"(\d{1,2})[ ,]+(\d{1,2})", t)
            if not m:
                return None
            row = int(m.group(1)) - 1
            col = int(m.group(2)) - 1
    if not (0 <= row < size and 0 <= col < size):
        return None
    return row * size + col


HELP = ("命令: 输入坐标落子；undo 悔棋(人机)；hint 让 AI 推荐一手；"
        "resign 认输；quit 退出")

"""全局常量、预设参数组与训练超参数配置。

细节参数集中管理，避免长命令行：
- train.json: 默认参数文件（DEFAULT_PARAM_FILE），主要参数都在里面改；
- PRESETS: 代码内置的整套参数组，--preset <name> 一键选用；
- 生效优先级：命令行单项 > --preset > 参数文件 > 本文件 Config 默认值。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

BOARD_SIZE = 9
WIN_LEN = 5
FEATURE_PLANES = 4

DEFAULT_MODEL = "runs/default/best.pt"   # play.py 默认主模型
DEFAULT_PARAM_FILE = "train.json"        # 运行时默认加载的参数文件名
CUDA_WORKER_CAP = 6                      # GPU 自对弈进程上限（每进程一个 CUDA 上下文）
CPU_WORKER_RESERVE = 2                   # CPU 自对弈进程数 = 核数 - 该保留数
EVAL_WORKER_RESERVE = 4                  # 评估 CPU 进程数 = 核数 - 该保留数
CPU_WORKER_COMMIT_MB = 480               # 实测：CPU 推理 worker 进程的提交内存占用
EVAL_COMMIT_HEADROOM_MB = 1500           # 开评估 CPU 池前须保留的提交内存余量
LIGHT_SIM_MIN = 24                       # 提示/胜率预判等轻量搜索的深度下限
LIGHT_SIM_DIV = 3                        # 轻量搜索深度 = max(下限, 主 sim // 该值)
STATS_GAMES = 5                          # GUI「N局统计」的局数

# 预设参数组：键名必须与 Config 字段一致，值覆盖参数文件、被命令行单项覆盖
PRESETS: dict[str, dict] = {
    # 不覆盖任何参数，完全走 命令行 > 参数文件 > Config 默认
    "default": {},
    # 冒烟：一两分钟验证整条流水线，独立 run 目录，不碰正式训练数据
    "smoke": {
        "run_dir": "runs/smoke", "iterations": 2,
        "games_per_iter": 4, "sim_selfplay": 32, "mcts_batch": 32,
        "train_steps": 50, "eval_games": 2, "sim_eval": 32,
        "eval_device": "cuda",   # 2 局评估不值得再开一个 CPU 进程池
    },
    # 快速：十几分钟走完「自对弈→训练→晋升」完整闭环
    "quick": {
        "run_dir": "runs/quick", "iterations": 3,
        "games_per_iter": 40, "sim_selfplay": 64,
        "train_steps": 500, "eval_games": 10, "sim_eval": 100,
    },
    # 强训：长时间正式训练，lr 衰减节奏同步加快
    "strong": {
        "run_dir": "runs/strong", "iterations": 100,
        "games_per_iter": 800, "sim_selfplay": 160,
        "train_steps": 4000, "eval_games": 60, "sim_eval": 256,
        "buffer_window": 15, "lr_decay_every": 20,
    },
}


@dataclass
class Config:
    size: int = BOARD_SIZE
    channels: int = 64
    blocks: int = 6

    sim_selfplay: int = 128
    sim_eval: int = 200
    sim_play: int = 400
    mcts_batch: int = 64
    c_puct: float = 1.6
    noise_eps: float = 0.25
    noise_alpha: float = 0.12
    tau_moves: int = 12
    resign_value: float = -0.95
    resign_moves: int = 20

    iterations: int = 30
    games_per_iter: int = 500
    train_steps: int = 2500
    batch_size: int = 512
    use_amp: bool = False  # GPU 混合精度开关。实测 1650 上本模型规模反而慢 ~3 倍
                           # （GradScaler 每步同步 + 小 kernel 发射受限），默认关；
                           # 换大通道/大批量时可用 --amp 再试
    lr_init: float = 2e-3
    lr_decay: float = 0.5
    lr_decay_every: int = 50
    weight_decay: float = 1e-4
    buffer_window: int = 10
    human_epochs: int = 0   # >0 时额外用人机对局数据训练 N 个 epoch

    eval_games: int = 40
    eval_threshold: float = 0.55
    eval_noise: float = 0.05
    no_eval: bool = False
    eval_device: str = "cpu"   # 晋升评估设备：cpu 独立进程池，与 GPU 训练重叠；
                               # cuda 则与自对弈共享进程池（评估延后一轮收集）
    eval_workers: int = 0      # 评估 CPU 进程数，0 为自动（核数 - 4）

    workers: int = 0
    device: str = "auto"
    sp_device: str = "auto"
    run_dir: str = "runs/default"
    seed: int = 0
    resume: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def param_fields() -> set[str]:
    """参数文件/预设允许的键集合（即 Config 全部字段名）。"""
    return {f.name for f in fields(Config)}


def load_param_file(path: str | Path) -> dict:
    """读取 JSON 参数文件。不存在返回 {}；未知键直接报错，防止拼错静默失效。"""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"参数文件 {p} 不是合法 JSON：{e}")
    if not isinstance(data, dict):
        raise SystemExit(f"参数文件 {p} 顶层必须是 JSON 对象")
    unknown = sorted(set(data) - param_fields())
    if unknown:
        raise SystemExit(f"参数文件 {p} 含未知键 {unknown}；"
                         f"可用键见 gobang/config.py 的 Config 字段")
    return data

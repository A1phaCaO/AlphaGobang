# AlphaGobang

五子棋强化学习项目，走 AlphaGo Zero 式路线：轻量残差 CNN（策略头 + 价值头）+ PUCT 蒙特卡洛树搜索，纯自对弈迭代训练，不依赖任何人类棋谱。默认 9×9 棋盘，五连即胜、无禁手，单张入门显卡（GTX 1650）即可跑通。

## 环境

Python 3.14+，依赖用 [uv](https://docs.astral.sh/uv/) 管理（见 `pyproject.toml` / `uv.lock`）。

## 快速开始

```bash
# 训练（超参集中在 train.json，不用敲长命令）
uv run python main_train.py

# 三档预设：冒烟 / 快速闭环 / 正式训练
uv run python main_train.py --preset smoke
uv run python main_train.py --preset quick
uv run python main_train.py --preset strong

# 换一份参数文件（train_5060.json 是 8GB 5060 的长跑配置）
uv run python main_train.py --config train_5060.json

# 训练不收敛先体检自对弈数据：标签里到底有没有内容
uv run python check_data.py runs/default

# 人机对战（图形窗口，点击落子）
uv run python play.py
uv run python play.py -o runs/default/best.pt   # 指定模型

# 控制台字符棋盘
uv run python play.py --cli
```

未训练前直接运行 `play.py` 会提示找不到模型，先跑一轮 `main_train.py`。

## 目录结构

```
gobang/        规则引擎、模型、MCTS、自对弈、数据、训练、UI
main_train.py  训练主循环：自对弈 -> 训练 -> 评估晋升
play.py        人机 / 机器 vs 机器 / 多局统计
check_data.py  自对弈数据体检（目标熵、gap、ce-H）
train.json     默认超参文件（train_5060.json 为 5060 长跑配置）
tests/         自检脚本
docs/          详细设计与调参笔记
```

## 更多说明

设计思路、参数优先级、训练日志解读、硬件实测与调参经验见 [docs/训练细节.md](docs/训练细节.md)。

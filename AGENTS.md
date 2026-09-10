# AGENTS.md

五子棋 AlphaGo Zero 式强化学习（Python 3.14 + PyTorch + tkinter GUI），Windows + uv 管理。

## 命令

- 一切 Python 都走 uv：`uv run python <file>`（解释器在 `.venv`，勿直接调系统 python）。
- 训练：`uv run python main_train.py`（`--preset smoke|quick|strong`，`--resume` 断点续训）。
- 对战：`uv run python play.py`（tkinter GUI；`--cli` 控制台，`--you white` 让 AI 先手）。
- 自检：**不是 pytest**，是独立脚本 `uv run python tests/test_core.py`（assert + print）。
- 数据体检：`uv run python check_data.py runs/default`（训练不收敛时先看自对弈标签）。
- 无 lint / format / typecheck / CI 配置，不要凭空发明命令。

## 工具链陷阱

- `torch==2.13.0+cu132` 来自南大 cu132 专用 index，PyPI 默认源是阿里云镜像（见 pyproject `[[tool.uv.index]]`）。改动依赖或 index 会装不出来。
- `main_train.py` 顶部在 import torch **之前**强制 OPENBLAS/MKL/OMP 线程数为 1：Windows 下 spawn 多个子进程 import torch 会按核数预分配 arena，撑爆提交内存。任何新入口若要多进程 spawn + torch，需要照做。
- 混合精度默认关闭（`use_amp=False`）：实测小模型在 1650 上 AMP 反而慢 3 倍，不要"顺手"打开。
- 自对弈进程数只有 `workers=0`（自动）才会套 `min(核数-2, CUDA_WORKER_CAP=6)`；参数文件里写死 `workers: 8` 会绕过这个上限（`gobang/selfplay.py:132`）。本地 1650 开 8 个 CUDA 上下文会直接 `CUDA-capable device(s) is/are busy`，5060 这类大显存卡才开得起。评估 CPU 池延迟到首轮训练后才创建，同样是提交内存问题。

## 配置体系（易踩坑）

- 生效优先级：**命令行单项 > `--preset` > 参数文件（默认 `train.json`）> `Config` 默认值**（`gobang/config.py`）。
- 参数文件里未知键直接 `SystemExit` 报错；`_` 开头的键视为注释放行。加 `Config` 字段才能配套加参数文件键，`main_train.py` 会把未映射到 Config 的命令行参数视为内部错误。
- `mcts_batch` 不是搜索预算，只是单批叶子数；`mcts.clamp_batch` 会收紧到 `sim // SEARCH_BATCH_ROUNDS`。batch ≥ sim 时整步搜索只有一次价值回传，访问分布退化为均匀分布、训练不收敛（历史事故，见 mcts.py 顶部注释与 git 405a177）。
- 样本数看日志「样本」列，别按 `games_per_iter` 估：每轮样本 ≈ 局数 x 平均局长，局长从 55 手掉到 20 手时 400 局只剩 8k 样本，`train_steps` 的等效 epoch 数会悄悄翻几倍。
- 加数据优先于加重复率：等效 epoch = `train_steps x batch_size / (buffer_window x 每轮样本)`，超过约 8 就先把 `games_per_iter` 翻倍，而不是加 `train_steps`（后者只会过拟合旧样本）。
- 改 `channels` / `blocks` 后旧权重不能 `--resume`（shape 不匹配）。日志抬头「参数量」是配置漂移的探针：908,954 = 64ch x 12块，1,685,178 = 96ch x 10块；远程那份 `train_5060.json` 常被手改，pull 前先 `rg -n "channels" train_5060.json` 核对。
- 晋升评估是纯开销，用 `eval_every` 摊薄；但自动算出的 `eval_workers` 在提交内存紧的机器上会建不起池子而退回串行（每 `eval_every` 轮白站约 190s），此时显式写小值（如 `eval_workers: 4`）。

## 训练不收敛的排查顺序

- 先 `uv run python check_data.py runs/<name>`，不要先调 lr。它只读 `runs/<name>/buffer/iter_*.npz`，不动权重、不用等训练停。
- 决定性指标是 `gap = ln(合法点数) - H(策略目标)`：gap≈0 说明访问分布是均匀的，标签本身就是随机落子，模型只是忠实地学随机。此时 ce 会紧贴标签熵（`ce - H_tgt ≈ 0`），表现为「在降但降不动」，属于搜索/infra bug，不是超参问题。
- 判定只看开局档（前 20 手），全盘均值会被残局骗过去：残局合法点本来就少，战术强制掩码还会把先验压到一两个点，gap 天然偏低。
- 健康读数参考（5060 修复后第 1~8 轮）：开局档 gap 2.0~2.4、`unif`（最大概率 < 0.05 的占比）0%、`ce-Hw` 约 1.2 且逐轮下降、`draw` 占比随棋力上升而下降。
- `mse` 反弹不一定是 bug：`draw` 变少会抬高价值地板，要看解释率 `1 - mse/(1-draw)` 是否仍在涨。
- 日志里 `自对弈0s` / `1.4M局/s` 是跨轮重叠生效（上一轮的自对弈已提前跑完），ETA 跳变同源，都不是故障。

## 已知未修：自对弈第 0 手随机点格

- `gobang/selfplay.py:47 play_one_game` 的第 0 手是在 81 个空点里带 Dirichlet 噪声选的。空盘点位只构成 15 个 D4 等价类、先验几乎相等，`noise_alpha=0.12` 下约 89% 概率落在天元 3x3 之外（其中四成落在边线），等于黑棋开局点废格。
- 症状是先手胜率异常偏低（实测 42% -> 30~37%）。两边共用同一份权重，唯一的不对称就是这一手，所以别把它当成棋盘不公平。
- 修法：自对弈和评估的第 0 手固定天元（8 个 D4 操作下唯一不动的格子，不破坏 D4 增强）。**尚未实施**，改动约 3 行。

## 架构要点

- 引擎/界面/训练分层：`gobang/game.py`（位棋盘规则引擎，`placed()` 返回新对象、不可变风格，MCTS 靠它复用状态）→ `mcts.py`（PUCT 批量叶子评估）→ `agent.py` → `selfplay.py`（多进程对局）→ `trainer.py`；入口在根目录。
- `game.py::encode()` 输出以**当前行棋方视角**的 4 个特征平面（己方/对方/上一手/先手恒置）。改平面数量或语义（`FEATURE_PLANES`）会与 `runs/*/*.pt` 旧权重不兼容，须重训。
- 五子棋规则黑棋必然先行，「谁先手」等价于「人类执黑还是执白」（GUI 下拉框 / `--you`），不需要另设先手开关。
- 训练产物全部落在 `runs/<name>/`（已 gitignore）：`latest.pt` / `best.pt`（原子写：tmp + replace，新增落盘逻辑要沿用）、`buffer/iter_*.npz`（自对弈）、`buffer/human/`（人机对局冷启动数据）、`meta.json`。
- `gui.py`：AI 推理一律在后台线程跑、经 `after()` 回投主线程（`_bg`/`_post`），并用 `self.seq` 作废过期回调——不要引入阻塞 Tk 主循环的代码，也不要用未校验 seq 的结果改界面。
- 代码注释/文档面向中文读者，新代码注释保持中文风格。

## 更多背景

设计思路与调参经验在 `docs/训练细节.md`（中文文件名，PowerShell 控制台会显示为乱码，用 glob 找）。

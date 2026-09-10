"""轻量残差 CNN：共享主干 + 策略头 + 价值头。"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .config import BOARD_SIZE, FEATURE_PLANES

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class GomokuNet(nn.Module):
    def __init__(self, size: int = BOARD_SIZE, channels: int = 64, blocks: int = 6,
                 in_planes: int = FEATURE_PLANES):
        super().__init__()
        self.size = size
        self.action_space = size * size
        self.arch = {"size": size, "channels": channels, "blocks": blocks}
        self.trunk = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            *[ResBlock(channels) for _ in range(blocks)],
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * self.action_space, self.action_space),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.action_space, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        f = self.trunk(x)
        return self.policy_head(f), self.value_head(f).squeeze(-1)


def build_net(cfg) -> GomokuNet:
    return GomokuNet(cfg.size, cfg.channels, cfg.blocks)


def save_ckpt(path: str | Path, net: GomokuNet, extra: dict | None = None):
    """先写 .tmp 再原子替换：中断/断电不会留下半截 checkpoint。"""
    obj = {"arch": net.arch, "model": net.state_dict()}
    obj.update(extra or {})
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_ckpt(path: str | Path, device: str = "cpu") -> tuple[GomokuNet, dict]:
    # checkpoint 只含张量与普通 dict，weights_only 安全且免反序列化任意对象
    obj = torch.load(path, map_location=device, weights_only=True)
    arch = obj.pop("arch")
    model = obj.pop("model")
    net = GomokuNet(**arch)
    net.load_state_dict(model)
    net.to(device)
    net.eval()
    return net, obj

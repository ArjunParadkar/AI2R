"""ResNet-18 (CIFAR stem) exactly as in the original exp_vision.py, plus a
CIFAR-10 loader with subset + symmetric label noise."""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
from torch.utils.data import Subset, DataLoader, TensorDataset

CIFAR_ROOT = "/home/god/whitebox_experiments_clone/data"


class BasicBlock(nn.Module):
    def __init__(self, i, p, s=1):
        super().__init__()
        self.conv1 = nn.Conv2d(i, p, 3, s, 1, bias=False); self.bn1 = nn.BatchNorm2d(p)
        self.conv2 = nn.Conv2d(p, p, 3, 1, 1, bias=False); self.bn2 = nn.BatchNorm2d(p)
        self.shortcut = nn.Sequential()
        if s != 1 or i != p:
            self.shortcut = nn.Sequential(nn.Conv2d(i, p, 1, s, bias=False), nn.BatchNorm2d(p))

    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x))); o = self.bn2(self.conv2(o)); return F.relu(o + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._mk(64, 64, 1); self.layer2 = self._mk(64, 128, 2)
        self.layer3 = self._mk(128, 256, 2); self.layer4 = self._mk(256, 512, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)   # named so the extractor can hook penultimate features
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _mk(i, p, s):
        return nn.Sequential(BasicBlock(i, p, s), BasicBlock(p, p, 1))

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.layer4(self.layer3(self.layer2(self.layer1(h))))
        return self.fc(self.pool(h).flatten(1))


def cifar_tensors(train: bool):
    mean = [0.4914, 0.4822, 0.4465]; std = [0.2470, 0.2435, 0.2616]
    ds = torchvision.datasets.CIFAR10(CIFAR_ROOT, train=train, download=True, transform=T.Compose([T.ToTensor(), T.Normalize(mean, std)]))
    X = torch.stack([ds[i][0] for i in range(len(ds))]); y = torch.tensor(ds.targets)
    return X, y


def noisy_subset(X, y, n_train: int, noise: float, seed: int):
    """Random subset of size n_train with symmetric label noise (a flipped label
    is always a *different* class). Returns X, y_noisy, y_clean, flipped mask."""
    rng = np.random.default_rng(seed); idx = rng.choice(len(X), n_train, replace=False)
    Xs, yc = X[idx], y[idx].clone(); yn = yc.clone()
    flip = torch.as_tensor(rng.random(n_train) < noise)
    yn[flip] = (yc[flip] + torch.as_tensor(rng.integers(1, 10, int(flip.sum())))) % 10
    return Xs, yn, yc, flip


def probe_from(X, y, n_per_class=50, seed=0):
    rng = np.random.default_rng(seed); idx = []
    for c in range(10):
        idx += rng.choice(np.where(y.numpy() == c)[0], n_per_class, replace=False).tolist()
    return X[idx], y[idx]

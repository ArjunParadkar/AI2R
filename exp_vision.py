"""Validation 1+2 (vision): ResNet-18 on CIFAR-10.

Per WHITEBOX §4.1/§4.2 + Appendix A:
  ResNet-18, SGD lr=0.1 + cosine annealing, batch 128, 100 epochs.
  Probe: 500 images (50/class), fixed at epoch 0.
  Layers: layer1[1], layer2[1], layer3[1], fc.
  PCA->30 (fit at epoch 0), Ripser maxdim=1, thresh = 95th-pct pairwise distance.
  Checkpoints every 2 epochs. 3 seeds.
  Outputs: SV(t), TC(t), Betti, class-conditional t_c*, training loss, train/test acc.
"""
from __future__ import annotations
import argparse, json, os, time, math
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA

import whitebox as wb


# --------------------------------------------------------------------------- #
# ResNet-18 for CIFAR-10 (3x3 stem; standard for cifar)
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, in_planes, planes, blocks, stride):
        layers = [BasicBlock(in_planes, planes, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)
    def forward(self, x, return_acts=False):
        h = F.relu(self.bn1(self.conv1(x)))
        a1 = self.layer1(h)
        a2 = self.layer2(a1)
        a3 = self.layer3(a2)
        a4 = self.layer4(a2 if False else a3)
        h = F.adaptive_avg_pool2d(a4, 1).flatten(1)
        out = self.fc(h)
        if return_acts:
            return out, {
                "layer1": a1.detach(),
                "layer2": a2.detach(),
                "layer3": a3.detach(),
                "fc":     h.detach(),  # pre-classifier features
            }
        return out


# --------------------------------------------------------------------------- #
def load_cifar10(root="/home/god/whitebox_experiments_clone/data"):
    Path(root).mkdir(parents=True, exist_ok=True)
    mean = [0.4914, 0.4822, 0.4465]; std = [0.2470, 0.2435, 0.2616]
    tr_train = T.Compose([
        T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize(mean, std),
    ])
    tr_test = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train = torchvision.datasets.CIFAR10(root, train=True,  download=True, transform=tr_train)
    test  = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tr_test)
    probe_tr = torchvision.datasets.CIFAR10(root, train=True, download=False, transform=tr_test)
    return train, test, probe_tr


def build_probe(probe_dataset, n_per_class=50, seed=0):
    rng = np.random.default_rng(seed)
    targets = np.array(probe_dataset.targets)
    indices = []
    for c in range(10):
        idx = np.where(targets == c)[0]
        choice = rng.choice(idx, size=n_per_class, replace=False)
        indices.extend(choice.tolist())
    return Subset(probe_dataset, indices), np.array([probe_dataset.targets[i] for i in indices])


@torch.no_grad()
def extract_activations(model, probe_loader, device):
    model.eval()
    acts = {"layer1": [], "layer2": [], "layer3": [], "fc": []}
    for x, _ in probe_loader:
        x = x.to(device, non_blocking=True)
        _, a = model(x, return_acts=True)
        for k in acts:
            v = a[k]
            if v.ndim == 4:
                v = F.adaptive_avg_pool2d(v, 1).flatten(1)
            acts[k].append(v.cpu().numpy())
    return {k: np.concatenate(v, 0) for k, v in acts.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0; n = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        n += y.numel()
    return correct / n


def train_one_seed(seed, epochs, ckpt_every, layers, label_noise=0.0, device="cuda",
                   out_dir="/home/god/Desktop/whitebox/results/vision",
                   tag="v1", probe_n_per_class=50, batch_size=128, lr=0.1):
    torch.manual_seed(seed); np.random.seed(seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train, test, probe_src = load_cifar10()

    if label_noise > 0:
        rng = np.random.default_rng(seed + 999)
        targets = np.array(train.targets)
        n_flip = int(label_noise * len(targets))
        flip_idx = rng.choice(len(targets), size=n_flip, replace=False)
        new_lbl = rng.integers(0, 10, size=n_flip)
        targets[flip_idx] = new_lbl
        train.targets = targets.tolist()

    probe_set, probe_labels = build_probe(probe_src, probe_n_per_class, seed=0)  # fixed across seeds
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    test_loader  = DataLoader(test,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    probe_loader = DataLoader(probe_set, batch_size=256, shuffle=False, num_workers=2)

    model = ResNet18().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")

    # ----- Epoch 0: extract activations and fit PCA per layer -----
    acts0 = extract_activations(model, probe_loader, device)
    pcas = {k: wb.fit_pca(acts0[k], n_components=30) for k in layers}

    # storage
    diag_film = {k: [] for k in layers}             # global persistence diagrams
    classcond_diag = {k: {c: [] for c in range(10)} for k in layers}  # per class
    train_losses = []
    train_accs = []; test_accs = []
    epoch_marks = []  # checkpoint epochs

    def record_topology(acts, epoch):
        for k in layers:
            X = acts[k]
            D = wb.persistence_diagrams(X, pcas[k], maxdim=1)
            diag_film[k].append(D)
            # class-conditional only for `fc` layer (for crystallization)
        if "fc" in layers:
            X = acts["fc"]
            for c in range(10):
                Xc = X[probe_labels == c]
                Dc = wb.persistence_diagrams(Xc, pcas["fc"], maxdim=1)
                classcond_diag["fc"][c].append(Dc)

    # epoch 0 record
    record_topology(acts0, 0)
    epoch_marks.append(0)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0; ncnt = 0; cor = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(x)
                loss = F.cross_entropy(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            running += loss.item() * y.size(0); ncnt += y.size(0)
            cor += (out.argmax(1) == y).sum().item()
        sched.step()
        train_loss = running / ncnt
        train_acc  = cor / ncnt
        train_losses.append(train_loss); train_accs.append(train_acc)
        if epoch % ckpt_every == 0 or epoch == epochs:
            acts = extract_activations(model, probe_loader, device)
            record_topology(acts, epoch)
            test_acc = evaluate(model, test_loader, device)
            test_accs.append(test_acc)
            epoch_marks.append(epoch)
            elapsed = time.time() - t0
            print(f"[{tag} seed={seed}] ep {epoch:3d}/{epochs} "
                  f"loss={train_loss:.3f} train={train_acc:.3f} test={test_acc:.3f} "
                  f"({elapsed:.1f}s)", flush=True)

    # ----- Compute SV(t), TC(t) trajectories from diag_film -----
    def to_traj(film):
        sv = [0.0]
        tc = [wb.topological_complexity(film[0][1])]
        b0 = [len(film[0][0])]; b1 = [len(film[0][1])]
        for i in range(1, len(film)):
            sv.append(wb.structural_velocity(film[i][1], film[i-1][1]))
            tc.append(wb.topological_complexity(film[i][1]))
            b0.append(len(film[i][0])); b1.append(len(film[i][1]))
        return np.array(sv), np.array(tc), np.array(b0), np.array(b1)

    metrics = {}
    for k in layers:
        sv, tc, b0, b1 = to_traj(diag_film[k])
        metrics[k] = {"SV": sv.tolist(), "TC": tc.tolist(),
                      "B0": b0.tolist(), "B1": b1.tolist()}

    # class-conditional SV(t) per class for `fc`
    classcond_sv = {}
    if "fc" in layers:
        for c in range(10):
            film_c = classcond_diag["fc"][c]
            sv_c = [0.0]
            for i in range(1, len(film_c)):
                sv_c.append(wb.structural_velocity(film_c[i][1], film_c[i-1][1]))
            classcond_sv[c] = sv_c

    out = {
        "seed": seed, "epochs": epochs, "ckpt_every": ckpt_every,
        "label_noise": label_noise, "tag": tag,
        "epoch_marks": epoch_marks,
        "train_loss": train_losses,
        "train_acc": train_accs,
        "test_acc": test_accs,
        "metrics": metrics,
        "classcond_sv": classcond_sv,
    }
    out_path = out_dir / f"{tag}_seed{seed}_noise{int(label_noise*100)}.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--tag", type=str, default="v1")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    layers = ["layer1", "layer2", "layer3", "fc"]
    for s in args.seeds:
        train_one_seed(s, epochs=args.epochs, ckpt_every=args.ckpt_every,
                       layers=layers, label_noise=args.noise, device=args.device,
                       tag=args.tag)

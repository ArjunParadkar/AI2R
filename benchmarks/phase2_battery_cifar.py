"""Phase 2 battery (vision): ResNet-18 on a 5k-image CIFAR-10 subset, no
augmentation, symmetric label noise in {0,20,40,60,80,100}% x 3 seeds, 40
epochs so that noisy labels are actually *memorised* (train-acc on the noisy
labels reported). Plus a 'strong generaliser' per seed: full 50k, flip+crop
augmentation, 12 epochs. For every model we save the probe activations
(500 train images seen at epoch 0, 50/class) at layers layer3 (pooled) and
pool (penultimate), at epochs {10,20,30,40} + final, and the ground truth:
train acc on noisy labels, train acc on clean labels, test acc, and the
fraction of *flipped* labels the model reproduces (memorisation fraction)."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models import ResNet18, cifar_tensors, noisy_subset, probe_from
from whitebox.activations import ActivationExtractor
from common import RESULTS

OUT = f"{RESULTS}/battery_cifar"; os.makedirs(OUT, exist_ok=True); DEV = "cuda"
NOISES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]; SEEDS = [0, 1, 2]; N_TRAIN = 5000; EPOCHS = 40
Xtr, ytr = cifar_tensors(True); Xte, yte = cifar_tensors(False)
Xp, yp = probe_from(Xtr, ytr, 50, seed=0)
probe_batches = [(Xp[i:i + 250], None) for i in range(0, 500, 250)]


@torch.no_grad()
def acc(model, X, y, bs=1000):
    model.eval(); c = 0
    for i in range(0, len(X), bs):
        c += (model(X[i:i + bs].to(DEV)).argmax(1).cpu() == y[i:i + bs]).sum().item()
    return c / len(X)


def train(tag, Xs, yn, yc, flip, epochs, augment, lr, seed):
    torch.manual_seed(seed); model = ResNet18().to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs); scaler = torch.amp.GradScaler("cuda")
    ex = ActivationExtractor(model, ["layer3", "pool"]); ck = {}
    dl = DataLoader(TensorDataset(Xs, yn), 128, shuffle=True, drop_last=True)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            if augment:
                if torch.rand(1) < 0.5: x = x.flip(3)
                pad = F.pad(x, (4, 4, 4, 4), mode="reflect"); i, j = np.random.randint(0, 9, 2); x = pad[:, :, i:i + 32, j:j + 32]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        if ep % 10 == 0 or ep == epochs:
            a = ex(probe_batches); ck[ep] = {k: v.astype(np.float32) for k, v in a.items()}
    gt = dict(train_acc_noisy=acc(model, Xs, yn), train_acc_clean=acc(model, Xs, yc), test_acc=acc(model, Xte, yte),
              seconds=time.time() - t0, epochs=epochs, n_train=len(Xs))
    with torch.no_grad():
        model.eval(); pred = torch.cat([model(Xs[i:i + 1000].to(DEV)).argmax(1).cpu() for i in range(0, len(Xs), 1000)])
    gt["memorized_flipped_frac"] = float((pred[flip] == yn[flip]).float().mean()) if flip.any() else float("nan")
    gt["generalization_gap"] = gt["train_acc_noisy"] - gt["test_acc"]
    np.savez_compressed(f"{OUT}/{tag}.npz", **{f"ep{e}_{k}": v for e, a in ck.items() for k, v in a.items()})
    json.dump(gt, open(f"{OUT}/{tag}.json", "w"), indent=1); ex.remove()
    print(tag, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in gt.items()}), flush=True)


for seed in SEEDS:
    for noise in NOISES:
        tag = f"resnet18_n{int(noise * 100):03d}_s{seed}"
        if os.path.exists(f"{OUT}/{tag}.json"): continue
        Xs, yn, yc, flip = noisy_subset(Xtr, ytr, N_TRAIN, noise, seed)
        train(tag, Xs, yn, yc, flip, EPOCHS, False, 0.05, seed)
    tag = f"resnet18_full_aug_s{seed}"
    if not os.path.exists(f"{OUT}/{tag}.json"):
        train(tag, Xtr, ytr, ytr, torch.zeros(len(ytr), dtype=torch.bool), 12, True, 0.1, seed)
print("cifar battery done")

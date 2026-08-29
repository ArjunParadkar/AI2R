"""Does the memorization signal appear DURING training, or only once
memorisation is complete? Features at every saved checkpoint of the CIFAR
battery (epochs 10..40) and GPT-2 (0,5,10), penultimate layer."""
import csv, glob, json, os, numpy as np
from whitebox.signals.memorization import memorization_features, Calibration
from common import RESULTS
rows = []
for p in sorted(glob.glob(f"{RESULTS}/battery_cifar/*.npz")):
    tag = os.path.basename(p)[:-4]; z = np.load(p); noise = 0 if "full" in tag else int(tag.split("_n")[1][:3])
    for k in sorted(z.files, key=lambda s: int(s.split("_")[0][2:])):
        if not k.endswith("_pool"): continue
        f = memorization_features(z[k], distance_device="cuda"); rows.append(dict(arch="resnet18", tag=tag, noise=noise, epoch=int(k.split("_")[0][2:]), **f))
for p in sorted(glob.glob(f"{RESULTS}/battery_gpt2/*.npz")):
    tag = os.path.basename(p)[:-4]; z = np.load(p); noise = int(tag.split("_n")[1][:3])
    for k in sorted(z.files, key=lambda s: int(s.split("_")[0][2:])):
        if not k.endswith("_L12"): continue
        f = memorization_features(z[k], distance_device="cuda"); rows.append(dict(arch="gpt2", tag=tag, noise=noise, epoch=int(k.split("_")[0][2:]), **f))
with open(f"{RESULTS}/phase2_trajectory.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
cals = {"resnet18": Calibration.load("cnn_classifier"), "gpt2": Calibration.load("lm_transformer")}
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
archs = sorted({r["arch"] for r in rows}); fig, ax = plt.subplots(1, len(archs), figsize=(6 * len(archs), 4), squeeze=False)
for a, arch in zip(ax[0], archs):
    rs = [r for r in rows if r["arch"] == arch]; noises = sorted({r["noise"] for r in rs}); cm = plt.cm.viridis
    for i, n in enumerate(noises):
        tags = sorted({r["tag"] for r in rs if r["noise"] == n})
        for t in tags:
            tr = sorted([r for r in rs if r["tag"] == t], key=lambda r: r["epoch"])
            a.plot([r["epoch"] for r in tr], [cals[arch].score(r) for r in tr], "-o", ms=3, color=cm(i / max(1, len(noises) - 1)), label=f"noise {n}%" if t == tags[0] else None, alpha=0.8)
    a.axhline(0.5, color="k", ls=":", lw=0.7); a.set_xlabel("epoch"); a.set_ylabel("calibrated memorization score"); a.set_title(f"{arch}: score along training (penultimate layer)"); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{RESULTS}/phase2_trajectory.png", dpi=300); print("ok", len(rows))

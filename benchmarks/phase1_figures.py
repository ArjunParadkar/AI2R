import csv, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
from common import RESULTS
t = list(csv.DictReader(open(f"{RESULTS}/phase1_gpt2_timing.csv"))); wall = list(csv.DictReader(open(f"{RESULTS}/phase1_wall.csv")))
rep = json.load(open(f"{RESULTS}/phase1_gpt2_report.json"))
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for b in ("ripser", "giotto-ph", "giotto-ph-nocollapse", "landmark"):
    r = [x for x in t if x["backend"] == b]; ax[0].plot([int(x["n"]) for x in r], [float(x["seconds"]) for x in r], "o-", label=b)
ax[0].plot([int(x["n"]) for x in wall], [float(x["seconds"]) for x in wall], "k^--", label="giotto-ph nocollapse (wall run)")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("probe size n"); ax[0].set_ylabel("PH seconds (L12, PCA30, thresh95)"); ax[0].legend(fontsize=7); ax[0].set_title("PH wall-clock vs probe size")
ax[1].plot([int(x["n"]) for x in wall], [float(x["TC"]) for x in wall], "o-"); ax[1].set_xlabel("probe size n"); ax[1].set_ylabel("TC (H1)"); ax[1].set_title("TC grows with n: raw TC is size-dependent")
names = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]; P = [124e6] + [rep[n]["n_params"] for n in names[1:]]
ext = [rep["gpt2_extract_2000_seconds"] / 4] + [rep[n]["extract_seconds"] for n in names[1:]]
frm = [rep["gpt2_frame_5layers_n500_boot20_seconds"]] + [rep[n]["frame_seconds"] for n in names[1:]]
ax[2].plot(P, ext, "o-", label="activation extraction (500 sents, 5 layers)"); ax[2].plot(P, frm, "s-", label="PH frame: 5 layers x (1 + 20 bootstraps)")
ax[2].set_xscale("log"); ax[2].set_xlabel("parameters"); ax[2].set_ylabel("seconds per epoch"); ax[2].legend(fontsize=7); ax[2].set_title("Per-epoch cost vs model size (RTX 5070 8GB)")
fig.tight_layout(); fig.savefig(f"{RESULTS}/phase1_timing.png", dpi=300); print("ok")

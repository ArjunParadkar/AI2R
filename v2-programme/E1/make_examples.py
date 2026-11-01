"""E1 example figures: (A) a planted gauge-equivalent pair (same circuit under a random permutation +
scaling; canonical forms coincide), (B) two genuinely distinct minimal circuits of one trained network
(different neuron subsets, different gate assignments, both valid under mean ablation)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1lib.functions import truth_table
from e1lib.models import MLP
from e1lib.enumerate import enumerate_valid, minimal_ids, circuit_layers, binarised_acts, mask_from_id
from e1lib.gauge import random_gauge, apply_gauge, permute_mask_id, canonical_form, sub_implementation
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
X, y = truth_table("XOR3")
GATES = {(0, 0, 0, 0, 0, 0, 0, 0): "0", (1, 1, 1, 1, 1, 1, 1, 1): "1"}


def gate_name(bits):
    """Name the Boolean function of the 3 inputs realised by a unit's bits, if a standard one."""
    b = tuple(int(v) for v in bits)
    x0, x1, x2 = X[:, 0], X[:, 1], X[:, 2]
    cands = {"x0": x0, "x1": x1, "x2": x2, "¬x0": 1 - x0, "¬x1": 1 - x1, "¬x2": 1 - x2,
             "x0∧x1": x0 * x1, "x0∧x2": x0 * x2, "x1∧x2": x1 * x2, "x0∨x1": np.maximum(x0, x1),
             "x0∨x2": np.maximum(x0, x2), "x1∨x2": np.maximum(x1, x2), "x0⊕x1": (x0 + x1) % 2,
             "x0⊕x2": (x0 + x2) % 2, "x1⊕x2": (x1 + x2) % 2, "XOR3": X.sum(1) % 2, "¬XOR3": 1 - X.sum(1) % 2,
             "MAJ": (X.sum(1) >= 2).astype(int), "AND3": (X.sum(1) == 3).astype(int), "OR3": (X.sum(1) >= 1).astype(int),
             "0": np.zeros(8, int), "1": np.ones(8, int)}
    for k, v in cands.items():
        if tuple(int(t) for t in v) == b:
            return k
    for k, v in cands.items():
        if tuple(1 - int(t) for t in v) == b:
            return "¬(" + k + ")"
    return "".join(map(str, b))


def draw_circuit(ax, model, mid, X, title):
    layers = circuit_layers(mid, model.widths)
    B, out = binarised_acts(model, X, mid, "mean")
    off = np.cumsum([0] + list(model.widths))
    cols = [list(range(model.n_in))] + [list(u) for u in layers] + [[0]]
    pos = {}
    for li, units in enumerate(cols):
        for k, u in enumerate(units):
            pos[(li, u)] = (li, k - (len(units) - 1) / 2)
    Ws, bs, _ = sub_implementation(model, mid)
    prev = list(range(model.n_in))
    for l, units in enumerate(layers + [[0]]):
        W = Ws[l]
        for i, pu in enumerate(prev):
            for j, u in enumerate(units):
                w = W[i, j]
                ax.plot(*zip(pos[(l, pu)], pos[(l + 1, u)]), color="tab:red" if w > 0 else "tab:blue",
                        lw=0.4 + 2.5 * abs(w) / (abs(W).max() + 1e-9), alpha=0.8, zorder=1)
        prev = units
    for li, units in enumerate(cols):
        for u in units:
            x_, y_ = pos[(li, u)]
            if li == 0:
                lab = f"x{u}"
            elif li == len(cols) - 1:
                lab = "out"
            else:
                lab = f"h{li}[{u}]\n" + gate_name(B[:, off[li - 1] + u])
            ax.scatter([x_], [y_], s=900, color="white", edgecolor="k", zorder=2)
            ax.text(x_, y_, lab, ha="center", va="center", fontsize=6, zorder=3)
    ax.set_title(title, fontsize=8); ax.axis("off")


# pick a network: XOR3 d2 w6, first seed with >= 2 minimal mean-valid circuits with different gate multisets
chosen = None
for seed in range(24):
    p = os.path.join(RES, "networks", f"XOR3_d2_w6_s{seed}.npz")
    if not os.path.exists(p):
        continue
    m = MLP.load(p)
    mins = minimal_ids(enumerate_valid(m, X, y, "mean"))
    if len(mins) >= 2:
        keys = {}
        for c in mins:
            B, _ = binarised_acts(m, X, int(c), "mean")
            layers = circuit_layers(int(c), m.widths)
            off = np.cumsum([0] + list(m.widths))
            keys[int(c)] = tuple(tuple(sorted(gate_name(B[:, off[l] + u]) for u in units)) for l, units in enumerate(layers))
        distinct = {}
        for c, k in keys.items():
            distinct.setdefault(k, c)
        if len(distinct) >= 2:
            chosen = (seed, m, list(distinct.values())[:2], mins)
            break
seed, model, (cA, cB), mins = chosen
print(f"network XOR3 d2 w6 seed {seed}: {len(mins)} minimal mean-valid circuits; showing {cA} and {cB}")

# (A) planted gauge pair
rng = np.random.default_rng(0)
g_model, perms, scales = random_gauge(model, rng)
g_mid = permute_mask_id(cA, perms, model.widths)
assert (g_model.predict(X) == y).all()
kA = canonical_form(model, cA, 0.05)[0]; kG = canonical_form(g_model, g_mid, 0.05)[0]
print("planted pair canonical forms equal:", kA == kG)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), dpi=300)
draw_circuit(axes[0], model, cA, X, f"circuit {cA:#x} in trained net (seed {seed})")
draw_circuit(axes[1], g_model, g_mid, X, "same circuit after random permutation + scaling\n(canonical forms identical → one gauge orbit)")
fig.suptitle("E1 example A — a gauge-equivalent pair (planted; none occur across seeds)", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_example_gauge_pair.png"))

# (B) two genuinely distinct circuits
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), dpi=300)
draw_circuit(axes[0], model, cA, X, f"minimal circuit {cA:#x}")
draw_circuit(axes[1], model, cB, X, f"minimal circuit {cB:#x}")
fig.suptitle(f"E1 example B — two genuinely distinct valid circuits of one network (XOR3 d2 w6 seed {seed}, mean ablation)\n"
             "different neuron subsets and different gate assignments; no g ∈ G maps one to the other", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_example_distinct_pair.png"))
print("saved")

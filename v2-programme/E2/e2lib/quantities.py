"""The candidate quantities of SPEC (I1-I8 expected invariant, N1-N6 the indictment list),
all computed from one MLP in one call so orbit sampling costs one forward pass per rep.

Persistence via v1 whitebox (Vietoris-Rips, H0+H1, 95th-percentile threshold), metrics
TC / Betti-at-median-scale / persistent entropy from whitebox.metrics.
"""
import numpy as np
from whitebox import compute_diagrams, topological_complexity, betti_at_scale, median_scale
from whitebox.metrics.tc import persistent_entropy

# name -> (family, spec_id, expected-invariant?, description)
QUANTITY_INFO = {
    "logits_tt": ("sanity", "I1", True, "logits on the truth table (f itself)"),
    "input_grad_tt": ("sanity", "I2", True, "d logit / d x at each truth-table input"),
    "W_product": ("sanity", "I3", True, "W0 W1 ... Wd (linear part only)"),
    "sv_layer": ("spectrum", "I4", True, "singular values of every hidden layer map + output row norm"),
    "tc0_tt": ("topology", "I5", True, "TC(H0) of truth-table activation cloud, per layer"),
    "tc1_tt": ("topology", "I5", True, "TC(H1) of truth-table activation cloud, per layer"),
    "tc0_box": ("topology", "I5", True, "TC(H0) of box activation cloud, per layer"),
    "tc1_box": ("topology", "I5", True, "TC(H1) of box activation cloud, per layer"),
    "b0_tt": ("topology", "I6", True, "beta0 at median scale, truth-table cloud"),
    "b1_tt": ("topology", "I6", True, "beta1 at median scale, truth-table cloud"),
    "b0_box": ("topology", "I6", True, "beta0 at median scale, box cloud"),
    "b1_box": ("topology", "I6", True, "beta1 at median scale, box cloud"),
    "pe0_box": ("topology", "I7", True, "persistent entropy H0, box cloud"),
    "pe1_box": ("topology", "I7", True, "persistent entropy H1, box cloud"),
    "tc0_box_std": ("topology_repair", "I8", True, "TC(H0) of per-unit standardised box cloud"),
    "tc1_box_std": ("topology_repair", "I8", True, "TC(H1) of per-unit standardised box cloud"),
    "act_mag": ("indictment", "N1", False, "per-neuron mean |activation| (raw index)"),
    "act_mag_sorted": ("indictment_sorted", "N1", False, "N1 sorted within layer"),
    "gradxact": ("indictment", "N2", False, "per-neuron grad x act attribution (raw index)"),
    "gradxact_sorted": ("indictment_sorted", "N2", False, "N2 sorted within layer"),
    "ablation": ("indictment", "N3", False, "per-neuron zero-ablation effect (raw index)"),
    "ablation_sorted": ("indictment_sorted", "N3", False, "N3 sorted within layer"),
    "in_norm": ("indictment", "N4", False, "per-neuron incoming weight norm (raw index)"),
    "in_norm_sorted": ("indictment_sorted", "N4", False, "N4 sorted within layer"),
    "W_abs": ("indictment", "N4", False, "all |W| entries (raw index)"),
    "top_neuron_mag": ("indictment_discrete", "N5", False, "argmax neuron by |activation|"),
    "top_neuron_abl": ("indictment_discrete", "N5", False, "argmax neuron by ablation effect"),
    "gates": ("indictment_discrete", "N6", False, "per-neuron binarised truth table [a>0] (raw index)"),
    "gates_sorted": ("indictment_sorted", "N6", False, "N6 as per-layer sorted multiset"),
    "n_separations": ("indictment_discrete", "N6", False, "per-neuron Meloux separation count (raw index)"),
}


def _relu_masks(model, X):
    """Pre-ReLU masks and post-ReLU acts per hidden layer."""
    h = X
    masks, acts = [], []
    for W, b in zip(model.Ws[:-1], model.bs[:-1]):
        pre = h @ W + b
        m = (pre > 0).astype(np.float64)
        h = np.maximum(pre, 0.0)
        masks.append(m)
        acts.append(h)
    return masks, acts


def input_gradients(model, X):
    """d logit / d x for each row of X (exact for a ReLU net inside each activation region)."""
    masks, _ = _relu_masks(model, X)
    N = X.shape[0]
    g = np.broadcast_to(model.Ws[-1][:, 0], (N, model.Ws[-1].shape[0])).copy()   # (N, w_d)
    for l in range(model.depth - 1, -1, -1):
        g = g * masks[l]                       # through ReLU
        g = g @ model.Ws[l].T                  # (N, fan_in)
    return g


def hidden_gradients(model, X):
    """d logit / d a_l (post-ReLU activations of each hidden layer), list of (N, w_l)."""
    masks, _ = _relu_masks(model, X)
    N = X.shape[0]
    g = np.broadcast_to(model.Ws[-1][:, 0], (N, model.Ws[-1].shape[0])).copy()
    grads = [None] * model.depth
    grads[model.depth - 1] = g
    for l in range(model.depth - 1, 0, -1):
        g = (g * masks[l]) @ model.Ws[l].T
        grads[l - 1] = g
    return grads


def ablation_effects(model, X):
    """Mean |logit - logit with unit j zero-ablated| over X, per hidden unit (global index)."""
    base = model.logits(X)
    out = []
    for l in range(model.depth):
        for j in range(model.widths[l]):
            h = X
            for k, (W, b) in enumerate(zip(model.Ws[:-1], model.bs[:-1])):
                h = np.maximum(h @ W + b, 0.0)
                if k == l:
                    h = h.copy(); h[:, j] = 0.0
            lg = (h @ model.Ws[-1] + model.bs[-1])[:, 0]
            out.append(np.mean(np.abs(lg - base)))
    return np.asarray(out)


def cloud_summary(A, standardise=False):
    """TC0, TC1, b0, b1, pe0, pe1 of a point cloud (rows = points)."""
    A = np.asarray(A, dtype=np.float64)
    if standardise:
        sd = A.std(0)
        keep = sd > 1e-12
        A = (A[:, keep] - A[:, keep].mean(0)) / sd[keep] if keep.any() else np.zeros((A.shape[0], 1))
    D = compute_diagrams(A, maxdim=1, backend="ripser")
    d0, d1 = D[0], D[1]
    return {"tc0": topological_complexity(d0), "tc1": topological_complexity(d1),
            "b0": betti_at_scale(d0, median_scale(d0)), "b1": betti_at_scale(d1, median_scale(d1)),
            "pe0": persistent_entropy(d0), "pe1": persistent_entropy(d1)}


def box_inputs(n_in, seed, n=200, box=(-1.0, 2.0)):
    rng = np.random.default_rng(seed)
    return rng.uniform(box[0], box[1], size=(n, n_in))


def _sorted_per_layer(vec, widths):
    out, off = [], 0
    for w in widths:
        out.append(np.sort(vec[off:off + w])); off += w
    return np.concatenate(out)


def gate_bits(model, X):
    """[a > 0] per hidden unit over X, shape (N, n_hidden), int8."""
    _, acts = _relu_masks(model, X)
    return np.concatenate([(a > 0).astype(np.int8) for a in acts], axis=1)


def separation_counts(model, X):
    _, acts = _relu_masks(model, X)
    A = np.concatenate(acts, axis=1)
    return np.asarray([len(np.unique(np.round(A[:, j], 9))) - 1 for j in range(A.shape[1])])


def all_quantities(model, X_tt, X_box=None, with_topology=True):
    """Every SPEC quantity as a dict name -> ndarray."""
    widths = model.widths
    q = {}
    q["logits_tt"] = model.logits(X_tt)
    q["input_grad_tt"] = input_gradients(model, X_tt)
    P = model.Ws[0]
    for W in model.Ws[1:]:
        P = P @ W
    q["W_product"] = P
    sv = [np.linalg.svd(W, compute_uv=False) for W in model.Ws[:-1]]
    sv.append(np.linalg.norm(model.Ws[-1]).reshape(1))
    q["sv_layer"] = np.concatenate(sv)

    masks, acts = _relu_masks(model, X_tt)
    if with_topology:
        for l, A in enumerate(acts):
            s = cloud_summary(A)
            for k in ("tc0", "tc1", "b0", "b1"):
                q.setdefault(f"{k}_tt", []).append(s[k])
        if X_box is not None:
            _, acts_box = _relu_masks(model, X_box)
            for l, A in enumerate(acts_box):
                s = cloud_summary(A)
                for k in ("tc0", "tc1", "b0", "b1", "pe0", "pe1"):
                    q.setdefault(f"{k}_box", []).append(s[k])
                s2 = cloud_summary(A, standardise=True)
                q.setdefault("tc0_box_std", []).append(s2["tc0"])
                q.setdefault("tc1_box_std", []).append(s2["tc1"])
        for k in list(q):
            if isinstance(q[k], list):
                q[k] = np.asarray(q[k], dtype=np.float64)

    act_mag = np.concatenate([np.abs(a).mean(0) for a in acts])
    q["act_mag"] = act_mag
    q["act_mag_sorted"] = _sorted_per_layer(act_mag, widths)
    grads = hidden_gradients(model, X_tt)
    gxa = np.concatenate([(g * a).sum(0) for g, a in zip(grads, acts)])
    q["gradxact"] = gxa
    q["gradxact_sorted"] = _sorted_per_layer(gxa, widths)
    abl = ablation_effects(model, X_tt)
    q["ablation"] = abl
    q["ablation_sorted"] = _sorted_per_layer(abl, widths)
    in_norm = np.concatenate([np.sqrt((W ** 2).sum(0) + b ** 2) for W, b in zip(model.Ws[:-1], model.bs[:-1])])
    q["in_norm"] = in_norm
    q["in_norm_sorted"] = _sorted_per_layer(in_norm, widths)
    q["W_abs"] = np.concatenate([np.abs(W).ravel() for W in model.Ws])
    q["top_neuron_mag"] = np.asarray([int(np.argmax(act_mag))])
    q["top_neuron_abl"] = np.asarray([int(np.argmax(abl))])
    B = gate_bits(model, X_tt)
    q["gates"] = B.astype(np.float64)
    # sorted multiset per layer: sort columns lexicographically within each layer
    cols, off = [], 0
    for w in widths:
        blk = B[:, off:off + w]
        order = np.lexsort(blk[::-1])
        cols.append(blk[:, order]); off += w
    q["gates_sorted"] = np.concatenate(cols, axis=1).astype(np.float64)
    q["n_separations"] = separation_counts(model, X_tt).astype(np.float64)
    return q


def gate_agreement(values):
    """N6 raw-index agreement: fraction of reps in which the bit-vector at index i equals the
    base one, averaged over units i. values: (R, N, n_hidden)."""
    V = np.asarray(values)
    same = (V == V[0][None]).all(axis=1)     # (R, n_hidden)
    return float(same.mean())


def scaling_sweep(model, X_tt, X_box, alphas=None):
    """Uniform scaling of ONE hidden layer at a time by alpha; returns rows of
    (layer, alpha, tc0_tt, tc1_tt, b0_tt, b1_tt, tc0_box, tc1_box, b0_box, b1_box, pe0_box, pe1_box)."""
    from e1lib.gauge import apply_gauge
    if alphas is None:
        alphas = np.logspace(-2, 2, 9)
    rows = []
    for l in range(model.depth):
        for a in alphas:
            scales = [np.ones(w) for w in model.widths]
            scales[l] = scales[l] * a
            m = apply_gauge(model, None, scales)
            _, acts = _relu_masks(m, X_tt)
            _, acts_box = _relu_masks(m, X_box)
            s_tt = cloud_summary(acts[l]); s_box = cloud_summary(acts_box[l])
            rows.append({"layer": l, "alpha": float(a),
                         **{f"{k}_tt": s_tt[k] for k in ("tc0", "tc1", "b0", "b1")},
                         **{f"{k}_box": s_box[k] for k in ("tc0", "tc1", "b0", "b1", "pe0", "pe1")}})
    return rows


def fit_exponent(alphas, values):
    """Slope k and R^2 of log(values) vs log(alphas); NaN if any value <= 0."""
    a = np.log(np.asarray(alphas, dtype=np.float64)); v = np.asarray(values, dtype=np.float64)
    if (v <= 0).any() or len(v) < 3:
        return float("nan"), float("nan")
    lv = np.log(v)
    k, c = np.polyfit(a, lv, 1)
    pred = k * a + c
    ss = np.sum((lv - lv.mean()) ** 2)
    r2 = 1.0 - np.sum((lv - pred) ** 2) / ss if ss > 0 else 1.0
    return float(k), float(r2)

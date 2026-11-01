"""Fly head-direction connectome loader (read-only; nothing under the Fiete-lab tree is written).
W[i,j] = presynaptic i -> postsynaptic j (settled convention, InvProj1/README.md 2026-08-27)."""
import os, re, numpy as np

RAW = "/home/god/MIT Fiete Lab/InvProj1/data/raw"
CORE_TYPES = ("EPG(PB08)", "PEG(PB07)", "PEN_a(PB06a)", "PEN_b(PB06b)", "Delta7(PB15)")

def load_connectome(core=True, raw_dir=RAW):
    """Returns dict(W, labels, types, wedges, idx). core=True -> the 148-neuron EPG/PEG/PEN_a/PEN_b/Delta7 core."""
    W = np.loadtxt(os.path.join(raw_dir, "weight_matrix.csv"), delimiter=",")
    labels = [l.strip() for l in open(os.path.join(raw_dir, "neuron_labels.csv")) if l.strip()]
    assert W.shape == (len(labels), len(labels)), (W.shape, len(labels))
    types = np.array([l.split(")")[0] + ")" for l in labels])                      # e.g. 'Delta7(PB15)_L1L9R8_R' -> 'Delta7(PB15)'
    wedges = np.array([l.split(")", 1)[1].strip("_") for l in labels])               # PB glomerulus / wedge suffix ('' if absent)
    idx = np.arange(len(labels))
    if core:
        keep = np.isin(types, CORE_TYPES); idx = idx[keep]
        W = W[np.ix_(idx, idx)]; labels = [labels[i] for i in idx]; types = types[idx]; wedges = wedges[idx]
    return dict(W=W, labels=labels, types=types, wedges=wedges, idx=idx)

def type_block_matrix(W, types, order=CORE_TYPES):
    """Signed total weight from type a to type b (rows presynaptic)."""
    B = np.zeros((len(order), len(order)))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            B[i, j] = W[np.ix_(types == a, types == b)].sum()
    return B

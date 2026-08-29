"""Shared helpers for benchmarks: synthetic manifolds and a GPT-2 probe."""
from __future__ import annotations
import json, os
import numpy as np

CORPUS = "/home/god/whitebox_experiments_clone/data/general_corpus.json"
RESULTS = os.path.join(os.path.dirname(__file__), "results")


def embed(X2, d=30, noise=0.05, seed=0):
    """Isometrically embed a low-dim cloud in R^d by a random orthonormal map, add noise."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Xd = np.zeros((len(X2), d)); Xd[:, :X2.shape[1]] = X2
    return Xd @ Q.T + noise * rng.standard_normal((len(X2), d))


def circle(n, seed=0):
    t = np.random.default_rng(seed).uniform(0, 2 * np.pi, n)
    return np.c_[np.cos(t), np.sin(t)]


def two_circles(n, seed=0):
    a = circle(n // 2, seed); b = circle(n - n // 2, seed + 1) + [3.0, 0]
    return np.vstack([a, b])


def torus(n, R=2.0, r=0.7, seed=0):
    rng = np.random.default_rng(seed); u = rng.uniform(0, 2 * np.pi, n); v = rng.uniform(0, 2 * np.pi, n)
    return np.c_[(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)]


def sphere(n, seed=0):
    X = np.random.default_rng(seed).standard_normal((n, 3)); return X / np.linalg.norm(X, axis=1, keepdims=True)


def blob(n, seed=0):
    return np.random.default_rng(seed).standard_normal((n, 3))


def clusters(n, k=5, seed=0):
    rng = np.random.default_rng(seed); c = rng.standard_normal((k, 3)) * 4
    return c[rng.integers(k, size=n)] + 0.3 * rng.standard_normal((n, 3))


SYNTHETIC = {"circle": circle, "two_circles": two_circles, "torus": torus, "sphere": sphere,
             "blob": blob, "clusters": clusters}


def probe_sentences(n, seed=0, offset=2000):
    allg = json.load(open(CORPUS)); rng = np.random.default_rng(seed); perm = rng.permutation(len(allg))
    return [allg[i] for i in perm[offset:offset + n]]


def gpt2_activations(name, sents, layers, device="cuda", max_len=40, batch=32, dtype=None):
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from whitebox.activations import hf_hidden_states
    tok = GPT2TokenizerFast.from_pretrained(name); tok.pad_token = tok.eos_token
    kw = {"dtype": dtype} if dtype is not None else {}
    model = GPT2LMHeadModel.from_pretrained(name, **kw).to(device).eval()
    e = tok(sents, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    acts = hf_hidden_states(model, e["input_ids"], e["attention_mask"], layers, batch_size=batch, device=device)
    return model, acts

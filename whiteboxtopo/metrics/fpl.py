"""Definition 5 — Feature Persistence Lifetime.

Two operationalisations existed in the original code; both are kept because
they answer different questions and only one carries a validated result:

* ``top_k_lifetime`` — mean lifetime of the k longest H1 bars of ONE diagram
  (expB ``top3_fpl``). Averaged over post-crystallisation epochs per class,
  this is the quantity with the P1 result: Spearman rho = -0.84 (p=0.002)
  against per-class CIFAR-10 test accuracy (longer-lived loops => harder class).

* ``feature_persistence_runs`` — the paper's run-length definition over a
  film: how many consecutive epochs a layer holds an above-median count of
  long-lived features (original ``feature_persistence_lifetimes``). No
  validated downstream result; kept for completeness.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np


def top_k_lifetime(D, k: int = 3) -> float:
    D = np.asarray(D, dtype=np.float64)
    if D.size == 0:
        return 0.0
    lt = np.sort(D[:, 1] - D[:, 0])[::-1][:k]
    return float(np.mean(lt)) if lt.size else 0.0


def feature_persistence_runs(diagrams: Sequence[np.ndarray]) -> np.ndarray:
    """Run lengths (in epochs) of above-median long-feature counts. Returns
    array([0]) if there is no run."""
    medians = []
    for D in diagrams:
        D = np.asarray(D, dtype=np.float64)
        medians.append(float(np.median(D[:, 1] - D[:, 0])) if D.size else 0.0)
    if not medians:
        return np.array([0])
    pos = [m for m in medians if m > 0]
    global_med = float(np.median(pos)) if pos else 0.0
    above = []
    for D in diagrams:
        D = np.asarray(D, dtype=np.float64)
        above.append(int(np.sum((D[:, 1] - D[:, 0]) > global_med)) if D.size else 0)
    above = np.asarray(above)
    thresh = np.median(above)
    runs, cur = [], 0
    for v in above:
        if v > thresh:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return np.asarray(runs) if runs else np.array([0])

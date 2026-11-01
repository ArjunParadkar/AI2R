"""Definition 4 — Topological Complexity TC^l(t) = sum_i (d_i - b_i)^2.

Note: this is the squared-lifetime total persistence, i.e. the squared
2-Wasserstein distance of the diagram to the empty diagram. It is the
quantity behind the memorization result (clean vs 100%-noise: ~1000x on
ResNet-18, ~2900x on GPT-2 124M). Related but distinct from persistent
entropy (Rucco et al. 2016), which is scale-free; see ``persistent_entropy``.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np


def topological_complexity(D) -> float:
    D = np.asarray(D, dtype=np.float64)
    if D.size == 0:
        return 0.0
    return float(np.sum((D[:, 1] - D[:, 0]) ** 2))


def tc_trajectory(diagrams: Sequence[np.ndarray]) -> np.ndarray:
    return np.asarray([topological_complexity(D) for D in diagrams])


def persistent_entropy(D) -> float:
    """Shannon entropy of the normalised lifetimes (Rucco, Castiglione,
    Merelli, Pettini 2016; Atienza et al. 2019). Scale-invariant companion to
    TC: TC says how much loop structure, PE says how evenly it is spread."""
    D = np.asarray(D, dtype=np.float64)
    if D.size == 0:
        return 0.0
    lt = D[:, 1] - D[:, 0]
    lt = lt[lt > 0]
    if lt.size == 0:
        return 0.0
    p = lt / lt.sum()
    return float(-np.sum(p * np.log(p)))

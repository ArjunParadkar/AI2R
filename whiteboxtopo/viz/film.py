"""Minimal plotting for diagrams and metric trajectories. 300 dpi on save."""
from __future__ import annotations
import numpy as np


def plot_diagram(D, ax=None, label=None, **kw):
    import matplotlib.pyplot as plt
    D = np.asarray(D, dtype=np.float64)
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    if D.size:
        m = float(D.max()) * 1.05
        ax.plot([0, m], [0, m], "k--", lw=0.8)
        ax.scatter(D[:, 0], D[:, 1], s=10, label=label, **kw)
    ax.set_xlabel("birth"); ax.set_ylabel("death"); ax.set_aspect("equal")
    if label:
        ax.legend()
    return ax


def plot_trajectories(epochs, series: dict, ylabel="", ax=None, title=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    for name, y in series.items():
        ax.plot(epochs, y, marker="o", ms=3, label=str(name))
    ax.set_xlabel("epoch"); ax.set_ylabel(ylabel); ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax


def savefig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")

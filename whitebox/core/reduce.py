"""Dimensionality reduction fitted ONCE (epoch 0) and held fixed across a film.

Holding the projection fixed is what makes diagrams at different epochs
comparable: the 2-Wasserstein distance between them (SV) would otherwise
mix representational change with projection change.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA


@dataclass
class Reducer:
    """A fitted linear projection. ``transform`` is deterministic and cheap."""
    pca: PCA
    n_components: int

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(np.asarray(X, dtype=np.float64))

    @property
    def explained_variance_ratio(self) -> float:
        return float(np.sum(self.pca.explained_variance_ratio_))


def fit_reducer(X: np.ndarray, n_components: int = 30, random_state: int = 0) -> Reducer:
    """Fit PCA on the epoch-0 activations of a probe set.

    n_components is clipped to min(n_components, n_samples-1, n_features),
    matching the original ``whitebox.fit_pca``.
    """
    X = np.asarray(X, dtype=np.float64)
    k = int(min(n_components, X.shape[0] - 1, X.shape[1]))
    pca = PCA(n_components=k, random_state=random_state).fit(X)
    return Reducer(pca=pca, n_components=k)

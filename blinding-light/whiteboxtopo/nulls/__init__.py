"""Every positive result must survive these."""
import numpy as np

def feature_shuffle(X, seed=0):
    """Shuffle each activation coordinate independently across the probe.

    Preserves every marginal exactly and destroys all joint structure -- so any topology,
    geodesic closure or coordinate that survives this was never structure to begin with.
    """
    rng = np.random.default_rng(seed)
    Y = np.array(X, copy=True)
    for j in range(Y.shape[1]):
        rng.shuffle(Y[:, j])
    return Y

def label_permute(y, seed=0):
    """Permute the labels, keeping the cloud fixed."""
    rng = np.random.default_rng(seed)
    return rng.permutation(np.asarray(y))

def shuffle_predictor(v, seed=0):
    """Shuffle a scalar predictor, for the P5 null."""
    rng = np.random.default_rng(seed)
    return rng.permutation(np.asarray(v))

__all__ = ['feature_shuffle', 'label_permute', 'shuffle_predictor']

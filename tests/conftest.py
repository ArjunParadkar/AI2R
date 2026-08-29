import numpy as np
import pytest


def circle(n=200, r=1.0, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    X = np.c_[r * np.cos(t), r * np.sin(t)] + noise * rng.standard_normal((n, 2))
    return X


def blob(n=200, d=2, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d))


@pytest.fixture
def circle_cloud():
    return circle()


@pytest.fixture
def blob_cloud():
    return blob()

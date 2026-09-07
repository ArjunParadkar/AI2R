"""S3 tests: MP on Gaussian noise -> no outliers (nominal rate) over 100 trials; planted spikes recovered."""
import numpy as np
from whiteboxtopo import nulls


def test_mp_gaussian_false_positives_at_nominal_rate():
    hits = 0
    for t in range(100):
        E, _ = nulls.planted("gaussian", (1000, 100), seed=t)
        ev = np.linalg.eigvalsh(np.cov(E.T))
        hits += nulls.marchenko_pastur(ev, 0.1, n=1000, q=0.99).n_outliers > 0
    assert hits <= 4          # nominal 1 % per trial; binomial(100, 0.01) P(>4) ~ 0.003


def test_mp_recovers_planted_spikes():
    E, truth = nulls.planted("spikes", (2000, 200), k=5, seed=1)
    ev = np.linalg.eigvalsh(np.cov(E.T))
    assert nulls.marchenko_pastur(ev, 0.1, n=2000).n_outliers == truth["n_outliers"]


def test_feature_shuffle_keeps_marginals():
    E, _ = nulls.planted("lowrank", (500, 20), k=3, seed=0)
    S = nulls.feature_shuffle(E, seed=1)
    assert np.allclose(np.sort(S, 0), np.sort(E, 0))
    assert np.abs(np.corrcoef(S.T)[np.triu_indices(20, 1)]).mean() < 0.1


def test_label_permute_is_permutation():
    y = np.arange(10); z = nulls.label_permute(y, seed=3)
    assert sorted(z.tolist()) == list(range(10)) and not np.array_equal(y, z)


def test_planted_circle_has_loop():
    from whiteboxtopo.topology import rips
    X, truth = nulls.planted("circle", 100, seed=0, noise=0.02)
    D = np.asarray(rips(X, maxdim=1).dgms[1]).reshape(-1, 2)
    life = D[:, 1] - D[:, 0]
    assert truth["b1"] == 1 and (life > 1.0).sum() == 1

"""Planted-truth tests for the breaking score b_z (SPEC3 S5)."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from whiteboxtopo.symmetry_breaking import breaking_score

torch.manual_seed(0)
N, S, D, K = 300, 3, 16, 7           # N > 256 so that batching is exercised
X = torch.randn(N, S, D, dtype=torch.float64)
W1 = torch.randn(2 * D, 32, dtype=torch.float64); W2 = torch.randn(32, K, dtype=torch.float64)
Wz = torch.randn(D, 32, dtype=torch.float64)


def test_ignores_block_gives_zero():
    f = lambda s: torch.tanh(s[:2].reshape(-1) @ W1) @ W2           # never reads slot 2
    r = breaking_score(f, X, nuisance=2)
    assert r['b_z'] < 1e-6 and r['b_z_rom'] < 1e-6 and r['n'] == N


def test_depends_only_on_block_gives_one():
    f = lambda s: torch.tanh(s[2] @ Wz) @ W2
    r = breaking_score(f, X, nuisance=2)
    assert abs(r['b_z'] - 1) < 1e-9 and abs(r['b_z_rom'] - 1) < 1e-9
    assert breaking_score(f, X, nuisance=0)['b_z'] < 1e-6


def test_linear_map_exact_ratio():
    A = [torch.randn(D, K, dtype=torch.float64) * c for c in (1.0, 2.0, 0.5)]
    f = lambda s: sum(s[i] @ A[i] for i in range(S))
    n2 = torch.stack([(a ** 2).sum() for a in A])
    for z in range(S):
        r = breaking_score(f, X, nuisance=z, return_metric=True)
        exact = float(n2[z] / n2.sum())
        assert abs(r['b_z'] - exact) < 1e-10 and abs(r['b_z_rom'] - exact) < 1e-10
        assert abs(sum(r['slot_share']) - 1) < 1e-12
        G = r['metric']; blk = G[z * D:(z + 1) * D, z * D:(z + 1) * D]
        assert abs(float(torch.trace(blk) / torch.trace(G)) - exact) < 1e-10      # hint share of trace E[g] == ratio of means


def test_mean_of_ratios_differs_from_ratio_of_means():
    # input-dependent gain on the nuisance block: J_z = g(x) * B, J_rest = A.  The two averages are different and both known.
    A = torch.randn(D, K, dtype=torch.float64); B = torch.randn(D, K, dtype=torch.float64)
    gain = torch.linspace(0, 3, N, dtype=torch.float64)
    Xg = X.clone(); Xg[:, 1, 0] = gain                                # slot 1, coord 0 carries the gain, and gets no other use
    A1 = torch.zeros(D, K, dtype=torch.float64)

    def f(s):
        return s[0] @ A + s[1] @ A1 + (s[1, 0].detach() * s[2]) @ B
    a2, b2 = (A ** 2).sum(), (B ** 2).sum()
    r = breaking_score(f, Xg, nuisance=2)
    mor = float((gain ** 2 * b2 / (a2 + gain ** 2 * b2)).mean()); rom = float((gain ** 2).mean() * b2 / (a2 + (gain ** 2).mean() * b2))
    assert abs(r['b_z'] - mor) < 1e-10 and abs(r['b_z_rom'] - rom) < 1e-10 and abs(mor - rom) > 1e-3


def test_tuple_targets_per_layer():
    f = lambda s: (s[2] @ Wz, torch.tanh(s[:2].reshape(-1) @ W1) @ W2)
    r = breaking_score(f, X, nuisance=2, names=['only_z', 'ignores_z'])
    assert r[0]['name'] == 'only_z' and abs(r[0]['b_z'] - 1) < 1e-9 and r[1]['b_z'] < 1e-6


def test_zero_jacobian_counts_as_zero():
    f = lambda s: torch.zeros(K, dtype=torch.float64) * s.sum()
    r = breaking_score(f, X[:10], nuisance=2)
    assert r['b_z'] == 0 and r['n_zero'] == 10

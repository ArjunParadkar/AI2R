"""SPEC5 block T planted check: the finite g-length on maps whose answer is known."""
import math, torch
from whiteboxtopo.finite_tower import finite_g_length, finite_q


def test_linear_map_equals_secant():
    torch.manual_seed(0); W = torch.randn(7, 5, dtype=torch.float64); h0, h1 = torch.randn(40, 5, dtype=torch.float64), torch.randn(40, 5, dtype=torch.float64)
    L = finite_g_length(lambda h: h @ W.T, h0, h1); assert torch.allclose(L, ((h1 - h0) @ W.T).norm(dim=1), atol=1e-9)


def test_arc_length_on_circle():
    th = torch.tensor([[0.3], [1.0], [2.5]], dtype=torch.float64); h0 = torch.zeros_like(th)
    L = finite_g_length(lambda h: torch.cat([torch.cos(h), torch.sin(h)], 1), h0, th)
    assert torch.all((th[:, 0] - L) / th[:, 0] < 0.01) and torch.all(L <= th[:, 0] + 1e-12)          # under-measures an arc, by < 1%
    chord = 2 * torch.sin(th[:, 0] / 2); assert torch.all(L >= chord - 1e-12)                       # and is never shorter than the secant


def test_null_direction_has_zero_length():
    psi = lambda h: h[:, :1] * 3.0; h0 = torch.randn(10, 4); h1 = h0.clone(); h1[:, 1:] += torch.randn(10, 3)
    assert float(finite_g_length(psi, h0, h1).max()) < 1e-6
    assert finite_q(psi, h0, h1, torch.randn(10, 4)) < 1e-6

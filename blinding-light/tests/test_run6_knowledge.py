"""SPEC6 K.1 planted check: an exactly equivariant map has kappa = 1 and eps = 0; a random map has kappa ~ 1/n_classes."""
import torch
from whiteboxtopo.knowledge import orbit_stats, resnet_group, z113_group, golan_group


def test_equivariant_map():
    torch.manual_seed(0); x = torch.randn(50, 3, 32, 32)
    f = lambda z: torch.stack([z.mean((1, 2, 3)), z.abs().mean((1, 2, 3)), (z ** 2).mean((1, 2, 3))], 1)     # shift- and flip-invariant
    s = orbit_stats(f, x, resnet_group()); assert s['kappa'].min() == 1.0 and s['eps'].max() < 1e-5 and s['kappa_s'].min() > 0.9999


def test_random_map_is_incoherent():
    torch.manual_seed(0); x = torch.randn(400, 3, 32, 32); W = torch.randn(3072, 10)
    f = lambda z: z.reshape(len(z), -1) @ W; s = orbit_stats(f, x, resnet_group()); assert 0.03 < s['kappa'].mean() < 0.25 and s['eps'].mean() > 0.5


def test_z113_group_relabels_the_answer():
    P = 113; x = torch.stack([torch.arange(0, 50), torch.arange(10, 60), torch.full((50,), P), torch.full((50,), P + 1)], 1)
    f = lambda t: torch.nn.functional.one_hot((t[:, 0] + t[:, 1]) % P, P).float() * 5     # the exact rule
    s = orbit_stats(f, x, z113_group(u=17)); assert s['kappa'].min() == 1.0 and s['eps'].max() < 1e-6
    g = lambda t: torch.nn.functional.one_hot((t[:, 0] * 3 + t[:, 1]) % P, P).float() * 5  # wrong rule: translation is not a symmetry
    s2 = orbit_stats(g, x, z113_group(u=17)); assert s2['kappa'].mean() < 0.5


def test_golan_has_72():
    assert len(golan_group()) == 72

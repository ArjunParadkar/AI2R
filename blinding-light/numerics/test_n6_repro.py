"""N6 reproducibility — T-N6-01..03."""
import numpy as np, torch
from whiteboxtopo.numerics import repro, jacobian
from common import tanh_mlp, mlp_weights


def _n1_payload(seed, out):
    repro.set_determinism(seed)
    Ws = mlp_weights([6, 12, 12, 4], seed); f = tanh_mlp(Ws); g = torch.Generator().manual_seed(seed)
    X = torch.randn(10, 6, generator=g, dtype=torch.float64)
    J = torch.stack([torch.autograd.functional.jacobian(f, x) for x in X])
    return {'J': J.numpy()}


def test_n6_01_bitwise(cell, tmp_path):
    a, b, c = _n1_payload(7, 0), _n1_payload(7, 0), _n1_payload(8, 0)
    ha, hb, hc = (repro.payload_hash(p) for p in (a, b, c))
    cell('T-N6-01', {'seed': 7}, 7, {'same': ha == hb, 'diff': ha != hc}, {'same seed identical': ha == hb, 'other seed differs': ha != hc})


def test_n6_02_records(cell, tmp_path):
    cfg = {'a': 1, 'b': [1.5, 2], 'c': {'x': 1, 'y': 2}}; cfg2 = {'c': {'y': 2, 'x': 1}, 'b': [1.5, 2], 'a': 1}
    rec = repro.save_result(tmp_path / 'r.json', 'x', {'v': np.arange(3)}, cfg, 3)
    keys = {'seed', 'config', 'config_hash', 'versions', 'determinism'}
    cell('T-N6-02', cfg, 3, {'keys': sorted(rec)}, {
        'fields present': keys <= set(rec), 'hash key-order invariant': repro.config_hash(cfg) == repro.config_hash(cfg2),
        'hash changes with value': repro.config_hash(cfg) != repro.config_hash({**cfg, 'a': 2}),
        'hash changes with nested value': repro.config_hash(cfg) != repro.config_hash({**cfg, 'c': {'x': 1, 'y': 3}})})


def test_n6_03_flags(cell):
    fl = repro.set_determinism(0)
    cell('T-N6-03', {}, 0, fl, {'deterministic_algorithms': fl['deterministic_algorithms'] is True,
                                'flash off': fl['flash_sdp'] is False, 'mem_efficient off': fl['mem_efficient_sdp'] is False,
                                'math on': fl['math_sdp'] is True})

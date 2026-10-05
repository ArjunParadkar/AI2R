"""Shared fixtures: CPU-only threads, TOLERANCES.md freeze check, per-cell result records (seed + config hash)."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '6'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '6')
os.environ.setdefault('MKL_NUM_THREADS', '6'); os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, json, hashlib, shutil
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
RESULTS = HERE / 'results'
CELLS = []


def _js(o):
    import numpy as np
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def pytest_sessionstart(session):
    frozen = (HERE / 'TOLERANCES.sha256').read_text().split()[0]
    now = hashlib.sha256((HERE / 'TOLERANCES.md').read_bytes()).hexdigest()
    if frozen != now:
        raise pytest.UsageError('TOLERANCES.md changed after it was frozen: thresholds never change after the fact')
    if not os.environ.get('NUMERICS_KEEP_RESULTS'):
        shutil.rmtree(RESULTS, ignore_errors=True)
    RESULTS.mkdir(exist_ok=True)


@pytest.fixture
def cell():
    """cell(id, cfg, seed, measured, conds) -> asserts all conds; saves the record with seed + config hash."""
    from whiteboxtopo.numerics.repro import save_result
    def run(cid, cfg, seed, measured, conds):
        passed = all(bool(v) for v in conds.values())
        rec = save_result(RESULTS / f'{cid}.json', cid, {'measured': measured, 'conditions': {k: bool(v) for k, v in conds.items()},
                                                          'passed': passed}, cfg, seed)
        CELLS.append({'id': cid, 'passed': passed, 'measured': measured,
                      'conditions': {k: bool(v) for k, v in conds.items()}})
        bad = [k for k, v in conds.items() if not v]
        assert passed, f'{cid} FAILED conditions {bad}; measured={measured}'
    return run


def pytest_sessionfinish(session, exitstatus):
    prev = RESULTS / 'summary.json'
    old = json.loads(prev.read_text()) if prev.exists() and os.environ.get('NUMERICS_KEEP_RESULTS') else {}
    for c in CELLS: old[c['id']] = c
    prev.write_text(json.dumps(old, indent=1, sort_keys=True, default=_js))

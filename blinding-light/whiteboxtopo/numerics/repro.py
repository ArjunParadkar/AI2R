"""N6: reproducibility.  Every saved result carries seed + config + config hash + versions + flags."""
from __future__ import annotations
import hashlib, json, os, platform, random, time
from pathlib import Path
import numpy as np


def set_determinism(seed: int, threads: int | None = None) -> dict:
    """Seed python/numpy/torch, force deterministic algorithms, restrict SDPA to the MATH kernel.
    (EFFICIENT_ATTENTION backward is wrong on the RTX 5070 here; MATH-only is the project rule.)"""
    import torch
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for fn in ('enable_flash_sdp', 'enable_mem_efficient_sdp'):
        try: getattr(torch.backends.cuda, fn)(False)
        except Exception: pass
    try: torch.backends.cuda.enable_math_sdp(True)
    except Exception: pass
    if threads: torch.set_num_threads(threads)
    return flags()


def flags() -> dict:
    import torch
    out = {'deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()),
           'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
           'cudnn_benchmark': bool(torch.backends.cudnn.benchmark)}
    for k, fn in (('flash_sdp', 'flash_sdp_enabled'), ('mem_efficient_sdp', 'mem_efficient_sdp_enabled'),
                  ('math_sdp', 'math_sdp_enabled')):
        try: out[k] = bool(getattr(torch.backends.cuda, fn)())
        except Exception: out[k] = None
    return out


def _canon(o):
    if isinstance(o, dict): return {str(k): _canon(o[k]) for k in sorted(o, key=str)}
    if isinstance(o, (list, tuple)): return [_canon(v) for v in o]
    if isinstance(o, np.ndarray): return {'__nd__': o.dtype.str, 'shape': list(o.shape),
                                          'sha': hashlib.sha256(np.ascontiguousarray(o).tobytes()).hexdigest()}
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if hasattr(o, 'detach'): return _canon(o.detach().cpu().numpy())
    if isinstance(o, float): return repr(o)          # exact round-trip, nan/inf safe
    if isinstance(o, (int, str, bool)) or o is None: return o
    return repr(o)


def config_hash(cfg: dict) -> str:
    """sha256 of the canonical JSON of cfg (key-order invariant, arrays hashed by bytes)."""
    return hashlib.sha256(json.dumps(_canon(cfg), sort_keys=True).encode()).hexdigest()


def payload_hash(result) -> str:
    return hashlib.sha256(json.dumps(_canon(result), sort_keys=True).encode()).hexdigest()


def versions() -> dict:
    import torch, scipy
    out = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
           'torch': torch.__version__}
    for m in ('ripser', 'persim'):
        try: out[m] = __import__(m).__version__
        except Exception: out[m] = None
    return out


def make_record(name: str, result, cfg: dict, seed: int) -> dict:
    return {'name': name, 'seed': int(seed), 'config': _canon(cfg), 'config_hash': config_hash(cfg),
            'versions': versions(), 'determinism': flags(), 'result_sha256': payload_hash(result),
            'result': _canon(result)}


def save_result(path, name: str, result, cfg: dict, seed: int, timestamp: bool = True) -> dict:
    rec = make_record(name, result, cfg, seed)
    if timestamp: rec['saved_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')   # excluded from result_sha256
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rec, indent=1, sort_keys=True))
    return rec

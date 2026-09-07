import numpy as np


def minimal_ids_fast(valid_ids):
    """Minimal valid subsets: v is minimal iff no *minimal* valid m is a proper subset of v
    (a valid proper subset implies a minimal one). Processed in popcount order, vectorised."""
    v = np.unique(np.asarray(valid_ids, dtype=np.int64))
    if len(v) == 0:
        return v
    pc = np.array([bin(int(x)).count("1") for x in v]); order = np.argsort(pc, kind="stable"); v = v[order]; pc = pc[order]
    minimal = []
    M = np.zeros(0, dtype=np.int64)
    for c in np.unique(pc):
        batch = v[pc == c]
        if len(M):
            # m ⊂ v (m has fewer bits, so proper). Chunked so the (batch x M) broadcast stays <= 2^25 int64
            # (256 MB): the un-chunked version peaked at 21 GB on OR4 d3w8 (10M valid) and was OOM-killed (AMENDMENTS #9).
            step = max(1, (1 << 25) // len(M)); keep = np.empty(len(batch), dtype=bool)
            for s in range(0, len(batch), step):
                b = batch[s:s + step]
                keep[s:s + step] = ~((b[:, None] & M[None, :]) == M[None, :]).any(1)
        else:
            keep = np.ones(len(batch), bool)
        minimal.append(batch[keep]); M = np.concatenate([M, batch[keep]])
    return np.sort(np.concatenate(minimal))

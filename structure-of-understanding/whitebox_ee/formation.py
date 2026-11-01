"""Formation: the concept graph across checkpoints (master prompt §3.1, K14).
Run the same discovery (same proposer configuration and thresholds) at every
checkpoint; report per-checkpoint depth scores, principles, principle births and
grounding events. Lesson 1.7: formation claims need every checkpoint."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .info import normalised_vi


def formation(per_ckpt: Sequence[dict], tau: float, eps_merge: float, n_eff: int):
    """per_ckpt: list of dicts with keys 'ckpt', 'G', 'g' (generality dict),
    'scores' (depth_scores output), optional 'heldout_acc' (dict behaviour -> float).
    Node identity across checkpoints is partition equivalence (normalised VI)."""
    tracks: List[dict] = []          # each: {'values': ..., 'first': ckpt index, 'principle_from': idx}
    rows = []
    for i, d in enumerate(per_ckpt):
        G, g = d["G"], d["g"]
        prin = [c for c in G.concepts() if g.get(c, 0) >= tau]
        for c in G.concepts():
            v = G.nodes[c].values
            hit = None
            for t in tracks:
                if normalised_vi(v, t["values"], n_eff) <= eps_merge:
                    hit = t
                    break
            if hit is None:
                hit = {"values": v, "first": i, "principle_from": None, "label": G.nodes[c].meta.get("label")}
                tracks.append(hit)
            if c in prin and hit["principle_from"] is None:
                hit["principle_from"] = i
        lam = {b: float(np.mean(s["lambda"] >= tau)) for b, s in d["scores"].items()}
        lbar = {b: float(np.mean(s["lambda"])) for b, s in d["scores"].items()}
        rows.append({"ckpt": d["ckpt"], "Lambda": lam, "lambda_bar": lbar, "n_principles": len(prin),
                     "heldout_acc": d.get("heldout_acc")})
    events = {}
    bs = rows[0]["Lambda"].keys()
    for b in bs:
        L = np.array([r["Lambda"][b] for r in rows])
        jumps = np.diff(L)
        events[b] = {"grounding_event": int(np.argmax(jumps) + 1) if len(jumps) else None,
                     "jump": float(jumps.max()) if len(jumps) else 0.0}
        if rows[0].get("heldout_acc"):
            A = np.array([r["heldout_acc"][b] for r in rows])
            ja = np.diff(A)
            events[b]["heldout_jump"] = int(np.argmax(ja) + 1)
    births = [{"label": t["label"], "first_seen": t["first"], "principle_from": t["principle_from"]} for t in tracks]
    return {"per_checkpoint": rows, "events": events, "births": births}

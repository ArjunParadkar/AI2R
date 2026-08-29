"""Calibrated, claim-bearing signals. Each entry point ships only with a
VALIDATION.md entry (Phase 4).

    memorization_signal           Phase 2  (per-family calibrated memorisation score)
    representation_correspondence Phase 3  (Sub-H4; RTD-style cross-space comparison)

`failure_neighborhoods` (paper P4) was evaluated in Phase 4 and NOT shipped:
under a label-permutation null the effect is not significant and per-sample
it is dominated by softmax confidence (VALIDATION.md).
"""
from .memorization import memorization_signal, memorization_features, MemorizationSignal, Calibration
from .correspondence import representation_correspondence, correspondence, rtd, Correspondence


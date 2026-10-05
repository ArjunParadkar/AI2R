"""Phase-1 numerics N1-N6 of WHITEBOX v2 (Arjun Paradkar and Reva Paradkar).

N1 jacobian   exact / sketched / gradient-free Jacobians, Hutchinson traces, Fisher-Rao pullback
N2 geodesic   g-weighted kNN graph geodesics
N3 symmetry   delta-symmetry search (Lie algebra, finite groups, calibration)
N4 killing    meshfree least-squares Killing solver
N5 homology   persistent homology with certified bars
N6 repro      seeds, config hashes, deterministic flags, result files
CPU only; nothing here touches CUDA.
"""
from . import repro, jacobian, geodesic, symmetry, killing, homology   # noqa: F401

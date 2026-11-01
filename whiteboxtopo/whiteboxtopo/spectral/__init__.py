"""whiteboxtopo.spectral -- spectral geometry of representation manifolds (SPEC 2.10, 6.5, 6.8).

    from whiteboxtopo.spectral import laplacian, diffusion_coords, out_of_manifold
    L, W = laplacian(cloud, k=15)                 # I - D^-1/2 W D^-1/2, Gaussian k-NN weights
    Psi, lam, phi = diffusion_coords(L, t=1)      # Psi_t = ((1-lam_i)^t phi_i)_i
    res = out_of_manifold(cloud, h_new)           # P4 score + 99th-percentile threshold
    null = random_projection_null(cloud, h_new, d=10)   # SPEC 6.5 null, same shape

Instruments, not signals. A small ``d_M`` is not a certificate that the model
knows the input, and the spectral gap is not a claim about "crystallization" -
each function's docstring states the null it was validated against and where
it stops being meaningful. Nothing here is gauge-invariant on its own: the
k-NN graph is permutation-invariant but per-unit scalings of the cloud change
the distances, so standardise the cloud first and record the gauge fixing.
"""
from .laplacian import (laplacian, diffusion_coords, nystrom, kernel_params, smallest_eigs,
                        default_eps, DENSE_MAX)
from .oom import out_of_manifold, random_projection_null, OOMResult
from .flow import eigenflow, EigenflowResult, SMALL_EIG_THRESHOLD

__all__ = ["laplacian", "diffusion_coords", "nystrom", "kernel_params", "smallest_eigs", "default_eps",
           "out_of_manifold", "random_projection_null", "OOMResult",
           "eigenflow", "EigenflowResult", "SMALL_EIG_THRESHOLD", "DENSE_MAX"]

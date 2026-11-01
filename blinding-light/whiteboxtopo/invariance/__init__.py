"""The path product: the exactly gauge-invariant linear map a ReLU network applies locally."""
import numpy as np


def path_product(J):
    """A(x) = W_L D_{L-1} ... D_1 W_1.

    For a piecewise-linear network this is exactly the Jacobian of the remaining map at x: on
    the activation cell containing x the network IS the single linear map A(x), and A is
    invariant to any rescaling gauge W_i -> c W_i, D -> D/c.  We therefore read it off the
    Jacobian we already computed rather than re-multiplying the weights.
    """
    return J


def check_invariance(psi, x, scale=2.0, eps=1e-4):
    """Confirm the path product is unchanged by a positive rescaling of an intermediate layer,
    which is the gauge freedom of a ReLU network."""
    raise NotImplementedError("model-specific; the Z113 and CNN heads are checked in analysis/")

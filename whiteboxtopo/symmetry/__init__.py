"""whiteboxtopo.symmetry — the gauge group G of a network (handbook S1).

    from whiteboxtopo import symmetry
    g = symmetry.random_element(model, seed=0, kind="combined")   # perm | scale | rot | combined
    model_g = symmetry.apply(model, g)                              # same function, renamed
    symmetry.output_deviation(model, g, X)                          # ~1e-12
    symmetry.hidden(model, X)                                       # fiber dim vs known dim

Backends: ReLU MLPs (numpy ``.Ws/.bs`` or torch ``nn.Sequential``),
``ToyTransformer`` (F7), HF ``GPT2LMHeadModel`` (F11). ``describe(model)`` lists
the layers G acts on and the continuous dimension of the known group.
"""
from .group import (_backend, GaugeElement, KINDS, describe, permutations, scalings, rotations, random_element,
                    uniform_scaling, apply, inverse_on_units, output_deviation)
from .hidden import hidden, HiddenSymmetryReport, scaling_generators, realisation_jacobian
from .transformer import ToyTransformer

__all__ = ["_backend", "GaugeElement", "KINDS", "describe", "permutations", "scalings", "rotations", "random_element",
           "uniform_scaling", "apply", "inverse_on_units", "output_deviation", "hidden", "HiddenSymmetryReport",
           "scaling_generators", "realisation_jacobian", "ToyTransformer"]

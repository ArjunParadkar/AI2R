"""Calibrated, claim-bearing signals. Each entry point ships only with a
VALIDATION.md entry (Phase 4). Phase 0 = stubs that name their phase.

    memorization_signal          Phase 2  (TC/beta1 collapse; the robust result)
    failure_neighborhoods        Phase 4  (P4; layer-3 TC of misclassified inputs)
    representation_correspondence Phase 3 (Sub-H4; RTD-style cross-space comparison)
"""


def _not_yet(name, phase):
    def f(*a, **k):
        raise NotImplementedError(f"whitebox.{name} is delivered in Phase {phase}.")
    f.__name__ = name
    return f


memorization_signal = _not_yet("memorization_signal", 2)
representation_correspondence = _not_yet("representation_correspondence", 3)
failure_neighborhoods = _not_yet("failure_neighborhoods", 4)

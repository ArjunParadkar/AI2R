"""whitebox.darkmatter -- is the SAE reconstruction error structured, or is it isotropic
noise? Two independent instruments (instrument only; verdict discipline below):

    from whitebox.darkmatter import residual_structure
    rep = residual_structure(errors)                 # E = x - SAE(x), (n_tokens, d)
    rep = residual_structure(acts=x, recon=sae(x))    # or pass the pair directly
    rep.n_outliers, rep.structured_fraction, rep.spectral_test_valid   # Test 1 (MP/TW null)
    rep.tc_ratio, rep.tc_ratio_ci, rep.gaussian_control_tc_ratio        # Test 2 (topology null)
    rep.verdict   # "neither" / "spectrally_structured" / "topologically_structured" / "both"

Ported from whitebox-v2 E6 (2026-08-30, GPT-2 small, gpt2-small-res-jb SAELens residual-
stream SAEs, blocks.{2,6,10}.hook_resid_pre, 65k OpenWebText tokens/layer -- see
benchmarks/DARKMATTER.md and E6/SPEC.md + E6/FINDINGS.md in the whitebox-v2 experiment
tree). Test 3 (the logit-delta ridge probe) is NOT ported: it answers a different
question (is e linearly predictive of the model's own output), was never part of the
pre-registered H-E6 pass/fail, and needs a live model + SAE (torch, transformer_lens,
SAELens) that this library does not depend on. This module takes plain numpy arrays.

**Verdict discipline (mirrors whitebox.legibility's refused score):** the two
instruments ship; the claim "dark matter is structured" does not, because E6's own
pre-registered criterion (H-E6: n_outliers>10 AND tc_ratio>1.2, both with CI clearing
their bar) FAILED at every layer tested -- Test 1 passed cleanly (n_outliers 179-198 of
768) but Test 2 failed, and not narrowly: tc_ratio(E) was 0.11-0.46, i.e. the error
cloud is topologically SIMPLER than its own feature-shuffled null (and simpler still
than a covariance-matched Gaussian control, which correctly calibrates to ~1). Every
layer's verdict on the only battery run so far was "spectrally_structured" -- never
"topologically_structured" or "both". Read this as: real, robust spectral structure
(survives a matched-diagonal-Gaussian kill and a top-50-PC-of-x projection kill) that is
NOT accompanied by the kind of "extra topological complexity" the v1 instrument is built
to detect. See ``residual_structure``'s docstring for the full account and
benchmarks/DARKMATTER.md for the battery numbers.
"""
from .report import residual_structure, DarkMatterReport, SPECTRAL_N_OUTLIERS_MIN, TOPOLOGY_TC_RATIO_MIN, \
    TOPOLOGY_CI_LOWER_MIN
from .rmt import (mp_outliers, mp_edges, mp_density, tw_threshold, feature_shuffle, gaussian_matched_diag,
                  gaussian_matched_cov, project_out_top_pcs, block_bootstrap_outliers, subsample_outliers, MPResult)
from .topo import tc_features, tc_ratio_bootstrap

__all__ = [
    "residual_structure", "DarkMatterReport", "SPECTRAL_N_OUTLIERS_MIN", "TOPOLOGY_TC_RATIO_MIN",
    "TOPOLOGY_CI_LOWER_MIN",
    "mp_outliers", "mp_edges", "mp_density", "tw_threshold", "feature_shuffle", "gaussian_matched_diag",
    "gaussian_matched_cov", "project_out_top_pcs", "block_bootstrap_outliers", "subsample_outliers", "MPResult",
    "tc_features", "tc_ratio_bootstrap",
]

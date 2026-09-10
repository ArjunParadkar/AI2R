# Block Z — amendments

**#1 (2026-09-09 12:30) Z4b — scale gauge-fix + calibrated null added after PAUSE_5 §Z4 flagged both as missing.**
`tokai_teio_b.py`, same 380 pairs (same seed and sampling loop as `run_Z4_Z5.z4`; the `hungarian_raw` condition reproduces Z4's
numbers exactly: 0.118 / 0.129 / 0.28). Conditions: `scalefix_hungarian` — every hidden unit's incoming vector normalised
to unit length with the outgoing row rescaled inversely (an exact symmetry; asserted to leave circuit validity unchanged
on every pair), then Hungarian matching and linear interpolation in that frame; `scalefix_random_perm` — random unit
assignment in the same frame (the null). Results (`results/Z4b_summary.json`): valid-path fraction 0.095 (scale-fixed
Hungarian) vs 0.024 (null); median fraction valid 0.20 vs 0.14; paired Wilcoxon p = 7e-9, Hungarian better on 54 % of
pairs, null better on 32 %. The scale fix does NOT raise the fraction (PAUSE_5's stated expectation was wrong — recorded
there); matching is what matters, and even matched straight lines break in ~90 % of pairs.

**#2 (2026-09-10 08:50) Seiun Sky (Z1) was killed by the kernel OOM killer on her first model (19.5 GB peak, `journalctl --user -u
wb-gpu-queue`: "oom-kill"; the log had no row).** Cause, measured on `modadd_mlp128_l10.0_wd0.0_dr0.2_s0`: the pre-registered
Rips scale — the *final* cloud's median pairwise distance (10.5) — is larger than the whole diameter of the early checkpoints (their
median distances run 5.7 → 10.5 over training), so every early union of 300 points is a *complete* complex (4.5 M triangles each,
≈ 85 M simplices over 19 unions; a complete complex is contractible and carries no H1 anyway). Even 50-point clouds give 2.1 M
simplices at that scale. Fix, applied before any Z1 row was written: (a) every checkpoint cloud is centred and scaled to unit median
pairwise distance (shape across training, not size — the same normalisation the film needs); (b) the Rips scale is the scale at
which the final normalised cloud has the most H1 classes (its ripser diagram; quantile 0.1 of its distances if it has none) — the
most informative single scale, recorded per model as `scale`, `scale_quantile`, `beta1_final_max`; (c) the union complex is
*counted* first (`dionysus.fill_rips` per consecutive union) and N_SUB is lowered 150 → 100 → 75 → 50 until it is under a 2.5 M
simplex budget (`n_sub`, `budget_reduced` recorded; on the probe model 150 points at the informative scale is ≈ 0.3 M simplices);
(d) the same 150 probe examples are used at every checkpoint (the SPEC's independent per-cloud subsample would have made the
union track different points, defeating the zigzag). The film's per-checkpoint diagrams use the same normalised clouds and the same
scale, so the zigzag-vs-film comparison is unchanged in kind. The pre-registered absolute-median setting cannot be run on this
hardware and is not reported. The paddock is relaunched with `MemoryMax=20G` so a future overrun kills the horse, not the host.

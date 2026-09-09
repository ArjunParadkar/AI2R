# Block Z — amendments

**#1 (2026-09-09 12:30) Z4b — scale gauge-fix + calibrated null added after PAUSE_5 §Z4 flagged both as missing.**
`run_Z4b.py`, same 380 pairs (same seed and sampling loop as `run_Z4_Z5.z4`; the `hungarian_raw` condition reproduces Z4's
numbers exactly: 0.118 / 0.129 / 0.28). Conditions: `scalefix_hungarian` — every hidden unit's incoming vector normalised
to unit length with the outgoing row rescaled inversely (an exact symmetry; asserted to leave circuit validity unchanged
on every pair), then Hungarian matching and linear interpolation in that frame; `scalefix_random_perm` — random unit
assignment in the same frame (the null). Results (`results/Z4b_summary.json`): valid-path fraction 0.095 (scale-fixed
Hungarian) vs 0.024 (null); median fraction valid 0.20 vs 0.14; paired Wilcoxon p = 7e-9, Hungarian better on 54 % of
pairs, null better on 32 %. The scale fix does NOT raise the fraction (PAUSE_5's stated expectation was wrong — recorded
there); matching is what matters, and even matched straight lines break in ~90 % of pairs.

# FINDINGS (Run 5) P7 — the generator as a leading indicator

**Plain English.** On three fresh seeds swept at every checkpoint, the full 113-cycle is accepted at the checkpoint of the largest
test-accuracy jump on two seeds and one checkpoint before it on the third — never after it — at test accuracy 0.68, 0.60, 0.51. It is never
accepted at any of the 102 memoriser checkpoints. Symmetry restoration and hint-slot collapse hold as registered. Persistence fails by the
letter on one seed: 35 of 37 later checkpoints (0.946 against 0.95). Three of four cells pass. Neighbouring work: LieGG (Moskalev et al., 2022).

| cell | registered | measured | verdict |
|---|---|---|---|
| P7.a lead = jump − first acceptance ≥ 0 on 3/3; never on memorisers | | clean 9: jump 13, accept 13, lead 0 (test 0.68); clean 10: 7, 7, 0 (0.60); clean 11: 10, 9, **+1** (0.51); memorisers: 0 of 51 + 51 checkpoints | **PASS** |
| P7.b q(swap) < 0.05 within ±3 of the jump on 3/3; memorisers > 0.30 at every train ≥ 0.99 checkpoint | | min within ±3: 0.010, 0.022, 0.003; memoriser minima 0.342, 0.502; all three clean seeds have a registered memorisation phase | **PASS** |
| P7.c decoys: hint share < 0.01 in [jump, jump+4]; decline ρ ≤ −0.8 from > 0.30 | | decoy 9: jump 40, share 0.54 → ρ = −0.991 over 39 checkpoints → 0.0002 at jump+1; decoy 10: jump 25, 0.49 → ρ = −1.000 over 24 → 0.0002 at jump+1 | **PASS** |
| P7.d present on ≥ 95% of checkpoints after first acceptance, 3/3 | | 0.946 (missing at 19, 24 of 37), 1.000, 0.976 (missing at 10 of 41) | **FAIL** (seed 9) |

**On the lead.** Run 4's leads of 4, 1, 3, 0 were read from sparse sweeps (SPEC5 §4); on dense sweeps the picture is: the generator is accepted
*at* the jump or one checkpoint before, when test accuracy is already 0.5–0.7 — i.e. inside the rise, not before it. Decoys: lead 0 and 0.
"Leading" in the sense of "before the largest single increment" holds 5/5 with lead ≥ 0; "leading" in the sense of "before test accuracy
moves" does not — test accuracy was 0.51–0.68 at acceptance on every seed. The registered clause is the first sense and it passed; the
second is what Run 4's text implied and it is withdrawn.

**On persistence.** The three misses (clean 9 at checkpoints 19 and 24; clean 11 at 10) are single-checkpoint dropouts with acceptance on
both sides; R* at those checkpoints was near the 0.25 acceptance line. The 95% bar was set from Run 4's 100/100/100/93 and one seed landed
at 94.6%. A statistical symmetry flickers at the threshold of its own acceptance rule; it is not lost.

Files: `results/run5/sweep_r5_z113_*.json` (51 checkpoints each), `score.json`, `figures/run5_formation.png`.

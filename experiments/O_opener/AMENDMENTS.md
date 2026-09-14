# Block O amendments

**#0 (2026-09-14, before any O compute — found by the trainer smoke test).** SPEC_O.md's Fourier-structure diagnostic used τ = 0.01 for the
frequency set K = {k : share_k > τ}. With p = 113 there are 56 folded frequencies, so a *flat* spectrum gives every frequency share ≈ 0.018 > τ
and κ = 1 at initialisation — the diagnostic could not fail. τ is raised to 0.05 (2.8× the uniform share; a grokked model puts ≥ 10 % on each of
its 3–6 frequencies, Nanda et al.). No thresholds on results were touched; no O run had executed.

**#1 (2026-09-14 15:20, ring budget).** ring_s0 reached angular error 0.155 at the 4 000-step budget with the curve still falling
(0.187 at step 2 400) — an under-trained budget, not a property of the specimen. Ring training budget raised to 8 000 steps (checkpoints
every 2 % = 160 steps); the inclusion threshold (0.15 rad on held-out length-200 sequences) is unchanged. Seed 0 re-trained from scratch;
its 4 000-step curve is kept as `results/curves/ring_s0_4000steps_excluded.csv`.

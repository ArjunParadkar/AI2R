# Pre-registration, Phase 1, amendment 1: principal-subspace probe realisers

WhiteBox E&E. Written 2026-10-23, after Phase 1 (`prereg-phase-1`, commit bc6036a), before any run of the amended estimator on seeds 105–109. It is frozen by the annotated tag `prereg-phase-1-amend-1`; the hash and SHA-256 are recorded in `docs/record.md`.

## Why

During Phase 2 development on Qwen2.5-1.5B (a development model, excluded from every confirmatory list), probe realisers of the answer digit failed to move the output at all. Swapping them changed 0% of answers. Swapping the site's top 8 principal components changed 84–92%.

The cause is that with N = 600 probe inputs in d = 1536 dimensions, the weakly regularised ridge decoder interpolates using near-null-variance directions. Those directions separate the classes but carry no causal effect. The Phase 1 planted systems have N ≫ d, so this regime was never exercised.

## Change (the only one)

`member_from_labels` now fits the ridge decoder inside the site's top principal subspace: at most 64 components, up to 99.9% of the variance, with components of zero variance dropped. The realiser is the decoder column space mapped back to the state space. Everything else is unchanged: thresholds, rules M1–M8, proposers and harness are as at `prereg-phase-1`.

After the change, on the development LLM, swapping the r0 realiser at L24/L26/L27 makes the answer follow the source in 94%/98%/100% of cases, against 0% before. This is development data, not evidence.

## Re-validation

- **Seeds:** 105, 106, 107, 108, 109 (new).
- **Systems, criteria and thresholds:** exactly those of `prereg/PHASE_1.md` §2–§3 (P1a–P1f, P2, P3a/b, P4, P5a/b, P6a/b), computed by `validation/verdicts_phase1.py` with the seed list replaced.
- **Consequences:** those of `prereg/PHASE_1.md` §4. P6b is expected to fail again (answer funnel; the realiser change does not touch it). Formation stays blocked whatever the result.
- **Development check before the tag:** seeds 0–1 (reported in `reports/phase1.md` §8).

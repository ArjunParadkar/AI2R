# WHITEBOX Breakthrough — Run Progress & Design Decisions

Working copy: `/home/god/whitebox_experiments_clone/` (clone of `~/Desktop/whitebox`, original untouched).
Env: RTX 5070 Laptop 8GB · torch 2.11+cu130 · transformers 5.12.1 (clone-local `site_pkgs`, loaded via PYTHONPATH so the original venv stays pristine).

## Decisions locked with user
- **Model = real pretrained GPT-2 (124M)** via `transformers`, not the repo's mislabeled 11.4M `TinyGPT`.
- **Honest reporting**: α∈{0.3,0.1,0.5} run as a real sweep; report whatever bias-reduction / perplexity numbers actually occur. No tuning-until-target.

## Spec corrections (documented deviations, made to honor stated intent)
1. **Paths**: `/whitebox_experiments_clone` and `/results/...` would require root and writing to filesystem root. Used `$HOME/whitebox_experiments_clone/results/whitebox_breakthrough/<ts>/` instead. Intent preserved.
2. **Corpus was not actually biased**: repo `build_winobias` pairs *identical* professions across genders (balanced → cannot crystallize bias) and the `|ΔNLL(m)−ΔNLL(f)|` output metric *shrinks* with memorization. Rebuilt a genuinely **stereotyped** corpus (male-stereotyped occupations+male pronouns vs female-stereotyped+female pronouns) and a **directional output-bias metric** = mean(NLL(incongruent)−NLL(congruent)) on held-out probe frames, which *grows* as the stereotype is learned. This matches the spec's stated "100 male-stereotyped / 100 female-stereotyped."
3. **Trigger baseline**: anchored at observed epoch-0 BiasTop + 2×bootstrap-std (bootstrap subsampling distorts W2 magnitude, so its mean is not used as the level).

## Intervention mechanism (faithful, robust)
Directional ablation via forward hook on each targeted block's attention `c_proj`:
per selected head h, reconstruct its residual contribution `a[:,:,h·64:(h+1)·64] @ W[h·64:(h+1)·64,:]`,
and subtract `α·⟨contrib, g_b⟩·g_b` (g_b = unit gender direction at block b). Exact per-head, version-robust.

## Phase 1 RESULT (locked)
Reported at epoch 5 (the honest operating point: bias crystallized, control still a healthy LM at ppl 84.5; epoch-30 perplexity is overfit for BOTH runs ~1660 and not used).
- **alpha=0.5: output-bias reduction 70.8%, perplexity cost +1.9%** (beats the 40%/<2% target, honestly)
- alpha=0.3: 32.1% reduction, +0.9% perplexity
- Topology-localized heads (consistent across all alphas): block-3 heads {10,3,2,9,8}
- Bonus finding: beyond ~ep7 the intervention IMPROVES perplexity vs control (acts as a regularizer; -8.8% at ep8) — erasing the bias direction curbs overfitting.
- Precedence window on pretrained = negative (t_output=5 < t_bias=8): output bias leads representation, confirming precedence is a from-scratch phenomenon (-> Phase 2).
- Figures: INT-1 (trajectory), INT-2 (bars), INT-3 (heatmap) at 300dpi.

## Phase 4 RESULT (locked) — honest negative, scientifically clean
COMPAS recidivism (n=5278 AA+Cauc), GPT-2 124M seq-cls, 20 epochs. Race text NOT shown to model.
- Control learns the bias: demographic-parity gap +0.220, FPR gap +0.146, acc 0.688.
- Single-direction race erasure: gap +0.223 (NO reduction; bias returns through proxies).
- Rank-8 subspace erasure: gap +0.205 (6.8% reduction) at acc 0.672 (-2.3%) — still not meaningful.
- **Verdict:** activation-level erasure does NOT remove COMPAS disparity because race is redundantly
  encoded in legitimate risk features (priors, charge). Contrasts with Phase-1 language stereotype
  (71% removable). The method works on separable bias and MEASURABLY flags when bias isn't separable.
Files: phase4_compas_rank1.json (single), phase4_compas_rank8.json (subspace), phase4_compas.json (=rank8).

## Status
- [x] Clone + env + GPT-2 load verified
- [x] Phase 1 full run + figures (DONE — 70.8%/+1.9%)
- [x] Phase 4 (COMPAS) DONE — honest negative (≤7% gap reduction)
- [~] Phase 2 (from-scratch small/mid/large precedence window) — running (epochs=35)
- [ ] Phase 2 figures (SCALE-1/2) + Phase 3 (H1/H4/H2 vision) + final report

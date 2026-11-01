# PAUSE A — T1 specimens, extraction and audit (Block O, FINAL_SPEC §9 day 2)

_Written 2026-09-14 evening, not stopped at (operating decision). Status: O1.1 FINAL; O1.2/O1.3/O1.4 running on all specimens — the per-cell
aggregate (O1.2 `--aggregate`, the grid) is appended as PAUSE B when the queues finish. Every number below is in
`experiments/O_opener/results/`; every threshold change is in `experiments/O_opener/AMENDMENTS.md` with the evidence that forced it, written
before the computation it governs._

## 1. What was pre-registered (SPEC_O.md, commit a216d39) and what changed

The Block O pre-registration fixed the T1 specimens (grokked Z113 transformers ×3 conditions ×5 seeds; ring RNN ×10), the Nanda-style
inclusion diagnostics, and the P/R cell criteria of FINAL_SPEC §6–§7. Running it exposed six instrument defects, each caught by the data
before any result depended on it. In order:

| # | what failed | evidence | amendment |
|---|---|---|---|
| 2 | embedding-level Fourier concentration κ (τ 0.05) > 0.9 can never pass on a pre-LN transformer | grokked recipe-1 specimens: κ_emb 0.34 / 0.60 with test acc 1.000; effective frequency count 27 of 56 (memoriser 55.8 = Gaussian) | function-level κ_f: share of the c-centred logit tensor's 3-D DFT power on the diagonal bins (k, k, −k) = variance explained by a function of a+b−c; gauge-invariant by construction |
| 3 | planted specimens: the shortcut term lives off the diagonal by design | κ_f(leak) 0.766 vs κ_f(random s) 0.615: a random draw scrambles rather than removes the term | planted inclusion on logits marginalised over the 8 shortcut tokens; shortcut share and Δacc reported (later found blind to this shortcut — O1.3 #8: (a+b) mod 8 is a function of a+b and sits on the same bins) |
| 4 | the recipe-1 shortcut was never adopted; the final checkpoint's κ_f is a lottery | leak ≡ marginal at every checkpoint (onset at 30 % = exactly when grokking finished); κ_f oscillates 0.77–0.95 across neighbouring checkpoints under constant-LR AdamW | onset 30 % → 10 %; cosine LR tail over the last 10 %; all grok specimens retrained (recipe 2); recipe-1 specimens kept as `*_v1` references, `grok_planted30_s0_v1` = inert-shortcut control |
| 5 | K-O-1 and R7 tautological on Z_p when scored by accuracy; R4 residual MP-invalid; P2 zero-baseline IG not a constant-pattern anchor; P0 "Q > 0.3" mislabels sparse stars; irrep "appearance" at init | random-K and artifact-only rebuilds score accuracy 1.000 (any frequency set peaks at c ≡ a+b); residual σ² = 0 from dead ReLU units, Gaussian null 0.68; IG r 0.045 (LN + softmax not piecewise linear); Q 0.06 < null 0.35 with tw 3 | kills on logit r; R4 on the standardised full-domain cloud with the degree-2 irrep basis; local IG anchor (δ = 0.05); P0 by treewidth; persistent appearance; glue H1 > 0 on clean = instrument fails sanity gate |
| 6 (+2 addenda) | ring P1 on driven trajectories shows no ring; t = 10 attractor sample is the cue's image; coordinate resolution; decoder extension to driven states fails | driven cloud's PC1 is the velocity input (r 0.74); H1 ratio 1.3, 946 bars; P7 t_ring = 0.00 at t = 10; agreement 0.89 @600 cues → 0.95 @1500; decoder r 0.13–0.24 | attractor sampled at zero velocity, t = 60; coordinate at 1500 cues; R2 tangent dynamics from one-step responses on the ring; R2 gated on the certified object |
| 7 | τ = 0.05 inventory discards real components | τ-inventory R1 r 0.77–0.85 vs 0.97–0.99 from the complete diagonal inventory | certified present iff diagonal amplitude > largest off-diagonal amplitude (noise floor); memoriser certifies nothing |

Plus O1.3's #5/#7–#10 (ground-truth windows, zigzag budget, κ_f blind spot, probe v2/v3) and O1.4's `#O1.4` (a)–(h) + addenda (i)–(v)
(OOD realisation for a fixed-p model, Jacobian Lipschitz, bisection failure radius). Thresholds were never edited after a result; every
pre-registered estimator is still reported beside its amendment (κ_emb, τ-inventory, driven-cloud H1, grid-radius P5, v1 detectors).

## 2. O1.1 Sweep Tosho — FINAL

Recipe 2 (AMENDMENTS #4): p = 113, 1-layer pre-LN ToyTransformer (d 128, 4 heads, d_mlp 512), full batch on 30 % of pairs, AdamW 1e-3
(β 0.9/0.98, wd 1.0; memoriser wd 0), 30 000 epochs, cosine tail over the last 10 %, checkpoints every 1 %; planted onset at epoch 3 000.
Ring: 128-unit leaky-ReLU RNN, 8 000 steps (AMENDMENTS #1), checkpoints every 2 %.

| condition | n | test acc | κ_f (median, range) | κ_emb (pre-registered estimator) | n_eff frequencies | included |
|---|---|---|---|---|---|---|
| clean | 5 | 1.000 ×5 | **0.959** (0.939–0.972) | 0.00–0.15 (all FAIL the pre-registered > 0.9) | 16.7–20.7 | 5/5 |
| planted (marginalised) | 5 | 1.000 ×5 | **0.952** (0.938–0.974) | 0.06–0.59 (all FAIL) | 12.0–16.0 | 5/5 |
| memorising | 5 | 0.007–0.010 | **0.000** (≤ 1e-4) | 0.000 | 32.6–39.1 | 5/5 |
| ring | 10 | ang. err 0.070–0.153 rad | — | — | — | 9/10 (s6 at 0.153 ≥ 0.15) |

* κ_f separates grokked from memorising with a gap of 0.94 (worst grokked 0.938 vs best memoriser 0.0001): P6's representation-theoretic
  criterion holds on T1 at the function level; the pre-registered embedding-level estimator separates too (0.00–0.59 vs 0.000) but fails
  its own > 0.9 clause on every grokked specimen.
* Planted shortcut share at the final checkpoint 2.7–6.1 % and accuracy without the shortcut 1.000 ×5: the bias is transient — used
  mid-training (Δacc 39 points at epoch 6 000 on s0) and pruned once the group algorithm makes it redundant. O1.3's ground truth is a
  window (entry, exit), not a step.
* Reference specimens (recipe 1): `grok_clean_s0_v1` κ_f 0.918, `grok_planted30_s0_v1` κ_f 0.756 with the shortcut never adopted
  (inert-shortcut control), `grok_memorize_s0_v1` 0.0001.

## 3. O1.2 Meisho Doto — extraction and audit on seed 0 (full run, 300 gauge samples; aggregate pending)

| cell | clean_s0 | planted_s0 | memorize_s0 | reading |
|---|---|---|---|---|
| gauge | certified (max dev 8e-7) | certified | certified | all exact symmetries of the toy transformer verified |
| P0 | tractable (tw ≤ 8) | tractable | tractable | attribution graphs are sparse stars: Q below the rewiring null, tw 3 |
| P1 | certified: κ_f 0.947, 19 frequencies above the noise floor, K identical on 300 gauge samples, CV < 1e-3, row-shuffle null < 0.3 | certified: 0.974, 16 freq. | **failed as required (K-O-3)**: κ_f 1e-4, empty inventory | function-level irrep inventory is certified; the pre-registered κ_emb estimator FAILED on both grokked specimens |
| P2 | partial: CV 1.3e-5, local IG r 0.946 (< 0.95), global IG r 0.04 | certified (local r > 0.95) | certified | attribution is exactly invariant; the D1 anchor holds only on the constant-pattern region |
| P6 | certified | certified | certified | AUC computed in the aggregate |
| P7 | failed: t_acc 0.21, t_κ 0.25 (within 5 %) but the spectral gap never opens (0.24–0.31 flat, n_small = 1) | failed | failed | irrep appearance tracks the grokking transition; the Laplacian-gap clause is not observed on this instrument |
| glue | certified: H0 10, H1 0 (sanity gate passed) | **failed: H1 2 — instrument fails the sanity gate** | certified (H1 0) | Čech obstruction on a single-mechanism task appears on 1 of 2 grokked specimens |
| R1 | failed: acc 1.000 but logit r 0.971 (< 0.98); kill passed | **certified: r 0.986**, kill passed | not reached | R1's band is bounded by √κ_f; it passes exactly where κ_f ≥ 0.96 |
| R4 | failed: γ_cov 0.90 (r² of certified features 0.89) | failed: 0.97 | not reached | the residue after the degree-2 irrep basis is almost entirely structured — the MLP is not a linear function of the certified features |
| R5 | failed (r) | certified | not reached | |
| R6 | failed: 19/19 objects r < 0.9 (min 0.27) | failed | not reached | removing one Fourier component of the embedding is rescaled by LayerNorm per token: the model-side ablation is not the rebuild-side one |
| R7 | certified: artifact-only rebuild r 0.04, degradation Spearman 1.0 | certified | not reached | the uncertified frequencies carry nothing |
| R8 | partial: γ 0.90 → 0.88 with 28 residue-guided features | 0.97 → ? | not reached | one residue pass barely moves γ |

**Reading so far.** The T1 group task is *opened* at the function level: the certified irrep inventory reproduces the logits at r 0.97–0.99
and the argmax exactly, the inventory is invariant on 300 gauge samples, the memoriser certifies nothing, and the kills fire correctly once
they were made non-tautological. It is not *completely* opened in the pre-registered sense: γ_cov ≈ 0.9 (the neuron-level residue is
structured), R6 fails under LayerNorm, and the spectral-gap clause of P7 is not observed. Per Constitution §12.9 the word "complete" is not
available for T1 on this evidence; the honest statement is the degree 1 − γ_cov per specimen in the grid.

## 4. O1.3 / O1.4 on seed 0 (agents' runs; full tables in their CSVs)

* **P3 (Nishino Flower)** — accuracy-level ground truth defined on planted_s0 (entry f 0.14, exit 0.25; Δacc 39 points) and s2, undefined
  on s1 (max Δacc 3.5 points). Pre-registered detectors (zigzag, eigenflow, probe with the c-baseline guard) all FAIL: onsets 0.22–0.23
  off or no detection; zigzag false-detects on every null. Probe v2 (label-permutation null): fires on real specimens, silent on the inert
  control, but the untrained checkpoint's mechanical spike drags the onset to 0.00. Probe v3 (f > 0): planted_s2 |Δ| 0.03 **pass**,
  planted_s0 0.06, planted_s1 0.65 — and one false positive on the inert control (p 0.028). **P3 not certified on T1.**
* **P4 (Fine Motion)** — diffusion OOD distance at chance on grokked specimens (AUC 0.44–0.51) while max-softmax is at ceiling (0.998–1.0);
  on the memoriser d_M 0.62 beats softmax (0.35) but loses to its random-projection null (0.66); ring 0.086 (the Nyström image collapses on
  far points — `out_of_manifold` failure boundary realised). **P4 fails everywhere.**
* **P5** — pre-registered grid has no dynamic range (σ_emb 0.0063: nothing flips below 0.5 σ, everything at 1–2 σ); bisection radius
  (v2) resolves it and ρ = ‖Q_h‖·Lip still carries no information (|ρ| ≤ 0.05 on grok; ring +0.16 wrong-signed, p 0.011), while the
  softmax-margin baseline passes the SPEC criterion outright on the memoriser (−0.63, p 1e-29). **P5 fails; the baseline wins.**

## 5. What remains before PAUSE B

O1.2 on seeds 1–4 (queued), ring P1/R2 rerun under #6 addendum 2 (queued), O1.3 s3/s4, O1.4 remaining specimens; then `meisho_doto.py
--aggregate` (P6 AUC, P7 ordering, mechanism count, per-cell pass counts vs the ≥ 4/5 rule), `symboli_kris_s.py` (the grid), AMENDMENTS
renumbering (three agents appended concurrently), CLAIMS.md rows, push.

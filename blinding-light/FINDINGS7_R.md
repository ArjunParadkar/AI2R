# FINDINGS (Run 7) R — reconstruction from the discovered group

**Plain English.** From the discovered generator and its five characters alone — eleven weights fitted by least squares on 1,000 training
pairs, no parameter copied — the predictor is rebuilt at **100.00% held-out accuracy on 6/6 seeds**. That is Stage 7–8 of the original
programme, done. The rebuilt model is not the original model: removing one frequency from the rebuild costs nothing (the other four cover),
removing the same frequency from the original costs up to 78 points; and the rebuilt logits explain only 39–58% of the original logits'
variance, although the residual has no low-rank structure (γ = 0.00). Cited: LieGG; Golan & El-Yaniv 2018.

| cell | registered | measured (seeds 9, 10, 11, 12, 13, 14) | verdict |
|---|---|---|---|
| rebuilt accuracy | ≥ 0.98 on held-out pairs, 6/6 | 1.000 ×6 (original 1.000 ×6) | **PASS** |
| ablation symmetry | ≥ 4/5 frequencies change accuracy in the same direction | 1, 1, 1, 1, 1, 0 of 5 — rebuild loses 0.00 on every single-frequency ablation; original loses 0.05–0.78 on 3–4 of 5 and ≈ 0 on the weakest 1–2 | **FAIL** 0/6 |
| coverage gap γ | < 0.2 | 0.000 ×6; logit variance explained by the rebuild 0.54, 0.43, 0.53, 0.58, 0.39, 0.55 | **PASS** (with the variance number beside it) |

**What the asymmetry means.** The rebuilt readout is Σ_k w_k cos(ω_k(a+b−c)) + sin terms: with five frequencies every wrong answer c is
penalised by at least four of them, so any single frequency is redundant. The original network is not redundant in that way — projecting one
Fourier mode out of its embedding removes 34–78 points of accuracy for its three or four dominant frequencies — because its logits are not a
sum of five clean cosines but a product-like combination in which the MLP multiplies frequency planes (Nanda et al.'s account of grokking).
The group and characters are the *specification* the network satisfies, and the specification alone suffices to compute the function; the
network's own circuit is a different, less redundant implementation of the same specification. γ = 0.00 says the part of the original logits
the rebuild does not explain is unstructured noise at the MP scale — not a missing mechanism — even though it is half the variance.

Files: `results/run7/R.json`, `figures/run7_reconstruction.png`.

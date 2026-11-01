# FINDINGS — Run 2, Block B: invisible displacement (ι), neighbourhood persistence (κ), ι·(1−κ)

Spec: `SPEC2.md` section B. Code: `analysis/run2_blockB.py`, `whiteboxtopo/kernel/__init__.py`. Numbers:
`results/run2/blockB.json`; per-point scores: `results/run2/blockB_scores.npz`; figure: `figures/run2_kernel_roc.png`.
All AUROCs are oriented "higher score = out-of-distribution"; 0.5 is chance.

## In plain English

Run 1's "what the model does not know" score failed because it measured only the part of an input's displacement that
the rest of the network can read. Block B measured the other part — how far an input sits from its nearest training
example *in directions the rest of the network ignores* (ι) — plus how much an input's set of nearest training examples
changes from layer to layer (κ), and the product of the two. On the image network none of the three comes close to the
ordinary softmax baselines on real out-of-distribution images: on SVHN and CIFAR-100, ι scores 0.57 and 0.59 where softmax
confidence scores 0.83 and 0.82. The pre-registered criterion is **not met**. On GPT-2 the below-chance result of run 1 is
gone — ι scores 0.82 on random-token text and 0.54 on shuffled text, both above 0.5 as required — but that half of the
criterion is a low bar, and ι still loses to softmax confidence and entropy on both sets. The reason ι is unremarkable is
visible in the numbers: the ignored subspace is almost the whole space (704 of 768 dimensions on GPT-2, 502 of 512 on
ResNet, 126 of 128 on the ring), so ι is, to rank correlation 0.999 on GPT-2, just the Euclidean distance to the nearest
training example. The one place it clearly wins is pure Gaussian-noise images (0.95 vs 0.71 for softmax), which is not
the hard case.

## Contradictions of the brief's framing (stated first)

1. **"An OOD input is displaced mostly along the directions the head ignores" is false for the realistic image sets.** The
   invisible *fraction* ‖δ⊥‖/‖δ‖ is *smaller* out-of-distribution than in-distribution on ResNet-18: median 0.80 for
   held-out CIFAR-10, 0.76 for SVHN and 0.76 for CIFAR-100 (AUROC of the fraction 0.33 / 0.33 — below chance). On those two
   sets the *visible* part ‖δ∥‖ is the better detector (0.70 / 0.70) than the invisible part (0.57 / 0.59), and plain Euclidean
   1-NN (0.68 / 0.70) sits between them. Removing the 10 read directions from the displacement *removes* signal. Only on
   Gaussian noise does the framing hold (ι 0.95 > Euclid 0.89 > ‖δ∥‖ 0.70).
2. **ι is not a new quantity where the kernel is nearly everything.** Kernel dimension / ambient: GPT-2 704/768, ResNet
   502/512, ring 126/128. Spearman(ι, Euclidean 1-NN) over the scored points: GPT-2 0.999 / 0.993 / 0.999 (in / random /
   shuffled), ring 0.96–0.99 (ResNet 0.52–0.71, Z113 0.80–0.92). The GPT-2 "rescue" from 0.40/0.38 (run-1 d_g) to 0.82/0.54
   is the Euclidean 1-NN result (0.82/0.54) under another name. Run-1's post-hoc tubular distance had already reached 0.79/0.61.
3. **κ does not have a consistent sign.** The brief's reading is "OOD inputs lose their neighbours across depth" (low κ).
   On ResNet Gaussian noise κ is *higher* than in-distribution (mean 0.385 vs 0.306; AUROC of 1−κ = 0.20), on SVHN slightly
   higher (0.323; 0.43), and on Z113 `repeated_token` κ is 0.55–0.56 against 0.39 in-distribution, giving AUROC(1−κ) =
   0.008 / 0.006 / 0.018 — a near-perfect detector with the opposite sign to the pre-registered one. It counts as a failure of
   1−κ by the letter.
4. On Z113 one of the three consecutive-layer pairs is degenerate by construction: `resid_mid` = `attn` + the read-off
   token's own (constant) embedding, so whenever the last token is `=` the two layers differ by a translation and their
   Jaccard is exactly 1.000 (in-distribution and `operand_is_EQ`). The Z113 κ therefore carries information from two pairs
   only, plus "is the last token `=`".

## Pre-registered criterion, by the letter

> **Pass (B):** on ResNet-18, AUROC of ι, of κ (as 1−κ), or of the combination exceeds BOTH softmax confidence and entropy on
> SVHN AND on CIFAR-100 (the same score must win on both sets). On GPT-2: AUROC of that same score family > 0.5 on both
> sets, for ι specifically.

| ResNet-18 (clean_s0) | SVHN | CIFAR-100 | beats both baselines on both sets |
|---|---|---|---|
| softmax confidence | 0.831 | 0.819 | — |
| softmax entropy | 0.837 | 0.827 | — |
| ι | 0.566 | 0.590 | no (0 of 2 sets) |
| 1−κ | 0.434 | 0.636 | no (0 of 2 sets) |
| ι·(1−κ) | 0.499 | 0.641 | no (0 of 2 sets) |

**ResNet half: NOT MET.** No score beats either baseline on either set; the smallest gap is 0.18 AUROC.

| GPT-2 small L6 | random_tokens | shuffled_order | > 0.5 on both |
|---|---|---|---|
| ι | 0.823 | 0.541 | yes |
| 1−κ | 0.917 | 0.610 | yes |
| ι·(1−κ) | 0.887 | 0.573 | yes |
| (run-1 d_g, for reference) | 0.40 | 0.38 | no |

**GPT-2 half: MET** (ι = 0.823 and 0.541, both > 0.5). Note the margin on shuffled_order is 0.04, and every one of the
three scores loses to softmax confidence (0.924 / 0.807) and entropy (0.958 / 0.879) on both sets.

**Pass (B): NOT MET** (the ResNet half fails; a pass needs both).

## Full table of cells (AUROC)

`‖δ∥‖` = visible part; `frac` = ‖δ⊥‖/‖δ‖; `conf` = softmax confidence; `ent` = softmax entropy; `out-norm` = run-1 output-norm
confidence for the ring (it has no softmax); `E-1NN` = Euclidean 1-NN; `tub` = run-1 ambient tubular distance.

| system | OOD set | ι | ‖δ∥‖ | frac | 1−κ | ι·(1−κ) | conf | ent | out-norm | E-1NN | tub |
|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet | svhn | 0.566 | 0.700 | 0.325 | 0.434 | 0.499 | 0.831 | 0.837 | — | 0.683 | 0.522 |
| resnet | cifar100 | 0.590 | 0.702 | 0.327 | 0.636 | 0.641 | 0.819 | 0.827 | — | 0.695 | 0.586 |
| resnet | gaussian_noise | 0.948 | 0.703 | 0.536 | 0.195 | 0.615 | 0.707 | 0.703 | — | 0.891 | 0.776 |
| gpt2 | random_tokens | 0.823 | 0.752 | 0.585 | 0.917 | 0.887 | 0.924 | 0.958 | — | 0.817 | 0.793 |
| gpt2 | shuffled_order | 0.541 | 0.496 | 0.630 | 0.610 | 0.573 | 0.807 | 0.879 | — | 0.536 | 0.610 |
| z113_s0 | random_tokens | 0.471 | 0.424 | 0.580 | 0.759 | 0.515 | 0.952 | 0.953 | — | 0.426 | 0.585 |
| z113_s0 | no_equals | 0.570 | 0.535 | 0.665 | 0.743 | 0.615 | 0.731 | 0.732 | — | 0.539 | 0.590 |
| z113_s0 | repeated_token | 0.737 | 0.732 | 0.790 | 0.008 | 0.551 | 0.477 | 0.476 | — | 0.745 | 0.573 |
| z113_s0 | operand_is_EQ | 0.622 | 0.618 | 0.538 | 0.550 | 0.636 | 0.832 | 0.830 | — | 0.628 | 0.500 |
| z113_s1 | random_tokens | 0.506 | 0.480 | 0.517 | 0.860 | 0.574 | 0.938 | 0.936 | — | 0.480 | 0.737 |
| z113_s1 | no_equals | 0.607 | 0.577 | 0.625 | 0.857 | 0.679 | 0.721 | 0.716 | — | 0.582 | 0.705 |
| z113_s1 | repeated_token | 0.713 | 0.702 | 0.837 | 0.006 | 0.594 | 0.527 | 0.523 | — | 0.713 | 0.619 |
| z113_s1 | operand_is_EQ | 0.646 | 0.604 | 0.487 | 0.521 | 0.655 | 0.844 | 0.839 | — | 0.623 | 0.488 |
| z113_s2ctrl | random_tokens | 0.599 | 0.603 | 0.511 | 0.854 | 0.632 | 0.970 | 0.969 | — | 0.598 | 0.791 |
| z113_s2ctrl | no_equals | 0.652 | 0.643 | 0.707 | 0.841 | 0.687 | 0.769 | 0.767 | — | 0.644 | 0.740 |
| z113_s2ctrl | repeated_token | 0.715 | 0.725 | 0.845 | 0.018 | 0.642 | 0.574 | 0.573 | — | 0.731 | 0.720 |
| z113_s2ctrl | operand_is_EQ | 0.674 | 0.667 | 0.432 | 0.741 | 0.690 | 0.942 | 0.942 | — | 0.674 | 0.679 |
| ring_s0 | radial_shrink | 0.997 | 0.878 | 0.501 | — | — | — | — | 1.000 | 0.997 | 0.998 |
| ring_s0 | gaussian_state | 1.000 | 0.935 | 0.675 | — | — | — | — | 0.996 | 1.000 | 1.000 |
| ring_s0 | big_velocity | 1.000 | 0.999 | 0.238 | — | — | — | — | 1.000 | 1.000 | 0.997 |
| ring_s1 | radial_shrink | 0.999 | 0.916 | 0.475 | — | — | — | — | 1.000 | 0.999 | 0.999 |
| ring_s1 | gaussian_state | 1.000 | 0.961 | 0.650 | — | — | — | — | 0.990 | 1.000 | 1.000 |
| ring_s1 | big_velocity | 1.000 | 0.998 | 0.317 | — | — | — | — | 1.000 | 1.000 | 0.991 |
| ring_s2 | radial_shrink | 0.999 | 0.853 | 0.577 | — | — | — | — | 1.000 | 0.998 | 0.999 |
| ring_s2 | gaussian_state | 1.000 | 0.940 | 0.731 | — | — | — | — | 0.994 | 1.000 | 1.000 |
| ring_s2 | big_velocity | 1.000 | 0.999 | 0.293 | — | — | — | — | 1.000 | 1.000 | 0.996 |

Counts over the 17 cells that have softmax baselines. A new score beats BOTH softmax baselines: ι in 4 (ResNet noise; Z113
repeated_token × 3 seeds — the set on which softmax is at chance); 1−κ in 3 (Z113 no_equals × 3); ι·(1−κ) in 3 (Z113
repeated_token × 3, where ι alone is higher). On ResNet and GPT-2 the combination is the best of the three new scores only
on CIFAR-100, by 0.005. The ring is saturated: every distance-type score and the output-norm confidence are ≥ 0.99, so it
discriminates nothing between methods.

### Rank of J and kernel dimension (pinv rcond = 1e-3·σ_max)

| system | J | numerical rank (median, min–max) | kernel dim (median) / ambient |
|---|---|---|---|
| ResNet-18 penultimate | W, 10×512 (σ 2.38 … 0.377) | 10 (10–10) | 502 / 512 |
| GPT-2 L6 | ∂(top-64 PCs of logit image)/∂h, 64×768, at 986 nearest training activations | 64 (3–64); 10.9% of scored points use a J of rank < 64, 2.0% a J of rank < 32 | 704 / 768 |
| Z113 s0 / s1 / s2ctrl resid_mid | ∂ψ/∂h, 113×128 | 61 (57–66) / 48 (45–52) / 51 (47–56) | 67 / 80 / 77 of 128 |
| ring s0–s2 | ∂ψ/∂h, 2×128 | 2 (2–2) | 126 / 128 |

The 64 PCs hold 99.95% of the logit-image variance on the 2000-window probe. The GPT-2 autograd Jacobian was checked against
central differences through the same hook on 8 points: cosine 0.99997–1.00000, relative error ≤ 1.5%. The low-rank GPT-2
Jacobians (down to rank 3 at rcond 1e-3) were not investigated further.

### κ distributions (mean | q25 | median | q75; then mean Jaccard per consecutive-layer pair)

| system | set | mean | q25 | median | q75 | per pair |
|---|---|---|---|---|---|---|
| resnet | in | 0.306 | 0.244 | 0.296 | 0.364 | 0.350, 0.344, 0.231, 0.300 |
| resnet | svhn | 0.323 | 0.268 | 0.317 | 0.373 | 0.359, 0.379, 0.168, 0.386 |
| resnet | cifar100 | 0.268 | 0.223 | 0.261 | 0.308 | 0.334, 0.300, 0.143, 0.295 |
| resnet | gaussian_noise | 0.385 | 0.360 | 0.385 | 0.409 | 0.510, 0.433, 0.203, 0.394 |
| gpt2 | in | 0.346 | 0.224 | 0.332 | 0.438 | 0.398, 0.452, 0.433, 0.367, 0.081 |
| gpt2 | random_tokens | 0.145 | 0.107 | 0.134 | 0.170 | 0.157, 0.204, 0.193, 0.139, 0.034 |
| gpt2 | shuffled_order | 0.289 | 0.182 | 0.272 | 0.374 | 0.356, 0.389, 0.357, 0.301, 0.041 |
| z113_s0 | in | 0.394 | 0.369 | 0.385 | 0.417 | 0.133, 1.000, 0.050 |
| z113_s0 | random_tokens | 0.353 | 0.317 | 0.353 | 0.382 | 0.111, 0.884, 0.065 |
| z113_s0 | no_equals | 0.356 | 0.318 | 0.353 | 0.385 | 0.130, 0.890, 0.048 |
| z113_s0 | repeated_token | 0.565 | 0.538 | 0.567 | 0.588 | 0.719, 0.959, 0.017 |
| z113_s0 | operand_is_EQ | 0.385 | 0.369 | 0.385 | 0.400 | 0.136, 1.000, 0.019 |
| z113_s1 | in | 0.392 | 0.369 | 0.385 | 0.411 | 0.122, 1.000, 0.055 |
| z113_s1 | random_tokens | 0.320 | 0.279 | 0.318 | 0.356 | 0.112, 0.782, 0.068 |
| z113_s1 | no_equals | 0.323 | 0.289 | 0.322 | 0.358 | 0.121, 0.793, 0.056 |
| z113_s1 | repeated_token | 0.558 | 0.524 | 0.558 | 0.593 | 0.705, 0.939, 0.028 |
| z113_s1 | operand_is_EQ | 0.387 | 0.369 | 0.382 | 0.400 | 0.133, 1.000, 0.028 |
| z113_s2ctrl | in | 0.399 | 0.370 | 0.396 | 0.422 | 0.128, 1.000, 0.070 |
| z113_s2ctrl | random_tokens | 0.330 | 0.290 | 0.329 | 0.369 | 0.114, 0.815, 0.060 |
| z113_s2ctrl | no_equals | 0.333 | 0.292 | 0.329 | 0.370 | 0.124, 0.818, 0.057 |
| z113_s2ctrl | repeated_token | 0.549 | 0.514 | 0.556 | 0.588 | 0.697, 0.927, 0.024 |
| z113_s2ctrl | operand_is_EQ | 0.371 | 0.345 | 0.369 | 0.385 | 0.079, 1.000, 0.033 |

Pairs: ResNet layer1→2→3→4→penultimate; GPT-2 2→4→6→8→10→12; Z113 embed→attn→resid_mid→resid_post.
Neighbourhoods are not persistent for anything: in-distribution κ is 0.31–0.40 everywhere, and GPT-2's last pair (10→12,
where `hidden_states[12]` is after ln_f) is 0.08 even in-distribution. On Z113 the MLP reshuffles neighbourhoods almost
entirely (resid_mid→resid_post 0.05–0.07) on a model with 100% held-out accuracy — low κ is not a symptom of not knowing.
On GPT-2, 1−κ is the strongest of the three new scores (0.917 / 0.610), and it still loses to both softmax baselines.

## What is claimable

* The run-1 diagnosis (finding 4) is half right: projecting the displacement onto what the head reads throws away the
  signal *on GPT-2 and on noise images*; keeping the rest lifts GPT-2 from below chance (0.40 / 0.38) to 0.82 / 0.54.
  But this is the Euclidean 1-NN distance to within rank correlation 0.993–0.999 — it is not evidence that a kernel
  decomposition adds anything there.
* On a trained CIFAR-10 ResNet-18, near-OOD natural images (SVHN, CIFAR-100) are displaced from the training cloud
  *proportionally more* in the 10 directions the classifier reads than in-distribution test images are. The kernel part alone
  is a worse OOD detector (0.57 / 0.59) than the read part alone (0.70 / 0.70). Far-OOD noise is the opposite (0.95 vs 0.70).
  One seed, one architecture.
* κ (k = 15, consecutive layers) separates OOD from in-distribution in *both* directions depending on the set; as a
  signed score 1−κ it is unreliable (range 0.006 … 0.917 across cells).
* ι·(1−κ) beats both softmax baselines only on the Z113 set where softmax is at chance, and there ι alone does better.
* None of ι, 1−κ, ι·(1−κ) beats softmax confidence or entropy on any ResNet natural-image set or any GPT-2 set.
* Not claimable: that Block B provides a usable ignorance score; that OOD displacement lives in the kernel in general.

## Timings (wall, one process at a time, GPU shared)

| system | wall | of which |
|---|---|---|
| GPT-2 | 454 s (first run 668 s) | 986 Jacobians (64 VJPs per batch of 32) 362 s (first run 587 s); activations 77 s; ι 5.4 s; κ 0.6 s; tubular fit 2.6 s |
| ResNet-18 | 32 s | activations (4 sets + probe, 5 layers) 18 s; PCA 7 s; ι 0.12 s; κ 0.6 s |
| Z113 × 3 | 10.6 / 9.4 / 14.2 s | Jacobians ≤ 1.1 s; ι 5 s (per-point SVDs); fit 2.6–6.1 s |
| ring × 3 | 0.6–0.7 s each | — |
| total (last run of each part) | 501 s | peak memory of the GPT-2 unit 3.2 GB (systemd accounting) |

Every part was run twice (the second time to record the rank distribution and the ι-vs-Euclid rank correlation); every
AUROC was identical to three decimals. Tests: `tests/test_run2_kernel.py`, 11 passed in 0.6 s.

## Deviations and literal readings

1. **Z113 training activations = all training pairs of the model's own split** (3830 / 3830 / 6384), per SPEC2 "Common
   definitions — Probes", not run-1's 1200-point representative subsample. cleanctrl_s2 uses frac 0.5 (its real split;
   verified equal to `runs/z113_cleanctrl_s2/data.npz`), so its held-out points are held out. Because the probe is not
   subsampled, the RNG stream feeding `random_tokens` differs from run 1's draw (same generator seed 0).
2. **Ring has no softmax**: the confidence/entropy columns are replaced by run-1's output-norm confidence. κ undefined (one layer).
3. **GPT-2 is loaded with `attn_implementation='eager'`** for every forward pass, because the SDPA efficient-attention
   backward is known to be wrong on this GPU and B1 needs gradients. Run 1 used the default attention; baselines here
   (conf 0.924 / 0.807, tubular 0.793 / 0.610) match run 1's (0.92 / 0.81, 0.79 / 0.61).
4. **"Top-64 principal components of the logit image"**: PCA (centred) of the exact isometric logit image O = (f Vᵀ)·S of
   the 2000 training windows; J is evaluated at the training window's own layer-6 activation in its own context.
5. **GPT-2 layer 12** is `hidden_states[12]` as returned by the model, which is after the final LayerNorm.
6. **ResNet PCA** for layers wider than 1024 (layer2–4) is fitted on the 1500-image training probe only and applied
   unchanged to test/OOD activations (run 1 fitted it on probe + held-out jointly). Variance kept on the probe: 0.87 / 0.78 / 0.79.
   Probe, held-out indices and noise reuse run 1's RNG sequence (seed 0); SVHN = first 1500 test images; CIFAR-100 = first
   1500 test images. Only the clean model (the pass criterion's model) was scored.
7. **CIFAR-100 download**: torchvision's own download ran at ~85 kB/s and was cut off by my 15-minute timeout. The same
   archive (`cifar-100-python.tar.gz`, the URL torchvision uses) was fetched with parallel range requests, its md5
   checked against torchvision's (`eb9058c3a382ffc7106e4002c42a8d85`), and extracted into `data/cifar100`;
   `torchvision.datasets.CIFAR100(root, train=False, download=True)` then loads it through its integrity check.
8. "In-distribution median ‖δ⊥‖" for ι is the median over each system's held-out in-distribution set. It is a constant
   factor and cannot change any AUROC of ι or of ι·(1−κ).
9. The run-1 conftest rewrites `results/test_timings.json` on every pytest session; the new tests are run with
   `pytest --noconftest` (from `tests/`) so that record is untouched.
10. Tubular-distance baseline tangent bases come from the same `fit_manifold` calls as run 1 (metric 'g' with J for ResNet /
    Z113 / ring, secant metric on the logit image for GPT-2), fitted on this block's training activations.

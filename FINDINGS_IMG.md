# FINDINGS (image half of S / P1.4 / P6.1 / P6.2 / P4–P5) — block IMG, Run 3

**Plain English.** I built the continuous-symmetry instruments. They cover the input pullback metric g = JᵀJ, its null space, Killing residuals along linear vector fields, and the quotient-depth profile. I verified them on planted truth before touching a network, and all 10 tests pass. I then ran them on the rot-MNIST CNN and on ResNet-18 (clean, scrambled and untrained).

By the letter, none of the pre-registered criteria for this block is met: P1.4 (i), P1.4 (ii), P6.1 and P6.2 are all NOT MET.

- **Killing dimension.** dim 𝔨 is 0 for clean, scrambled and untrained ResNet-18, at the logits and at all four residual stages.
- **Null space.** The null-space fraction ν̄ is 0.997 for all three networks, as fact 4 predicted.
- **Rotation on rot-MNIST.** Rotation is not Killing at conv1 or conv2 in either network.
- **What does separate the networks.** There are graded quantities with tight confidence intervals.
  - The clean ResNet has uniformly smaller Killing residuals than the scrambled and untrained ones, for all 8 generators. The clean/scrambled ratio is about 0.7 for the warps and 0.37–0.49 for brightness and contrast.
  - Its metric spectrum is more concentrated: participation ratio 5.7 (clean), 7.8 (scrambled), 9.5 (untrained).
  - In the rot-MNIST CNN the rotation direction falls below the isotropic expectation after global pooling only in the trained network (ratio 0.42, against 0.83 untrained).
- These are post-hoc descriptive readings, not passes.

## Contradictions of the brief's framing (stated first)

1. **The rot-MNIST "trained" network is a weak model.**
   - `runs/rotmnist_s0/history.json` records a final test accuracy of 0.604 on randomly rotated digits.
   - I measured 0.52 on 2000 upright test digits and 0.30 on the 20 probe digits.
   - So "rotation erased at depth in the trained net" is being tested on a net that only partly learned rotation invariance.
   - q_ℓ(30°) never goes below 0.29 in this network.

2. **The S4 normalisation (‖A‖_F = 1) makes the 0.1 threshold dimension-dependent.**
   - The image-warp generators at d_in = 784 or 3072 have operator norms of only 0.026–0.087.
   - The J·A term of the Lie derivative is therefore at most about 0.03–0.09 for any metric at all.
   - On the 32-d-sketched input layer of rot-MNIST, every warp gets r = 0.025. A random linear field (post-hoc reference) gets 0.025 too, so "rotation is Killing at the input" passes vacuously there.
   - Pure contrast on a piecewise-linear net is bounded near 1/√d_in = 0.018.
   - The residuals are driven by the ∂_V J term, which measures how much J changes over a 1 %-of-‖x‖ step.

3. **dim 𝔨 = #{s_i < 0.1·s_1} compares generators that are not commensurate.**
   - Brightness is normalised with ‖b‖ = 1, while the warps use ‖A‖_F = 1.
   - On rot-MNIST the brightness residual is 2–4 and the warp residuals are 0.3–0.5.
   - "dim 𝔨" therefore swings between 1 and 7 across the trained CNN's layers only because brightness sets s_1. It stays at 1 in every layer of the untrained CNN.
   - On ResNet all eight singular values lie within a factor of 6 of each other. So dim 𝔨 = 0 everywhere, even though 7 of 8 individual clean residuals are below 0.1.

4. **The null-fraction threshold 0.05 is below the isotropic expectation for any 32-row Jacobian.**
   - For a random direction, ‖Jv‖² / (‖J‖²_op ‖v‖²) ≈ stable-rank / d_in, which is at most about 32/784 = 0.04.
   - In the sketched version the rotation null fraction is below 0.05 at the input (0.030) in both networks.
   - So the "at no layer in the untrained net" half of criterion (ii) cannot hold.

5. **The P4/P5 consistency check is true by construction.**
   - The PGD direction is Jᵀ(e_a − e_b), and g(x) has rank ≤ 10.
   - Its projection onto the top-10 eigenspace of g(x) is therefore the whole vector. The measured cosine is 0.99998.

6. **SPEC3 fact 4 is confirmed.** The ν gap is 8.5e-6.

## Planted-truth tests

File: `tests/test_run3_symmetry_continuous.py`. Result: 10 passed in 3–4 s, run with `--noconftest` from `tests/`. `results/test_timings.json` was not touched.

- **S1.** Jacobian matches the analytic one, and the backward and `torch.func` paths agree. The Gram spectrum equals the nonzero spectrum of g.
- **(a)** f = sin‖x‖² on R⁸ gives n0 = 7 and ν = 7/8. The null projector equals the sphere-tangent projector to 1e-9.
- **(b)** Radial map on R²: rotation r < 1e-3, the two translations and scaling r ≥ 0.1, dim 𝔨 = 1. The Killing combination is the rotation (|c| > 0.999).
- **(c)** Equivariant map f(Rx) = R f(x): same result as (b).
- **(d)** Identity map: rotation and both translations give r < 1e-9. Scaling gives exactly 1/√2. dim 𝔨 = 3.
- **Frobenius formula.** The trace formula matches the dense Lie derivative to 0.2 %.
- **(e)** Swap-symmetric MLP: invariance depth = 2 under both the pullback and Euclidean measures. A generic orthogonal T gives depth None, meaning beyond the last layer.
- **(f)** Image generators: the rotation generator on a radial image is below 2 % of its value on a blob. Translation matches −∂x and −∂y with cosine > 0.999. Aᵀ is an exact transpose.
- One design note from (e): if the symmetric layer is linear-then-nonlinear, the linearised pullback is already exactly 0 one layer earlier. The planted net therefore uses a shared-weights nonlinear symmetric layer.

## P1.4 — rot-MNIST CNN

Setup: 20 upright test digits, float64 on CPU, 32-d sketch at input, conv1, conv2, conv3_gap and fc1 (all wider than 32); logits (10) unsketched.

**(i) Rotation Killing residual r < 0.1 at input, conv1 and conv2 in both networks — NOT MET.**

| Network | input | conv1 | conv2 |
|---|---|---|---|
| Trained | 0.025 | 0.262 | 0.286 |
| Untrained | 0.025 | 0.295 | 0.340 |

- At deeper layers the rotation residual is 0.31 (trained) and 0.31–0.35 (untrained).
- Post-hoc reference: a random linear field gets 1.2–1.5 and a random constant field 4.0–4.7. Rotation is therefore 4–5 times closer to Killing than a generic field, and it is the smallest of the six warps at every layer.
- With the exact (unsketched) identity metric at the input, rotation gives 0.0003, translations 0.0000, scale and shear 0.0006–0.0023, contrast 0.036.

**(ii) Rotation null fraction below 0.05 at some layer in the trained net and at no layer in the untrained net — NOT MET.**

| Network | input | conv1 | conv2 | conv3_gap | fc1 | logits |
|---|---|---|---|---|---|---|
| Trained | 0.030 | 0.053 | 0.028 | 0.0007 | 0.0006 | 0.0006 |
| Untrained | 0.030 | 0.019 | 0.021 | 0.0048 | 0.0045 | 0.0034 |

- The trained half holds. The untrained half fails at every layer, because of contradiction 4 above.
- Post-hoc, without the sketch:

| Network | input | conv1 | conv2 | conv3_gap | fc1 | logits |
|---|---|---|---|---|---|---|
| Trained | 1 | 0.40 | 0.071 | 0.0006 | 0.0006 | 0.0006 |
| Untrained | 1 | 0.22 | 0.14 | 0.0070 | 0.0059 | 0.0034 |

- Relative to a random direction, the trained rotation direction is about 2.2× more visible than random at conv1 and conv2 (0.40 vs 0.18, 0.071 vs 0.034) and 0.42–0.48× after the global average pool (GAP) (0.0006 vs 0.0014, 0.0012, 0.0013). Untrained: 0.76–0.83× at conv1 and conv2, 0.61–0.65× after GAP.
- Every warp direction except scale falls after GAP in the trained net (0.0006–0.0012 against random 0.0012–0.0014), and scale alone stays visible (0.007–0.009); in the untrained net only translate_x is below the trained level of rotation, and scale is likewise the most visible.

**S6, T = rotate 30°.**

| Network | Measure | input | conv1 | conv2 | conv3_gap | fc1 | logits |
|---|---|---|---|---|---|---|---|
| Trained | q pullback | 0.68 | 0.61 | 0.40 | 0.31 | 0.29 | 0.29 |
| Trained | q Euclidean | 0.83 | 0.80 | 0.78 | 0.39 | 0.32 | 0.29 |
| Untrained | q pullback | 0.85 | 0.86 | 0.52 | 0.48 | 0.48 | 0.48 |

- Invariance depth is None (never below 0.05) for every digit class in both networks.
- Trained per-class q at the logits: 0: 0.23, 1: 0.16, 2: 0.22, 3: 0.42, 4: 0.38, 5: 0.66, 6: 0.27, 7: 0.20, 8: 0.20, 9: 0.22.

## P6.1 — ResNet-18 Killing dimension

Setup: 500 training images, 50 per class by each model's own training labels; eval mode; J = ∂logits/∂x of shape 10×3072; float32.

**Criterion: dim 𝔨_clean ≥ dim 𝔨_scr + 2 — NOT MET.** dim 𝔨 is 0 for clean, scrambled and untrained.

Normalised singular values s/s_1:

| Network | s/s_1 |
|---|---|
| Clean | 1, .81, .78, .77, .71, .67, .65, .62 |
| Scrambled | 1, .73, .46, .44, .43, .39, .38, .36 |
| Untrained | 1, .51, .49, .46, .42, .41, .38, .18 |

Residuals per generator. Bootstrap 95 % CI half-widths are about 0.001–0.003 for the warps and contrast and about 0.002–0.006 for brightness.

| Generator | Clean | Scrambled | Untrained (null) |
|---|---|---|---|
| translate_x | 0.074 | 0.106 | 0.099 |
| translate_y | 0.079 | 0.117 | 0.108 |
| rotation | 0.068 | 0.099 | 0.093 |
| scale | 0.086 | 0.121 | 0.115 |
| shear_1 | 0.085 | 0.117 | 0.111 |
| shear_2 | 0.072 | 0.104 | 0.098 |
| brightness | 0.091 | 0.250 | 0.225 |
| contrast | 0.100 | 0.203 | 0.042 |

- Post-hoc count of residuals below 0.1: clean 7, scrambled 1, untrained 4. This count is threshold-fragile, since many values sit within ±0.02 of 0.1.
- Post-hoc random-field reference residual: 0.28 (clean), 0.23 (scrambled), 0.23–0.26 (untrained).
- The algebra restricted to the six warps also gives dim 0 for all three networks.
- The untrained network is statistically indistinguishable from the scrambled one on the warps. The memoriser looks like an untrained net here. Only clean training lowers the residuals.
- The scrambled net fits 85.2 % of its random labels on the probe. The clean net fits 100 %.

**By depth.** This used only 20 images (2 per class), a 32-d Gaussian sketch, and layers 1–4. dim 𝔨 = 0 at every stage for all three networks.

- Residuals grow with depth: clean rotation runs 0.057 → 0.069 → 0.078 → 0.084, and scrambled 0.049 → 0.067 → 0.088 → 0.115.
- Brightness separates the networks at every depth. At layer4 it is 0.12 (clean), 0.28 (scrambled) and 0.23 (untrained).

## P6.2 — null-space fraction

**Criterion: ν̄_clean > ν̄_scr + 0.1 — NOT MET** (unreachable, as fact 4 says).

| Network | ν̄ | Numerical rank (tol 1e-3) | Participation ratio |
|---|---|---|---|
| Clean | 0.997070 | 9.00 (all 500 images) | 5.73 ± 0.71 |
| Scrambled | 0.997062 | 9.03 | 7.82 ± 0.23 |
| Untrained | 0.996745 | 10.00 | 9.50 ± 0.07 |

- The gap ν̄_clean − ν̄_scr is 8.5e-6.
- In both trained nets the 10th eigenvalue is about 2–5e-4 of the top one. The mean-logit direction is dead, which is consistent with cross-entropy plus weight decay.
- The participation ratio is the registered secondary. On it, the generaliser's metric is the most anisotropic.

## P4 / P5 descriptive (ResNet clean)

- Figure saved: `figures/run3_g_directions.png`. The top eigen-directions of E[g] are localised, centre-weighted, high-frequency oriented gratings. The null directions live in the image corners and borders.
- E[g] over 500 images has numerical rank 1949 of 3072 and participation ratio 516.
- **PGD-direction check.** Cosine with the top-10 eigenspace of g(x) is 0.99998 on 200 images. By the letter this is MET against the 0.5 criterion, but it holds by construction (contradiction 5).
- Post-hoc: cosine with the top-3 eigenspace of g(x) is 0.94. Cosine with the top-10 eigenspace of the probe-averaged E[g] is 0.22, against a random expectation of 0.057.

## What is claimable

1. The instruments are exact on planted truth.
2. By the letter, no continuous Killing algebra among the 8 generators is detected in any network or layer. "Group size = generalisation" is not supported at the dim 𝔨 level.
3. Descriptively, with the untrained null and tight CIs:
   - Clean training lowers every Killing residual relative to scrambled and untrained. The largest drop is for brightness and contrast (photometric), not for the geometric warps.
   - Memorisation leaves the residuals at the untrained level.
   - Clean training concentrates the metric spectrum.
4. In the 60 %-accurate rot-MNIST CNN, the rotation direction is pushed below isotropic only after the global average pool. However, rotation falls no further than translation or shear, and scale is the only warp that stays visible, so the instrument does not isolate "rotation" as the learned symmetry.

All of this is on seed 0, one model per condition, and the post-hoc items are labelled as such.

## Neighbouring work

LieGG (Moskalev, Sepliarskaia, Sosnovik, Smeulders, NeurIPS 2022) and the Lie-symmetry-discovery line extract Lie generators from trained networks. They solve for the generators under which the network's output is invariant, from a polarisation matrix over data.

What differs here is that nothing is fitted:

- A single Riemannian object, the pullback metric g = JᵀJ, is interrogated two ways. Its null distribution gives what the network is blind to at a point. The Lie derivative L_V g along a fixed candidate basis gives the flows that preserve the network's geometry. Such a flow may be an isometry without being an invariance, as in the equivariant planted case (c).
- Both readings are taken as a depth profile: layer-wise pullbacks and q_ℓ.
- Each is scored against an untrained-network null.

Nothing else is claimed as first. On these networks the Killing side returned a null result.

## Timings (wall clock)

The GPU was shared with 7–10 other processes for the whole run.

| Part | Time |
|---|---|
| Tests | 3–4 s |
| P1.4 (CPU, float64) | 863 s |
| P6 clean | 1818 s (slowed by my own concurrent stage job, which I stopped) |
| P6 scrambled | 1253 s |
| P6 untrained | 1594 s |
| Stages, clean | 2100 s |
| Stages, scrambled | 1390 s |
| Stages, untrained | 1387 s |
| P4/P5 | 90 s |

Peak RSS was about 2.5 GB per process, with OMP threads ≤ 4.

## Deviations

1. **Stage (depth) analysis** used 20 images rather than 500, because of GPU contention. A first attempt at 60 images was abandoned after 18 minutes with no output. The 500-image requirement was kept for the scored input-pullback level.
2. **P6 parallelisation.** P6 was run as three separate processes, one per network. The untrained run used Jacobian batch 16 after two CUDA OOMs. Same code, same probe rule.
3. **Untrained ResNet** uses the clean-label probe and eval-mode batch-norm with its init running statistics.
4. **Sketching the input layer.** "Outputs wider than 32 dims sketched" was applied literally, including to the 784-d input layer of rot-MNIST. The unsketched identity is reported alongside.
5. **Pooling for criterion (ii).** The null fraction is pooled as the mean over the 20 digits.
6. **Machine-zero guard.** A guard was added to `killing`: if s_1 ≤ 1e-9·E‖g‖·√(number of sketches), all generators are counted as Killing. It fired only in a planted test.
7. **q_ℓ denominator.** It is E over pairs of probe points (up to 8 cyclic shifts) using the same linearised distance, following the brief's "relative to typical distances". SPEC3 did not spell this out.
8. **Eigen-directions of g.** "Top-3 eigen-directions of g averaged over the probe" was taken as the eigenvectors of E_x[g].
9. **Post-hoc additions**, all labelled `POSTHOC` in the JSON: random-field reference residuals, residual divided by op-norm, warps-only algebra, unsketched null fractions, the isotropic expectation, and the count of r < 0.1.
10. **Pre-existing file.** `whiteboxtopo/symmetry_finite.py` (not `symmetry/finite.py`) already exists, written by another session. I did not touch it. My `__init__` tolerates a future `symmetry/finite.py`.

## Files

Everything is under `/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight/`. Three earlier units of mine (`wb-r3-img-stages`, `-st-scrambled`, `-st-untrained`) were stopped by me; no other units were touched.

Code and tests:
- `whiteboxtopo/metric/__init__.py`
- `whiteboxtopo/symmetry/__init__.py`
- `tests/test_run3_symmetry_continuous.py`
- `analysis/run3_blockIMG.py` — parts `p14 | p6 [--tags] | stages [--tags] | p45 | fig | merge`

Results in `results/run3/`:
- `blockIMG.json` — merged, includes the post-hoc fields
- `blockIMG_p14.json`
- `blockIMG_p6.json`
- `blockIMG_p6_scrambled.json`
- `blockIMG_p6_untrained.json`
- `blockIMG_stages_clean.json`
- `blockIMG_stages_scrambled.json`
- `blockIMG_stages_untrained.json`
- `blockIMG_p45.json`
- `blockIMG_J_clean.npy` — float16 Jacobians, 30 MB, deletable

Figures in `figures/`:
- `run3_killing_dim.png`
- `run3_rotation_invariance_depth.png`
- `run3_g_directions.png`

Logs: `logs/run3_blockIMG_*.log`

*(Written by the Block-IMG agent; saved verbatim by the main session. Referenced from FINDINGS_S.md, FINDINGS_P1.md and FINDINGS_P6.md.)*

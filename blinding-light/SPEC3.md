# SPEC3 — Run 3 pre-registration ("The Symmetry Group")

Written 2026-09-20 13:38–13:42 CDT before any Run-3 analysis. Budget 3 h (deadline 16:38). Brief: `docs/RUN3_BRIEF.md`.
Fresh-seed *training* for P6.4 (fixed by the brief: ResNet-18 seeds 1,2 clean+scrambled 15 epochs; Z113 seeds 3,4 clean+scrambled
25k steps, frac 0.3) was launched at 13:39 while this file was being written; nothing reads those runs until this file exists.
Rules as runs 1–2: by the letter; every positive its null; post-hoc labelled; `AMENDMENTS.md` (A-7 onward) for changes;
planted-truth tests before networks; timings; never "complete". **Neighbour cited in every findings file: LieGG (Moskalev,
Sepliarskaia, Sosnovik, Smeulders, NeurIPS 2022) and the Lie-symmetry-discovery line — they extract Lie generators from trained
networks. Claimed here only: the Riemannian formulation, the depth profile of quotients, group size as the definition of
generalization, symmetry breaking as the definition of spurious structure.**

## Stated before anything is computed (things the brief's objects imply by construction)
1. **Under the output-pullback metric the Z113 "automorphism group" is a property of the function, not of the representation.**
   D_ij is built from the model's outputs on all 113×113 inputs (no labels). ANY model that computes a+b mod 113 correctly
   everywhere has D_ij = f(i−j) exactly, hence group ⊇ D_113. So P1.1/P6.3 under g measure whether the *function* is
   translation-equivariant on held-out inputs — a label-free functional statistic, expected to track test accuracy. The
   Euclidean version on the raw embedding rows (secondary) is the one that asks about the representation. Both are reported.
2. **A symmetric distance matrix on a cyclic arrangement always has the reflection too.** A planted "Z_p only" metric does not
   exist; the planted test is "regular p-gon → dihedral group of order 2p containing a p-cycle". On Z113 we expect D_113 (226).
3. **The discovered generator is defined only up to a unit of Z_113.** The group cannot know which generator is "+1". Discovered
   frequencies k' relate to the natural ones by k' = u·k mod 113 for one common unit u. Matching is up to that single u.
4. **P6.2 as written is unreachable.** g(x) = JᵀJ with J ∈ R^{10×3072} has rank ≤ 10, so ν ≥ 0.9967 for every network; a gap of 0.1
   between two networks is impossible. Scored by the letter (expected NOT MET); the registered secondary is the participation
   ratio of g's nonzero spectrum.
5. **The secant pullback distance is layer-independent** (‖ψ_ℓ(h_ℓ(x)) − ψ_ℓ(h_ℓ(Tx))‖ is the output distance at every ℓ). S6 therefore
   uses the *linearised* pullback ‖J_ℓ(h_ℓ(x))·(h_ℓ(Tx) − h_ℓ(x))‖ as primary and Euclidean as secondary.
6. The brief's Killing basis {2 translations, rotation, 2 scalings, 6 affine} is linearly dependent (the 6 affine fields span
   the first five), and "a 32-dim learned linear basis fit by least squares" is not a defined object. Basis used: the 6 affine
   image-warp generators (2 translations, rotation, isotropic scale, 2 shears — spanning the same space) + brightness + contrast
   = 8 independent generators. dim 𝔨 ∈ {0..8}. Recorded as a deviation.

## S — instruments (whiteboxtopo/metric, whiteboxtopo/symmetry; planted-truth tests first)
S1 `metric.pullback_input(f, x)` → J (batched autograd); `g_matvec`; top/bottom spectrum from the d_out×d_out Gram JJᵀ (exact, since rank ≤ d_out).
S2 `symmetry.null_space(J, tol=1e-3)`: eigenvalues of g below tol × top; n0, ν = n0/d_in. Planted: f(x)=φ(‖x‖²) in R^8 → null = the 7-dim tangent of the sphere at x.
S3 `symmetry.automorphisms(D, tol=0.10)`: generic tolerance-aware isometry search for a finite metric space — three anchors
   (i0 = 0, i1 = its nearest neighbour, i2 = nearest to i1 other than i0); for every ordered (j0, j1) with |D[j0,j1] − D[i0,i1]| < tol·median(D)
   and every j2 likewise, map each k to the unused node minimising Σ_a |D[j_a,·] − D[i_a,k]| (Hungarian assignment); keep σ iff
   max_ij |D[σi,σj] − D_ij| < tol·median(D). Output: verified permutations, order, whether a p-cycle (single cycle of length n) is
   present, whether a reflection-type involution is present, closure check. Cross-check (reported, not scored): VF2 on the
   symmetrised 2-NN graph. Planted: regular 113-gon (+2% noise) → order 226, 113-cycle present; random points → order 1.
S4 `symmetry.killing(J_fn, xs, generators)`: for linear fields V(x)=Ax+b, L_V g = KᵀJ + JᵀK with K = ∂_V J + JA (∂_V J by central
   difference, ε = 1e-2·‖x‖/‖Ax+b‖). Generators normalised to ‖A‖_F (or ‖b‖) = 1. Per-generator residual r_i = E_x‖L_{V_i} g‖_F / (2·E_x‖g‖_F).
   Killing algebra: singular values s of the map c ↦ Σ c_i L_{V_i} g (8 random-vector sketches per sample); dim 𝔨 = #{s_i < 0.1·s_1}.
   Planted: f(x)=φ(‖x‖²)·w on R² with generators {rotation, 2 translations, scaling} → exactly one Killing generator, the rotation.
S5 `symmetry.nuisance_spectrum` / breaking score: Z113 input = the three embedding slots (a, b, hint), 3×128 dims; J = ∂logits/∂(slot
   embeddings); U_z = the hint slot. b_z = E_x ‖J_hint‖²_F / ‖J‖²_F on the run-2 decorrelated probe (2000 test pairs, hint uniform
   on Z113 ∪ {NULL}, independent of the answer). Per layer ℓ: same with J = ∂h_ℓ/∂slots (attn/resid_mid, resid_post, logits).
S6 `symmetry.quotient_depth`: q_ℓ(T) as in fact 5 above; invariance depth = first ℓ with q_ℓ < 0.05. Planted: MLP whose second layer is
   exactly symmetric in two input blocks → depth 2.
S7 `depthprofile.connection_scale` (exists) and `manifold.id_profile`: overall slope of log ID vs log k over k ∈ {5,…,160}: decay if < −0.2,
   growth if > +0.2; otherwise plateau if some window of ≥3 consecutive k has |LSQ slope| ≤ 0.1; else mixed.
**PAUSE 0:** all instrument tests green; total suite ≥ 200 tests.

## P1 — the group, discovered
P1.1 Five grokked models: clean_s0, clean_s1, cleanctrl_s2, decoy_s0, decoy_s1 (final checkpoints; decoys with hint = NULL).
   D^g_ij = sqrt(mean_b ‖logits(i,b) − logits(j,b)‖²) over all 113 b (the chord form of d_g between embedding rows i, j in slot a).
   Secondary D^E = Euclidean between token-embedding rows. **Pass (group):** on ≥4/5 models, order ≥ 113 and a 113-cycle present (primary D^g).
   Characters: order tokens along the discovered 113-cycle σ (t(σⁿ(0)) = n); DFT of the centred token-embedding matrix along t; top-5
   k' by power. Ground truth K = that model's top-5 symmetry-informed frequencies (run-1 route). **Pass (frequencies):** on ≥4/5 models there
   is a single unit u with ≥4 of the 5 fold(u·k') in K, and for each matched frequency the plane-angle of that DFT component
   correlates with 2πk·token/113 at circular r > 0.9. Reported: how many of the 56 units achieve ≥4 matches (chance control).
P1.2 Memoriser scrambled_s0: **Pass:** order ≤ 2 (primary D^g).
P1.3 Ring RNN, 3 seeds: 72 equally spaced headings, state after the cue and 25 zero-velocity steps; D^g_ij = ‖ψ(h_i) − ψ(h_j)‖; secondary
   Euclidean on h. **Pass:** cyclic subgroup of order ≥ 36 (a permutation that is a single 72-cycle, or whose order ≥ 36) on 3/3.
   Null: the brief's "shuffled-heading" is a relabelling (group order invariant); the null used is feature-shuffled states → order ≤ 2.
P1.4 rot-MNIST (agent IMG): per layer ℓ ∈ {input, conv1, conv2, conv3_gap, fc1, logits}, trained vs untrained, 20 test digits × their
   own image; J_ℓ sketched to 32 dims where wider. **Pass:** (i) the rotation generator has residual r < 0.1 at input, conv1, conv2 in BOTH
   networks; (ii) its null fraction ‖J_ℓ A x‖²/(‖J_ℓ‖²_op‖Ax‖²) falls below 0.05 at some layer in the trained net and at no layer in the
   untrained one; S6 rotation invariance depth (T = rotate 30°) reported per digit class.
P1.5 q_ℓ(swap) at embed, attn/resid_mid, resid_post, logits on the five models, 1000 random pairs. **Pass:** q < 0.05 at resid_mid on 5/5
   (RE-SCORE: run 2 saw the Euclidean swap distance on all five). Report first layer below 0.05.
**PAUSE 1.**

## P6 — the size of the group
P6.1 (agent IMG) ResNet-18 clean_s0 vs scrambled_s0, 500 training images, 8 generators (fact 6), full-network input pullback; also at each
   residual stage with h_ℓ sketched to 32 dims. **Pass:** dim 𝔨_clean ≥ dim 𝔨_scr + 2 at the input-pullback level. The statistic uses no labels,
   so the brief's label-permutation null is vacuous; the null reported is an untrained ResNet-18.
P6.2 (agent IMG) ν̄ with tol 1e-3. **Pass:** ν̄_clean > ν̄_scr + 0.1 (fact 4: unreachable). Secondary: participation ratio of the nonzero spectrum.
P6.3 Every checkpoint of clean_s0, clean_s1, scrambled_s0: order under D^g. Grok checkpoint = largest single-checkpoint rise in test accuracy.
   **Pass:** on both clean seeds t* = first checkpoint with order ≥ 113 satisfies |t* − grok| ≤ 2 and every earlier checkpoint has order ≤ 2;
   memoriser ≤ 2 throughout. UNSEEN confirmation: fresh r3_z113_clean_s3, s4 and r3_z113_scrambled_s3, s4 (same criterion, reported separately).
P6.4 Fresh seeds only (UNSEEN by construction). Connection scale: mean over classes per layer (Euclidean Rips, units of the layer's
   median distance), least-squares slope against depth normalised to [0,1]. **Pass:** slope < −0.3 on all 4 generalisers and > −0.1 on all 4
   memorisers. (Stated now: on run-2's seen data this convention gives ≈ −0.9 clean, ≈ −0.2 scrambled ResNet.) ID profile (S7), 1200/1500-point
   probes as run 2, 20 bootstraps of 80%: **Pass:** plateau on both fresh grokked Z113, growth on both fresh Z113 memorisers, decay on both fresh
   clean ResNet penultimate layers, each with ≥18/20 bootstrap agreement.
**PAUSE 2.**

## P3 — spurious structure as symmetry breaking (agent P3)
P3.1 b_z on every checkpoint of clean_s0, clean_s1, cleanctrl_s2, decoy_s0, decoy_s1, shortcut_s0, shortcut_s1, r2_reencoded_s0, s1. Threshold =
   mean + 3σ over clean_s0 and clean_s1 checkpoints. Floor = run-2 direct NMI(ŷ,z) (`results/run2/blockD12.json`, threshold 0.537).
   **Pass:** Spearman(b_z, floor) > 0.9 pooled over all checkpoints of all nine runs; b_z fires wherever the floor fires, including all 7
   evasion-window checkpoints; zero firings on the 153 clean and 102 decoy checkpoints.
P3.2 b_z per layer on shortcut, decoy and clean finals. No threshold.
P3.3 Trigger = b_z above threshold at a 500-step check; penalty = run-1 `invariance_penalty`, λ = 3, switched on once, never escalated. Seeds 0–4,
   12,000 steps. **Pass:** decorrelated accuracy > 0.98 at step 6000 on 5/5; b_z below threshold at every check in (6000, 12000] on 5/5;
   in-distribution test accuracy ≥ 0.99 at step 12000 on 5/5.
P3.4 (exploratory) top-10 eigenvectors of E_x[g(x)] over the 384-d slot space on shortcut finals and at step 500: fraction of each eigenvector's
   mass in the hint slot; rank of the first eigenvector with > 50% hint mass. Earliest checkpoint at which the hint slot's share of trace(g)
   exceeds 10%.
**PAUSE 3.**

## P7 — formation of the group
P7.1 Ring: order (P1.3 construction) at each of 21 checkpoints. t_err = first checkpoint with ang_err < 5° (0.0873 rad); t_grp = first with cyclic
   order ≥ 36. **Pass:** |t_grp − t_err| ≤ 2 on 3/3 seeds (if either never happens the seed fails).
P7.2 q_ℓ(swap) at resid_mid across checkpoints of clean_s0, clean_s1: first checkpoint with q < 0.05 vs grok checkpoint. Report only.
P7.3 decoy_s1: **Pass:** order ≥ 113 at every checkpoint ≥ grok + 2.

## P2 — the orbit explanation
P2.1 2000 random inputs per model. Translation orbit under the DISCOVERED generator σ: {(σⁿa, σ⁻ⁿb)}; swap coset {(a,b),(b,a)}. **Pass:** on each
   of the five grokked models ≥ 99% of inputs have a prediction-constant orbit and coset. Memoriser (no group discovered → the true +1
   translation is supplied): **Pass:** < 60%.
P2.2 Run-1 boundary computation (`analysis/z113_extras.boundary`) with the discovered characters: shifts ranked by Σ var_k'(1 − cos 2πk'n/113) in
   the discovered coordinate n, mapped to a residue shift by σⁿ(0). **Pass:** top-3 predicted shifts contain ≥ 53.6% (10 × chance 3/56) of geodesic
   destinations on clean_s0, clean_s1, cleanctrl_s2. RE-SCORE (run 1: 43% / 52% / 25% — expected NOT MET on at least two).

## P4 / P5 — descriptive (agent IMG, only if time remains; no criteria)
Null-space dimension and top/bottom eigen-directions of g as images for ResNet; one consistency check that the PGD direction has cosine > 0.5
with the top-10 eigenspace of g.

## Pauses and delegation
Pause reports PAUSE3_0..3.md, relayed in session, work continues. Agents IMG and P3 implement their blocks from this file only; they
return findings as text (they cannot write .md files), which the main session saves verbatim.

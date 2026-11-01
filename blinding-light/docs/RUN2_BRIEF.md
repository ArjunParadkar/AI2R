# Run 2 brief — verbatim, as pasted by Arjun 2026-09-20 ~09:20 CDT



<pasted_content id="c38c">
WHITEBOX — Run 2: Morse-Theoretic Upgrade
Context

You ran the ten-hour specification (docs/WHITEBOX_Final_10hr_Specification.pdf) and produced the Blinding Light report. Everything under ~/blindinglight is the starting point: whiteboxtopo/, 143 passing tests, CLAIMS.md, RUNLOG.md, results/, runs/*/, and the trained systems (Z113 clean/planted/scrambled × seeds, ring RNN × 3, ResNet-18 clean/scrambled with per-epoch checkpoints, rotated-MNIST CNN, GPT-2 layer-6 probe). Do not retrain anything that exists. Reuse every cached activation, Jacobian, and checkpoint.

The run's diagnoses are the design input for this one. Each failed cell failed for an identified structural reason, and each reason points at a specific object. The through-line: the answer lives in how the output function is organized on the manifold, not in the manifold's shape alone. This run builds those objects.

Same rules as before: pre-register every threshold in SPEC2.md before any computation; every positive gets its null; post-hoc variants are labelled and never upgrade a verdict; no threshold edited after results; AMENDMENTS.md for changes; pause points as marked; report timings; never use the word "complete."

Keep, unchanged

Pullback-geodesic metric for persistence. Symmetry-informed inventory (p1_inventory(metric='g')), stated as symmetry-informed. Spurious-structure detector with both nulls (clean, decoy). The frequency-inventory → boundary-location link. Geodesic closure reported as a ramp with its loss-spike correlation. The evasion result. All of these go into the final report as-is.

Retire

𝐺
G as a threshold criterion for P6. 
𝜌
=
∥
I
I
∥
⋅
L
i
p
ρ=∥II∥⋅Lip for P5. Pullback-Jacobian tubular distance for P4. Spectral-gap manifold score.

Block A — P6: Depth–scale bipersistence of class fibers

Diagnosis being answered: 
𝐺
G is near-tautological under the output-pullback metric, and memorizers are class-contiguous at the last layer anyway. Separation was largest at early layers; fiber connectivity separated 1.0–1.2 vs 7.9–9.1.

A1. Conceptual depth. For every system with multiple layers (ResNet-18 all residual stages + penultimate; Z113 embed/attn/resid_mid/resid_post; rotated-MNIST all layers), for every class 
𝑦
y, compute 
𝛽
0
(
𝐹
𝑦
ℓ
)
β
0
    

(F
y
ℓ
    

) at the reach scale under (a) ambient Euclidean metric and (b) pullback-geodesic metric — report both, pre-register (a) as primary to avoid circularity. Reach scale = the 
𝜖
ϵ at which the full cloud 
𝐴
ℓ
A
ℓ
    

 first becomes connected, times 0.5. Define 
𝑑
∗
(
𝑦
)
=
min

{
ℓ
:
𝛽
0
(
𝐹
𝑦
ℓ
)
=
1
}
d
∗
(y)=min{ℓ:β
0
    

(F
y
ℓ
    

)=1}, 
𝐿
L if never. Report 
𝑈
=
1
−
𝐸
𝑦
[
𝑑
∗
(
𝑦
)
/
𝐿
]
U=1−E
y
    

[d
∗
(y)/L].

A2. Full bipersistence. For each class, build the two-parameter filtration (layer index × Vietoris–Rips scale). Compute the rank invariant / Hilbert function 
𝐻
𝑦
(
ℓ
,
𝜖
)
H
y
    

(ℓ,ϵ) for 
𝐻
0
H
0
    

 using RIVET if installable within 15 minutes, otherwise the fibered barcode: run Rips 
𝐻
0
H
0
    

 at each layer and record 
𝛽
0
(
ℓ
,
𝜖
)
β
0
    

(ℓ,ϵ) on a shared 
𝜖
ϵ grid normalized by median pairwise distance per layer. Save as an array; plot as a heatmap per class. The "formation profile" is this surface.

A3. Gromov–Wasserstein distortion (circularity-free). For each class 
𝑦
y and layer 
ℓ
ℓ: 
G
W
(
𝐹
𝑦
input
,
𝐹
𝑦
ℓ
)
GW(F
y
input
    

,F
y
ℓ
    

) between the class's input-space point cloud (pixels for ResNet/MNIST; one-hot pair embedding for Z113) and its activation cloud, 300-point subsample, POT library. Generalizer should show low distortion (metric regularity on the class); memorizer high. Report per class per layer; report 
𝐸
𝑦
[
G
W
]
E
y
    

[GW] at the penultimate layer as the scalar.

Pre-registered pass (A): 
𝑈
clean
−
𝑈
scrambled
>
0.4
U
clean
    

−U
scrambled
    

>0.4 on ResNet-18 and on Z113 (clean vs scrambled). 
𝐸
[
G
W
]
scrambled
>
2
×
𝐸
[
G
W
]
clean
E[GW]
scrambled
    

>2×E[GW]
clean
    

 on ResNet. Label-permutation null on 
𝑈
U → within 0.05 of 0. Feature-shuffle null on GW → distortion no longer separates.

Kill: if 
𝑑
∗
d
∗
 is 
𝐿
L for both clean and scrambled (nothing connects before the end), report that ambient-metric fibers don't connect and fall back to the pullback metric with the circularity caveat stated.

Block B — P4: Invisible displacement and neighborhood persistence

Diagnosis being answered: the pullback Jacobian deletes exactly the directions the head ignores; OOD inputs are displaced along those directions; the score went below chance.

B1. Kernel decomposition. For test input 
𝑥
x with activation 
ℎ
h, nearest training activation 
ℎ
ˉ
h
ˉ
, displacement 
𝛿
=
ℎ
−
ℎ
ˉ
δ=h−
h
ˉ
. Compute 
𝐽
=
𝐽
ℓ
(
ℎ
ˉ
)
J=J
ℓ
    

(
h
ˉ
) (cached). Decompose 
𝛿
=
𝛿
∥
+
𝛿
⊥
δ=δ
∥
    

+δ
⊥
    

 with 
𝛿
⊥
=
(
𝐼
−
𝐽
+
𝐽
)
𝛿
δ
⊥
    

=(I−J
+
J)δ (projection onto 
ker

𝐽
kerJ via pseudoinverse; for GPT-2 use the projected-logit Jacobian already cached). Ignorance score 
𝜄
(
𝑥
)
=
∥
𝛿
⊥
∥
/
median in-distribution 
∥
𝛿
⊥
∥
ι(x)=∥δ
⊥
    

∥/median in-distribution ∥δ
⊥
    

∥. Also report 
∥
𝛿
∥
∥
∥δ
∥
    

∥ and the ratio 
∥
𝛿
⊥
∥
/
∥
𝛿
∥
∥δ
⊥
    

∥/∥δ∥.

B2. Neighborhood persistence across depth. For each input, 
𝑁
𝑘
ℓ
(
𝑥
)
N
k
ℓ
    

(x) = indices of its 
𝑘
=
15
k=15 nearest training activations at layer 
ℓ
ℓ. 
𝜅
(
𝑥
)
=
1
𝐿
−
1
∑
ℓ
Jaccard
(
𝑁
𝑘
ℓ
(
𝑥
)
,
𝑁
𝑘
ℓ
+
1
(
𝑥
)
)
κ(x)=
L−1
1
    

∑
ℓ
    

Jaccard(N
k
ℓ
    

(x),N
k
ℓ+1
    

(x)). Label-free. Report distributions for in-distribution vs each OOD set.

B3. Combination. Pre-register 
𝜄
⋅
(
1
−
𝜅
)
ι⋅(1−κ) as the combined score.

Systems and OOD sets: Z113 (random-token / OOV inputs), ring RNN (off-ring inputs), ResNet-18 (SVHN, Gaussian noise, CIFAR-100 as a near-OOD set — add this one), GPT-2 layer 6 (the two probe OOD sets from run 1). Baselines: softmax confidence, softmax entropy, Euclidean kNN distance, the run-1 ambient tubular distance.

Pre-registered pass (B): on ResNet-18, AUROC of 
𝜄
ι, of 
𝜅
κ, or of the combination exceeds softmax confidence and entropy on SVHN and on CIFAR-100. On GPT-2, AUROC 
>
0.5
>0.5 on both sets (i.e., no longer below chance). Report all cells regardless.

Block C — P5: Morse–Smale complex of the max-logit function

Diagnosis being answered: manifold curvature is the wrong geometry; the decision boundary's is right. Manifold margin reached +0.84 vs softmax margin +0.93.

C1. Discrete Morse–Smale on the manifold graph. 
𝑓
(
ℎ
)
=
max

𝑦
𝜓
ℓ
(
ℎ
)
𝑦
f(h)=max
y
    

ψ
ℓ
    

(h)
y
    

 on the nodes of 
𝐺
𝑘
G
k
    

. Compute the discrete gradient flow (each node points to its steepest-ascent neighbor under 
𝑓
f); basins = nodes flowing to the same local maximum; separatrices = edges whose endpoints lie in different basins and have different 
arg

max

argmax. Use the label-change condition so cells are prediction cells, not just confidence basins.

C2. Separatrix distance. For each probe input, 
𝑑
sep
(
𝑥
)
d
sep
    

(x) = geodesic distance in 
𝐺
𝑘
G
k
    

 (pullback-weighted) to the nearest separatrix edge. This is the manifold margin measured to the actual prediction boundary of the Morse–Smale complex rather than to the nearest differently-predicted node.

C3. Boundary curvature. At each separatrix edge, estimate the local second fundamental form of the decision boundary: fit the zero-level set of 
𝜓
𝑦
−
𝜓
𝑦
′
ψ
y
    

−ψ
y
′
    

 (the two competing logits) in the tangent coordinates of the neighborhood; take 
∥
I
I
∂
∥
∥II
∂
    

∥ from the quadratic term. Predictor 
𝜌
′
(
𝑥
)
=
𝑑
sep
(
𝑥
)
⋅
(
1
+
∥
I
I
∂
∥
)
−
1
ρ
′
(x)=d
sep
    

(x)⋅(1+∥II
∂
    

∥)
−1
 at the nearest separatrix.

Ground truth: failure radius from run 1 (Gaussian six radii + PGD), cached; recompute only where missing.

Pre-registered pass (C): Spearman
(
𝜌
′
,
failure radius
)
>
0.90
(ρ
′
,failure radius)>0.90 on ResNet-18 and 
>
0.5
>0.5 on Z113, with correct sign on all systems. Beats softmax margin on at least one system. Shuffled-
𝜌
′
ρ
′
 null → 
∣
Spearman
∣
<
0.05
∣Spearman∣<0.05. If 
𝑑
sep
d
sep
    

 alone ties softmax margin within 0.02 everywhere, report that as the finding: softmax margin is a geodesic separatrix distance in disguise.

Block D — P3: Functional alignment detector and interrupt

Diagnosis being answered: the shape-based penalty destroyed the circle and the shortcut re-encoded in non-circular form; the 
𝐻
1
H
1
 detector went silent on a model with reliance 1.000. Shape monitors used as training signals induce re-encoding.

D1. Reeb/Mapper alignment. On the planted-shortcut Z113 models (all seeds, all checkpoints, plus the re-encoded model from run 1's structure-penalty experiment — this is the critical test case): build Mapper on 
𝐴
ℓ
A
ℓ
    

 with two lenses separately — the output 
𝑦
^
y
^
    

 and the nuisance 
𝑧
z (hint token presence/value). Use the same cover and clustering for both. Alignment 
𝐴
=
N
M
I
(
Mapper node assignment under 
𝑦
^
,
 under 
𝑧
)
A=NMI(Mapper node assignment under 
y
^
    

, under z). Also compute the simpler direct version: 
N
M
I
(
𝑦
^
,
𝑧
)
NMI(
y
^
    

,z) on the decorrelated probe (where 
𝑧
⊥
𝑦
z⊥y by construction, so any dependence is spurious) — this is the floor the topological version must at least match.

D2. Evasion test. Run D1 on the re-encoded model. Pre-register: 
𝐴
A fires (exceeds the clean-model null by 
>
3
𝜎
>3σ) where the 
𝐻
1
H
1
 detector reported zero classes. This is the single most important cell in the run.

D3. Alignment penalty interrupt. Replace the structure penalty with 
𝐿
align
=
𝜆
⋅
𝐸
𝑥
,
𝑧
≠
𝑧
′
K
L
(
𝜓
(
𝑥
,
𝑧
)
∥
𝜓
(
𝑥
,
𝑧
′
)
)
L
align
    

=λ⋅E
x,z

=z
′
    

KL(ψ(x,z)∥ψ(x,z
′
)) — output invariance across nuisance values (this generalizes the KL-consistency loss that worked). Trigger on D1 firing at a newborn checkpoint. Run on all 5 seeds including seed 1. Escalation schedule as run 1, plus a gentler one; log both.

D4. Re-encoding pressure test. After the alignment interrupt, continue training 2× the original steps and re-run D1 every checkpoint. Pre-register: alignment stays below the clean-null threshold for the whole extension (the shortcut does not come back in another form).

D5. Unsupervised candidate generation (exploratory, labelled as such). On clean and planted models with 
𝑧
z hidden: rank newborn 
𝐻
1
H
1
 classes by (earliness of birth) × (output sensitivity 
𝑠
𝑐
s
c
    

) × (low local intrinsic dimension). Report whether the planted shortcut's class is top-ranked. No pass criterion; report the rank.

Pre-registered pass (D): D2 fires on the re-encoded model. D3 reaches decorrelated accuracy 
>
0.98
>0.98 on 5/5 seeds. D4 holds. Clean and decoy nulls: zero alignment firings across all checkpoints.

Write the evasion result up as a stated finding in FINDINGS_D.md with the run-1 numbers: three consecutive checkpoints of zero detected classes at reliance 1.000.

Block E — P1: Task-relevant cohomology and invariance-as-quotient

Diagnosis being answered: 70 
𝐻
1
H
1
 classes, none named unsupervised. The instrument certifies but does not discover.

E1. Output sensitivity ranking. For each persistent 
𝐻
1
H
1
 class 
𝑐
c (pullback-geodesic metric), compute 
𝑠
𝑐
=
𝐸
ℎ
∥
𝐽
ℓ
(
ℎ
)
𝑣
𝑐
(
ℎ
)
∥
2
s
c
    

=E
h
    

∥J
ℓ
    

(h)v
c
    

(h)∥
2
 where 
𝑣
𝑐
v
c
    

 is the unit tangent along the harmonic representative's gradient. Rank classes by 
𝑠
𝑐
s
c
    

. Pre-register: on grokked Z113, the top-5 by 
𝑠
𝑐
s
c
    

 recover the five frequency circles (circular 
𝑟
>
0.9
r>0.9 against the frequencies) with no group parameter supplied. This is the unsupervised naming test that run 1 failed.

E2. Persistence under pullback vs Euclidean. Report how many classes survive the noise band under each metric on Z113, ring, GPT-2. Prediction: pullback collapses geometric debris the head does not read.

E3. Invariance as quotient. Formalize the commutativity discovery. For each class fiber 
𝐹
𝑦
F
y
    

: compute 
𝐻
∗
(
𝐹
𝑦
)
H
∗
    

(F
y
    

); compare to 
𝐻
∗
H
∗
    

 of the expected orbit (circle for Z113 answer-fibers under the naive expectation; circle for rotated-MNIST digits). Where the fiber's homology is that of a quotient of the orbit ($S^1/\mathbb{Z}_2 = $ interval; point for full quotient), report the quotient group as a learned invariance. On rotated-MNIST, the trained network's fiber homology at late layers vs the untrained network's — the trained one quotients the rotation circle (absence = invariance). Report per layer: "invariance depth" = first layer where the orbit's 
𝛽
1
β
1
    

 drops to 0 in the trained net but not the untrained.

Pre-registered pass (E): E1 recovers 
≥
4
/
5
≥4/5 frequencies unsupervised at 
𝑟
>
0.9
r>0.9. E3 identifies the 
𝑍
2
Z
2
    

 quotient on Z113 fibers on all seeds (axis ratio test from run 1 as the witness) and an invariance depth on rotated-MNIST that differs from the untrained network.

Block F — P7: The formation vineyard

Diagnosis being answered: seeds learn different frequencies; birth order has no shared identity to compare.

F1. Vineyard. For each Z113 seed, the vines: each persistent 
𝐻
1
H
1
 class tracked across checkpoints (existing zigzag/matching), plotted as a curve in (checkpoint, normalized lifetime). Record per vine: onset checkpoint, peak lifetime, 
𝑠
𝑐
s
c
    

 at convergence, survives-to-end.

F2. Birth point process. Per seed, the set of (onset, peak lifetime) for surviving vines. Cross-seed comparison via 2-Wasserstein distance between these point sets. Birth density: histogram of onsets; report the peak's offset from the grokking checkpoint.

Pre-registered pass (F): birth-density peak within ±2 checkpoints of grokking on every seed; cross-seed Wasserstein distance between birth point processes less than the distance to a null point process (onsets uniformly resampled). Vine count and median lifetime agree across seeds within 30%.

Block G — Manifold score replacement

Diagnosis being answered: spectral-gap 
𝑚
m swings 0.99 ↔ 0.45 with neighborhood size; ResNet scores 0.00 while everything else behaves.

G1. ID plateau. TwoNN intrinsic dimension at 
𝑘
∈
{
5
,
10
,
20
,
40
,
80
,
160
}
k∈{5,10,20,40,80,160} on each layer cloud. 
𝑚
plateau
m
plateau
    

 = (length of the longest contiguous window in 
log

𝑘
logk where ID varies by 
<
15
%
<15%) / (total 
log

𝑘
logk range). Report alongside the old 
𝑚
m for every system.

G2. Stability check. Bootstrap 20 subsamples per cloud; report the CV of 
𝑚
plateau
m
plateau
    

. Pre-register: CV 
<
0.15
<0.15 on every system (the old 
𝑚
m would have failed this).

Pass (G): G2 holds; 
𝑚
plateau
m
plateau
    

 is nonzero on ResNet layers; it rank-orders systems the same way the run-1 qualitative behavior did (Z113 > ring > ResNet > GPT-2).

Execution order and pauses

Everything reuses cached activations/Jacobians/checkpoints; GPU is needed only for D3/D4 retraining (Z113, minutes each) and for CIFAR-100/extra OOD activations in B.

G (fast, needed to annotate everything) → E1/E2 (uses existing persistence) → PAUSE 1: report whether unsupervised naming works.
D1/D2 on the re-encoded model → PAUSE 2: report whether alignment catches what 
𝐻
1
H
1
 missed. This decides the paper's headline.
A1–A3 → B1–B3 → C1–C3 → PAUSE 3: the three failed cells re-scored.
D3–D5 (retraining) → E3 → F1–F2 → PAUSE 4.
Report: update the grid; CLAIMS.md; FINDINGS_*.md per block; figures at 300 dpi black-and-white-safe; the demo transcript re-run; all timings.

Pause reports use the run-1 format: plain-English paragraph first; pre-registered threshold and met/not; kill-test outcomes; anything contradicting the framing stated first; what is now claimable.

Deliverables

SPEC2.md (pre-registration, written before step 1). Updated grid with both runs side by side. FINDINGS_A..G.md. Figures: bipersistence heatmaps per class (clean vs scrambled); kernel-decomposition ROC panel; Morse–Smale complex rendered on a 2-D embedding of one Z113 layer with separatrices drawn; alignment-vs-checkpoint on the re-encoded model with the 
𝐻
1
H
1
 detector's silence overlaid; vineyards per seed; 
𝑚
plateau
m
plateau
    

 vs old 
𝑚
m. The one-minute demo re-run with the new P4/P5/P6 outputs.

Standing statement for the write-up

Run 1's contribution was the object — a network as a sequence of Riemannian manifolds — and the discovery that the pullback metric is the right metric for its topology. Run 2's contribution is the second layer: what a network knows is how its output function is organized on those manifolds. Morse–Smale cells, kernel directions, level-set alignment, and depth profiles are the mathematics of that organization. Score by the letter. Label every post-hoc. Report the evasion result as a theorem-shaped finding regardless of what else happens.
</pasted_content id="c38c">

you have max 7hrs
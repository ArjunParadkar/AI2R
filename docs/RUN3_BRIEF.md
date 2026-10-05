# Run 3 brief — verbatim, as pasted by Arjun 2026-09-20 ~13:35 CDT (budget: 3 h)



<pasted_content id="c38c">
WHITEBOX — Run 3: The Symmetry Group
Context

Two runs, fourteen pre-registered blocks, under ~/blindinglight. Read CLAIMS.md, FINDINGS_A..G.md, PAUSE_1..4.md, report/whitebox-report.html and the Run-2 report before writing a line. Everything cached — activations, Jacobians, checkpoints, the five grokked Z113 models (three seeds, two data fractions, decoy variants), the planted-shortcut models, the regenerated evasion model with its checkpoints, ring RNN ×3, ResNet-18 clean/scrambled with per-epoch checkpoints, rotated-MNIST trained/untrained, GPT-2 layer-6 probe — is reused. Nothing is retrained except where this brief says so.

The pattern the two runs found. Every positive was about symmetry and function: the symmetry-informed inventory, the pullback metric, the Z₂ quotient replicating on unseen models, rotation erased at depth, frequency structure predicting boundaries, circles born in a burst at grokking, and the evasion asymmetry (shape monitors are evaded; function monitors are not). Every failure was about reading shape at the output layer, where the geometry is the softmax. This run builds on the pattern.

The object. Pull the output metric back to input space: 
𝑔
(
𝑥
)
=
𝐽
(
𝑥
)
⊤
𝐽
(
𝑥
)
g(x)=J(x)
⊤
J(x), 
𝐽
J the Jacobian of the whole network at 
𝑥
x (or of the network up to layer 
ℓ
ℓ, for layer-wise versions 
𝑔
ℓ
g
ℓ
    

). 
𝑔
g is the network's own geometry of its inputs. Its null distribution (directions with 
𝑔
(
𝑣
,
𝑣
)
≈
0
g(v,v)≈0) is what the network is blind to — learned invariances. Its Killing fields (flows preserving 
𝑔
g) and, for finite input sets, its automorphism group (input permutations that are isometries of 
𝑔
g) are learned symmetries. A memorizer has a trivial symmetry group. A generalizer has a nontrivial one, and generalizes because the rule holds across the whole orbit.

Three definitions this run tests. Generalization = nontrivial learned symmetry group. Spurious structure = a nuisance direction the true task treats as null that the model treats as non-null (symmetry breaking). What it learned = the learned group plus the quotient structure across depth.

Neighbor to cite. LieGG (Moskalev et al., 2022) and the Lie-symmetry-discovery line extract Lie generators from trained networks. Cite it in every findings file. Our claims are the Riemannian formulation (null distribution + Killing fields of one metric), the depth profile of quotients, symmetry-group size as the definition of generalization, and symmetry breaking as the evasion-proof definition of spurious structure. Nothing else is claimed as first.

Rules (unchanged)

SPEC3.md before any computation, with every threshold and every operational choice fixed; name the unseen models per block in advance. Every positive gets its null. Post-hoc variants labelled, never upgrading a verdict. AMENDMENTS.md timestamped. Instruments verified on planted truth (analytic groups, known invariances) before touching a network. Baselines wherever a baseline exists. Pause points as marked. Timings. Never "complete."

Block S — Instruments (build and verify first)

S1. Input-pullback metric. metric.pullback_input(model, x, layer=None) → 
𝑔
(
𝑥
)
=
𝐽
⊤
𝐽
g(x)=J
⊤
J with 
𝐽
=
∂
𝜓
/
∂
𝑥
J=∂ψ/∂x (full network) or 
∂
ℎ
ℓ
/
∂
𝑥
∂h
ℓ
    

/∂x. Batched autograd. For image inputs, 
𝐽
J is 
𝑑
out
×
𝑑
in
d
out
    

×d
in
    

 with 
𝑑
in
=
3072
d
in
    

=3072 (CIFAR) or 
784
784; compute 
𝐽
⊤
𝐽
J
⊤
J implicitly via matrix-free products where needed; for spectra use randomized eigensolvers on 
𝐽
⊤
𝐽
J
⊤
J (top-
𝑘
k and bottom-
𝑘
k).

S2. Null distribution. symmetry.null_space(g, tol) → the eigenvectors of 
𝑔
(
𝑥
)
g(x) with eigenvalue below tol × top eigenvalue; null dimension 
𝑛
0
(
𝑥
)
n
0
    

(x); the invariance fraction 
𝜈
(
𝑥
)
=
𝑛
0
(
𝑥
)
/
𝑑
in
ν(x)=n
0
    

(x)/d
in
    

. Verified on a planted rotation-invariant function (analytic null field known).

S3. Finite automorphism group. For a finite input set with representation rows 
𝐸
∈
𝑅
𝑝
×
𝑑
E∈R
p×d
 (Z113 embedding; rotated-MNIST digit orbit; ring RNN heading samples): distance matrix 
𝐷
𝑖
𝑗
=
𝑑
𝑔
(
𝐸
𝑖
,
𝐸
𝑗
)
D
ij
    

=d
g
    

(E
i
    

,E
j
    

) under the pullback metric. symmetry.automorphisms(D, tol) → the group of permutations 
𝜎
σ with $\max_{ij}|D_{\sigma i,\sigma j}-D_{ij}| < $ tol × median
(
𝐷
)
(D), found by graph-automorphism search on the thresholded distance graph (networkx VF2 on the 
𝜖
ϵ-graph, or pynauty if installable in 10 min). Return generators and order. Verified on a planted cyclic embedding (must recover 
𝑍
𝑝
Z
p
    

) and a planted dihedral one (
𝐷
𝑝
D
p
    

).

S4. Killing fields (continuous). For continuous input manifolds: fit vector fields 
𝑉
V on a probe neighborhood minimizing the Killing residual 
∥
𝐿
𝑉
𝑔
∥
∥L
V
    

g∥ over a basis of candidate generators (translations, rotations, scalings, and a learned linear basis 
𝑉
(
𝑥
)
=
𝐴
𝑥
+
𝑏
V(x)=Ax+b). symmetry.killing(g_samples, basis) → the Killing algebra: generators with residual below tol; Killing dimension 
dim

𝑘
dimk. Verified on a planted 
𝑆
𝑂
(
2
)
SO(2)-invariant map (must return one generator, the rotation).

S5. Nuisance spectrum. symmetry.nuisance_spectrum(g, U_z) → eigenvalues of 
𝑔
g restricted to a nuisance subspace 
𝑈
𝑧
U
z
    

 (the hint-token direction(s); the rotation generator). The breaking score 
𝑏
𝑧
=
tr

(
𝑃
𝑈
𝑧
𝑔
𝑃
𝑈
𝑧
)
/
tr

(
𝑔
)
b
z
    

=tr(P
U
z
    

    

gP
U
z
    

    

)/tr(g): the fraction of the network's geometry that lives along a direction the task should ignore.

S6. Quotient depth. For a known input transformation 
𝑇
T (operand swap; rotation by 
𝜃
θ): at each layer 
ℓ
ℓ, 
𝑞
ℓ
(
𝑇
)
=
𝐸
𝑥
 
𝑑
𝑔
ℓ
(
ℎ
ℓ
(
𝑥
)
,
ℎ
ℓ
(
𝑇
𝑥
)
)
/
𝐸
𝑥
,
𝑥
′
𝑑
𝑔
ℓ
(
ℎ
ℓ
(
𝑥
)
,
ℎ
ℓ
(
𝑥
′
)
)
q
ℓ
    

(T)=E
x
    

d
g
ℓ
    

    

(h
ℓ
    

(x),h
ℓ
    

(Tx))/E
x,x
′
    

d
g
ℓ
    

    

(h
ℓ
    

(x),h
ℓ
    

(x
′
)) — how far 
𝑇
T moves a representation relative to typical distances. Invariance depth 
=
min

{
ℓ
:
𝑞
ℓ
(
𝑇
)
<
0.05
}
=min{ℓ:q
ℓ
    

(T)<0.05}. Verified on a network with a planted invariance at a known layer.

S7. The two after-the-fact positives, as instruments. depthprofile.connection_scale(F_y^\ell) → the Rips scale at which the class fiber becomes one piece, in units of the layer's median pairwise distance. manifold.id_profile(cloud, ks=[5,10,20,40,80,160]) → the 
I
D
(
𝑘
)
ID(k) curve and its profile class: plateau (slope of 
log

I
D
logID vs 
log

𝑘
logk within 
±
0.1
±0.1 over 
≥
3
≥3 consecutive 
𝑘
k), decay (slope 
<
−
0.2
<−0.2), growth (slope 
>
0.2
>0.2), mixed.

PAUSE 0 — instrument tests on planted truth; 200+ tests total green.

Block P1 — What it learned: the group, discovered

P1.1 Z113 automorphism group, nothing supplied. For each of the five grokked models: 
𝐷
D from the 113 embedding rows under 
𝑔
embed
g
embed
    

 (and under Euclidean, reported secondary). Run S3. Pre-register: the recovered group has order 
≥
113
≥113 and contains a cyclic generator of order 113 on 
≥
4
/
5
≥4/5 models (the network learned 
𝑍
113
Z
113
    

). If a reflection is also found, report 
𝐷
113
D
113
    

. Then derive the character decomposition from the recovered generator — the frequencies — and compare to the symmetry-informed inventory: pre-register 
≥
4
/
5
≥4/5 frequencies recovered at 
𝑟
>
0.9
r>0.9 with the group discovered, not supplied. This is the unsupervised naming Run 2 failed, by the right route.

P1.2 Scrambled control. Same on the Z113 memorizer. Pre-register: automorphism group order 
≤
2
≤2 (trivial or a single accidental symmetry).

P1.3 Ring RNN. Automorphisms of the heading-sampled representation under 
𝑔
g: pre-register recovery of a cyclic group of order 
≥
36
≥36 (the sampled circle's rotation symmetry) on 3/3 seeds; trivial on a shuffled-heading null.

P1.4 Rotated-MNIST, Killing route. On each layer, S4 with candidate basis {rotation generator, 2 translations, scaling}. Pre-register: the rotation generator is Killing (residual below tol) at conv layers for both trained and untrained networks (rotation is a symmetry of pixel geometry) and becomes null (S2, moves into the null distribution) at depth only in the trained network. Report invariance depth (S6) per digit.

P1.5 The quotient, as a general statement. Formalize Run 2's Z₂ result with S6: 
𝑞
ℓ
(
swap
)
q
ℓ
    

(swap) across layers on all five Z113 models. Pre-register 
𝑞
<
0.05
q<0.05 at the opened layer on 5/5 (already seen on 5; this scores the instrument), and report the layer at which swap-invariance appears.

PAUSE 1 — did the group get discovered?

Block P6 — Memorization vs understanding: the size of the group

P6.1 Killing dimension, clean vs scrambled. ResNet-18 clean and scrambled-label: S4 on a probe of 500 training images with basis {2 translations, rotation, 2 scalings, 6 affine, plus a 32-dim learned linear basis fit by least squares}. Report 
dim

𝑘
dimk per model at the input and at each residual stage. Pre-register: 
dim

𝑘
clean
≥
dim

𝑘
scr
+
2
dimk
clean
    

≥dimk
scr
    

+2 at the input-pullback level; label-permutation null → clean and scrambled indistinguishable.

P6.2 Invariance fraction, clean vs scrambled. S2 on the same probe: 
𝜈
ˉ
ν
ˉ
. Pre-register: 
𝜈
ˉ
clean
>
𝜈
ˉ
scr
+
0.1
ν
ˉ
clean
    

>
ν
ˉ
scr
    

+0.1 (a generalizer is blind to more of input space — it has quotiented nuisance; a memorizer keeps everything).

P6.3 Z113 automorphism order across training. On checkpoints of two clean seeds and the memorizer: order of the automorphism group (S3) at each checkpoint. Pre-register: order jumps from 
≤
2
≤2 to 
≥
113
≥113 within 
±
2
±2 checkpoints of grokking on both clean seeds; stays 
≤
2
≤2 on the memorizer throughout. This is the structural signal of generalization on the object that cannot be faked.

P6.4 The after-the-fact positives, pre-registered on fresh seeds. Train two new ResNet-18 seeds (clean + scrambled, 15 epochs — the only retraining in this run, ~30 min) and two new Z113 seeds (clean + scrambled). Connection-scale depth trend (S7): pre-register slope of connection scale vs depth 
<
−
0.3
<−0.3 per unit depth in generalizers, 
>
−
0.1
>−0.1 in memorizers, on all fresh seeds. ID-profile class (S7): pre-register plateau on grokked Z113, growth on Z113 memorizer, decay on ResNet penultimate, on all fresh seeds; bootstrap the class over 20 subsamples and require 
≥
18
/
20
≥18/20 agreement.

PAUSE 2 — is generalization the size of the group?

Block P3 — Spurious structure as symmetry breaking

P3.1 Breaking score vs the floor. On the planted-shortcut models (all seeds, all checkpoints) and the regenerated evasion model (all checkpoints): 
𝑏
𝑧
b
z
    

 (S5) with 
𝑈
𝑧
U
z
    

 = the hint-token embedding direction(s). Compare to the floor statistic (prediction dependence on hint on the decorrelated probe). Pre-register: rank correlation between 
𝑏
𝑧
b
z
    

 and the floor across all checkpoints 
>
0.9
>0.9; 
𝑏
𝑧
b
z
    

 fires (above clean mean + 3σ) on every checkpoint where the floor fires, including all seven evasion checkpoints where the 
𝐻
1
H
1
 detector was silent; zero firings on 153 clean and 102 decoy checkpoints. The decoy null is the hard one: the hint is represented but unused, so 
𝑏
𝑧
b
z
    

 must be low there while any shape statistic is high.

P3.2 Depth of breaking. 
𝑏
𝑧
b
z
    

 per layer: at which layer does the hint direction stop being null? Report; no threshold.

P3.3 The interrupt, geometric trigger, best penalty. Trigger on 
𝑏
𝑧
b
z
    

 firing at a newborn checkpoint; penalty = Run 1's presence-invariance KL at 
𝜆
=
3
λ=3, one round, no escalation (the only recipe that ever worked). All 5 seeds. Pre-register: decorrelated accuracy 
>
0.98
>0.98 on 5/5; 
𝑏
𝑧
b
z
    

 stays below threshold for 2× training after (no re-encoding); task accuracy within 1 point of clean.

P3.4 Unsupervised candidate directions (exploratory). Without 
𝑧
z: the top-10 eigenvectors of 
𝐸
𝑥
[
𝑔
(
𝑥
)
]
E
x
    

[g(x)] on the planted model; report the rank of the hint direction among them and its overlap. Also the earliest-born non-null direction across checkpoints. No pass criterion; report ranks.

PAUSE 3 — does symmetry breaking catch what shape missed?

Block P7 — Formation of the group

P7.1 Killing/automorphism trajectory. Already covered by P6.3 for Z113; add ring RNN: automorphism order across its checkpoints vs integration error. Pre-register: order reaches 
≥
36
≥36 within 2 checkpoints of integration error dropping below 5°.

P7.2 Invariance trajectory. 
𝑞
ℓ
(
swap
)
q
ℓ
    

(swap) across Z113 checkpoints: when does commutativity get learned relative to grokking? Report. Prediction (not a pass criterion): swap-invariance precedes grokking.

P7.3 Vineyard, revisited with the group. Run 2's fifth model had circles knocked down repeatedly after grokking. Compute automorphism order across its checkpoints: pre-register that the group is stable (order 
≥
113
≥113 throughout) even where the circles flicker — the symmetry is the invariant, the circle is its shadow.

Block P2 — The orbit explanation

P2.1. For each Z113 input 
(
𝑎
,
𝑏
)
(a,b): its orbit under the discovered group, its coset in the quotient by swap, and the fiber it lands in. Explanation string: "predicted 
𝑐
c; input lies on translation-orbit 
{
(
𝑎
+
𝑠
,
𝑏
−
𝑠
)
}
{(a+s,b−s)} and swap-coset 
{
(
𝑎
,
𝑏
)
,
(
𝑏
,
𝑎
)
}
{(a,b),(b,a)}, both mapped to the residue-
𝑐
c fiber." Verify: the orbit's members all receive the same prediction on 
≥
99
%
≥99% of inputs (a generalizer respects its own symmetry). On the memorizer, report the fraction — pre-register 
<
60
%
<60%.

P2.2. Keep Run 1's boundary result (frequency inventory predicts geodesic destinations) and restate it with the discovered group: the nearest competing residue is the one reached by the smallest group element that nearly fixes all learned characters. Recompute with the discovered frequencies. Pre-register the same 10× chance.

Block P4 / P5 — Descriptive only, no benchmark

P4.1. Report the null distribution (dimension, top null directions rendered as images for ResNet, as token-space directions for GPT-2) per system. This is "what it cannot see." No detector, no AUROC.

P5.1. Report the top eigen-directions of 
𝑔
g (the steep directions) per system and confirm on one system that adversarial perturbation directions align with them (cosine 
>
0.5
>0.5) — a known result, cited, reported as a consistency check only.

Deliverables

SPEC3.md; both prior grids plus Run 3 in one figure; FINDINGS_S,P1,P6,P3,P7,P2.md; figures: the discovered Z113 group (Cayley-graph or distance-graph rendering with the cyclic generator highlighted), automorphism order vs checkpoint with grokking marked, Killing dimension clean vs scrambled by depth, invariance-depth curves for rotated-MNIST trained vs untrained, 
𝑏
𝑧
b
z
    

 vs floor statistic on the evasion model with the 
𝐻
1
H
1
 silence overlaid, the fresh-seed connection-scale trend and ID profiles; whiteboxtopo/symmetry/ and metric/ packages with tests; the demo re-run: open() now prints the discovered group, its order, the invariance depths, the Killing dimension, and the breaking score.

Standing statement

Run 1 found the object (a network as Riemannian manifolds; the pullback metric). Run 2 found that shape can be evaded and function cannot. Run 3 tests whether what a network knows is a symmetry group — discovered, not supplied; its size the definition of generalization; its breaking the definition of spurious structure. Score by the letter. Cite LieGG everywhere. If the group is not discoverable on Z113 without being supplied, say so first.
</pasted_content id="c38c">

 run full thing, u have 3 hours
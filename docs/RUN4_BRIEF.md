# Run 4 brief — verbatim, as pasted by Arjun 2026-09-20 ~15:55 CDT



<pasted_content id="c38c">
Ran a command, used a tool
Read it. Three runs in, the picture is actually clear now — the post-hoc results are the real ones, and they keep replicating. Here's what the data says about understanding vs memorization, then the Run 4 prompt built to turn it into passes.

What three runs say a memorizer is: it keys on every input direction (decoy models: a third of the geometry on each slot, hint included, until grokking collapses it to 0.0004); it breaks the symmetries its own architecture offers (swap-invariance is present at init, broken during memorization to 0.2–0.58, restored at grokking); it forms no group (no 113-cycle in 73 memorizer checkpoints vs 5/5 generalizers within ±1 checkpoint of grokking); its class fibers never tighten with depth (flat vs slopes of −0.5 in all four fresh generalizers); its metric is spread rather than concentrated (participation ratio 7.8 vs 5.7). Five independent instruments, one story: memorization is a network that refuses to quotient. Generalization is a network that has collapsed its geometry onto the task's quotient — blind to what the task doesn't need, keeping its architecture's symmetries, and organizing what remains into a group action. That's the definition. It's recursive in exactly the sense you meant: each layer quotients the previous one, and understanding is the tower of quotients reaching the task.

Why the ResNet memorizer half failed every time: the "memorizers" weren't. Fifteen epochs on random labels hit 56–59% train accuracy. You can't measure what a memorizer does with a model that hasn't memorized. That's a specimen bug, not a theory failure, and it's the first thing Run 4 fixes.

WHITEBOX — Run 4: The Tower of Quotients
Context
Three runs under ~/blindinglight. Read CLAIMS.md, all FINDINGS_*, all PAUSE*, AMENDMENTS.md (A-1…A-12), and the three reports. Everything cached is reused.

What is now known, from replicated evidence across runs:

The pullback metric is the right metric for topology (ring blob → circle, 
r
=
0.99
r=0.99).
Under the output-pullback metric, a 113-cycle generator is recoverable from outputs alone on 5/5 generalizers and no memorizer, appearing within ±1 checkpoint of grokking, present in 97–100% of later checkpoints, stable where circles flicker; its characters recover 5,4,4,4,4 of five frequencies at 
r
=
0.99
r=0.99. Post-hoc rule (mean-norm full-cycle), fixed at 14:17 before fresh seeds finished. Needs pre-registration on fresh seeds. That is P1 and P7.
Swap-invariance is architectural (present at init, 
q
=
0.01
q=0.01), broken during memorization (
q
=
0.2
q=0.2–0.58), restored at grokking. Generalizers keep a symmetry the architecture offers; memorizers break it.
Class-fiber connection scale falls with depth in 4/4 fresh generalizers (slopes −0.49 to −0.57) and is flat in Z113 memorizers. ID profile: plateau on grokked Z113 (20/20), decay on ResNet penultimate (20/20).
On decoy models the hint slot's share of the input-pullback trace is 0.44–0.48 through memorization and collapses to 0.0004 at grokking. A memorizer keys on every token.
On planted-shortcut models the hint is the top eigen-direction of 
g
ˉ
g
ˉ

  (95–96% of trace; <1% on clean). Unsupervised lead, no null yet.
Clean ResNet lowers Killing residuals 0.7× vs memorizer and untrained (which tie); clean metric is most anisotropic (PR 5.7 vs 7.8 vs 9.5).
Orbit explanation holds exactly (1.000 ×5 vs 0.000).
Symmetry breaking ≠ spurious structure (memorization breaks symmetry too). The functional floor stays the monitor of record.
Entrywise isometry is the wrong notion; trained networks are equivariant in distribution.
The definition this run tests. A network understands when it has quotiented: its input-pullback geometry has collapsed onto the task-relevant subspace, it preserves its architecture's symmetries, and a group generator is recoverable from its outputs. A network memorizes when it refuses to quotient: geometry spread across all input directions, architectural symmetries broken, no group. Depth is the recursion: each layer quotients the last, and the connection scale of class fibers falling with depth is the tower being built.

Rules
SPEC4.md before computation. Every threshold fixed. Unseen models named in advance. Every instrument verified on planted truth. Post-hoc labelled, never upgrading. Baselines where they exist. Cite LieGG in every findings file. Pauses as marked. Never "complete."

Specimen fix (first, non-negotiable)
True memorizers. ResNet-18 on a 5,000-image CIFAR-10 subset with random labels, no augmentation, weight decay 0, trained until train accuracy ≥ 99% (expect 80–150 epochs; ~40 min); two seeds. Matched clean models on the same 5,000 images with true labels to ≥ 99% train accuracy, two seeds. Checkpoints every 5 epochs. Z113: two fresh clean seeds and two fresh scrambled-label seeds trained to 100% train accuracy. Every P6 cell below is scored on these matched pairs; the old under-trained ResNet "memorizers" are excluded from all verdicts.

Sanity gate before scoring anything: train accuracy ≥ 99% on all memorizers; test accuracy ≤ 15% (ResNet) / ≤ 2% (Z113). If a memorizer fails the gate it is not a memorizer and is dropped.

Block P6 — Understanding vs memorization, five instruments, one definition
Each instrument gets its own pre-registered threshold; the block passes if ≥ 4/5 instruments pass on all matched pairs.

P6.1 Metric concentration. 
g
ˉ
=
E
x
[
J
(
x
)
⊤
J
(
x
)
]
g
ˉ

 =E 
x

 [J(x) 
⊤
 J(x)] over 500 training inputs (implicit / randomized eigensolver). Participation ratio 
P
R
=
(
∑
λ
)
2
/
∑
λ
2
PR=(∑λ) 
2
 /∑λ 
2
 . Concentration 
C
=
1
−
P
R
/
P
R
untrained
C=1−PR/PR 
untrained

 . Pre-register: 
C
clean
−
C
mem
>
0.25
C 
clean

 −C 
mem

 >0.25 on every matched pair (ResNet and Z113). Null: label-permuted clean model retrained is the memorizer — this is the memorizer, so the null is the untrained network, which must sit at 
C
≈
0
C≈0.

P6.2 Slot collapse (Z113 only). Trace share of 
g
ˉ
g
ˉ

  on each input slot (a, b, and hint where present). Pre-register: clean models put ≥ 0.9 of trace on the two task slots combined with neither slot below 0.3; memorizers spread with the max-slot share ≤ 0.5 on the three-slot decoy variant. On decoys, report the collapse trajectory (0.44 → 0.0004 replicated on fresh seeds within ±2 checkpoints of grokking).

P6.3 Architectural symmetry preservation. Z113: 
q
(
swap
)
q(swap) at the opened layer. Pre-register: clean 
<
0.05
<0.05, memorizer 
>
0.3
>0.3, on every matched pair. CNN: shift consistency — fraction of training-set predictions unchanged under ±2 px circular shifts (the conv architecture's own symmetry). Pre-register: clean ≥ 0.9, true memorizer ≤ 0.6. Baseline to cite: consistency-regularization literature; this is used as a memorization diagnostic on training data, which is the new use.

P6.4 Connection-scale depth slope. S7 from Run 3, on the true memorizers. Pre-register: slope 
<
−
0.3
<−0.3 on all clean, 
>
−
0.1
>−0.1 on all true memorizers.

P6.5 Group presence. Mean-norm full-cycle rule (Run 3 post-hoc, now registered as written in A-12): a 113-cycle among mean-accepted permutations. Pre-register: found on all fresh clean Z113, absent on all fresh Z113 memorizers. (CNN analogue omitted — no finite group to find.)

P6 synthesis figure: all five instruments as a radar/bar per model; clean and memorizer should separate on every axis.

PAUSE 1 — is memorization the refusal to quotient?

Block P1 — What it learned: the group, discovered, pre-registered
P1.1 Full-cycle discovery on fresh seeds. On the four fresh grokked Z113 models (named unseen): mean-norm statistical isometry test (δ = 0.06 of median distance, as A-12), full 113-cycle among accepted candidates. Pre-register: found on 4/4; absent on both fresh memorizers; on ≥ 3/4 the generator is translation by a unit 
u
u and the Fourier transform of the embedding ordered by 
u
u recovers ≥ 4/5 frequencies at 
r
>
0.9
r>0.9 with no group parameter supplied. This is unsupervised discovery of the group and its representation from outputs alone.

P1.2 Statistical equivariance, defined. Write the definition in SPEC4: σ is a δ-symmetry of 
f
f if 
E
i
j
∣
D
σ
i
σ
j
−
D
i
j
∣
/
m
e
d
i
a
n
(
D
)
<
δ
E 
ij

 ∣D 
σiσj

 −D 
ij

 ∣/median(D)<δ. The learned group 
G
δ
G 
δ

  is the group generated by accepted permutations. Verify on planted truth: a Fourier embedding with 2% noise must return 
Z
113
Z 
113

  (or 
D
113
D 
113

 ) at δ = 0.06 and nothing at δ = 0.01; an untrained model must return a group of order ≤ 2. This replaces entrywise isometry.

P1.3 The tower of quotients (P1 as a structure). For each system and each known transformation 
T
∈
T∈ {swap, translation by 
u
u, hint change, ±2 px shift, rotation}: 
q
ℓ
(
T
)
q 
ℓ

 (T) across layers. The tower is the sequence of layers at which each 
T
T becomes null (
q
ℓ
<
0.05
q 
ℓ

 <0.05). Report as one figure per system: rows = transformations, columns = layers, cell = 
q
q. Pre-register only the structural claim: on clean Z113, hint-change (decoy) becomes null at or before the layer where the group appears; on memorizers, nothing becomes null. This is P1 stated as what has been quotiented, in what order, at what depth.

P1.4 Orbit explanation — carry forward (holds exactly), computed with the discovered generator.

PAUSE 2 — is the group discoverable when registered in advance?

Block P7 — Formation, three signals, pre-registered
P7.1 Generator emergence. On all fresh clean Z113 checkpoints: first checkpoint at which the 113-cycle is found. Pre-register: within ±1 checkpoint of grokking on 4/4; present in ≥ 95% of subsequent checkpoints; never in any memorizer checkpoint.

P7.2 Symmetry restoration. 
q
(
swap
)
q(swap) across checkpoints. Pre-register on 4/4 fresh clean: 
q
<
0.05
q<0.05 at init, peak 
q
>
0.2
q>0.2 during the memorization phase (train acc = 1, test acc < 0.1), and 
q
<
0.05
q<0.05 again within 2 checkpoints of grokking. Memorizers: 
q
q never returns below 0.2.

P7.3 Slot collapse timing. On fresh decoy seeds (train two): hint-slot trace share > 0.3 throughout memorization, < 0.01 within ±2 checkpoints of grokking.

P7.4 The formation figure. Test accuracy, generator presence, 
q
(
swap
)
q(swap), hint-slot share, and connection-scale slope on one time axis per seed. Three structural signals of generalization, all appearing at grokking, all invisible to train loss.

PAUSE 3 — do the three signals coincide at grokking?

Block P3 — The unsupervised lead, with a null that can fail
P3.1 Dominant-direction rule. Unsupervised: flag any single input slot whose trace share of 
g
ˉ
g
ˉ

  exceeds 0.8. Pre-register: fires on both planted-shortcut models (hint share 0.95–0.96 replicated on 2 fresh planted seeds); does not fire on clean models (max slot share < 0.6 — a and b share the geometry); does not fire on grokked decoys (hint < 0.01). The null that can fail: construct a task where one slot is legitimately dominant — Z113 with 
c
=
a
 
m
o
d
 
113
c=amod113 (b irrelevant by design). Pre-register that the rule fires there too, and document this as the rule's stated limitation: it detects dominance, and dominance must be checked against the task. Report honestly.

P3.2 The interrupt, finally scored. Run 1's presence-invariance penalty at λ = 3, one round, functional-floor trigger at first firing, on an idle GPU (sequential, nothing else running). 5 seeds. Pre-register as before: decorrelated accuracy > 0.98 on 5/5; no re-firing over 2× training; task accuracy within 1 point of clean.

P3.3 Drop symmetry-breaking = spurious. Record in FINDINGS_P3 with the decoy numbers as the reason.

Block Q — Descriptive P4/P5 (no benchmark)
Report the null distribution and top eigen-directions of 
g
ˉ
g
ˉ

  per system as images/token directions. Carry forward.

Deliverables
SPEC4.md; four-run grid; FINDINGS_P6,P1,P7,P3.md; figures: five-instrument P6 separation on matched pairs; the discovered group on fresh seeds; the tower-of-quotients heatmap per system; the formation figure per seed; dominant-direction rule on planted/clean/decoy/legit-dominant; interrupt outcomes. Demo: open() prints the group and its generator, the tower, the five P6 instruments with a memorization verdict, and the dominant-direction flag.

Standing statement
Runs 1–3 found the objects, then found which of them survive scrutiny. Run 4 registers what replicated and fixes the specimens that made memorization unmeasurable. If the five instruments separate true memorizers from generalizers on matched pairs, understanding versus memorization has a mathematical definition: the refusal, or the achievement, of the quotient. Score by the letter. Say first if the group
</pasted_content id="c38c">

 lets get going, get this done quick
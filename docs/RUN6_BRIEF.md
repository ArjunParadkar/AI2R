# WHITEBOX — Run 6: Orbit-Closure  (brief as received 2026-09-21 ~22:48 CDT; equations de-duplicated from the paste, wording unchanged)

Preamble (verbatim). Why P4 and P5 kept failing: every instrument read the last layer, where geometry is softmax. Runs 3–5 established what a
network actually knows: a group acting on its inputs, recoverable from outputs, with orbits on which predictions transform coherently (1.000
on generalizers, 0.000 on memorizers). So stop measuring distance to a manifold. Measure whether the network's own learned rule holds at this input.
Knowledge is orbit-closure. A network knows an input x iff its learned group acts coherently on it: for every generator g, f(g·x) = ρ(g) f(x).
Orbit coherence κ(x) = the fraction of the orbit on which that holds. Failure is equivariance defect: ε(x) = max_g ‖f(g·x) − ρ(g) f(x)‖.
Neighbour: Golan & El-Yaniv 2018; the new thing is that the transformations come from the model, and that one definition covers knowledge,
failure and generalization. Seven problems, one object.

Context. Five runs under ~/blindinglight. Read SPEC5.md, FINDINGS5_*, CLAIMS.md, AMENDMENTS.md (A-14). Reuse everything cached. Three things:
close P6, attack P4 and P5 with the orbit theory, write the unified definition into the library. Standing rules unchanged.
Neighbours to cite in every findings file: Golan & El-Yaniv 2018; Hendrycks & Gimpel 2017 (max-softmax); Liang et al. 2018 (ODIN); Azulay &
Weiss 2019, Zhang 2019; LieGG. The novelty claimed is model-discovered transformations and the unification, not transformation-consistency.

Specimens (name before training). Z113: 3 fresh clean (seeds 12–14), 3 fresh true memorizers (15–17), 2 fresh decoys (12–13). ResNet-18: 2 fresh
clean with augmentation (5–6), 3 fresh true memorizers (5–7). Sanity gate as Run 5. Ceilings for P6 set from all four prior memorizers per instrument.

Block P6 — close the definition. P6.d re-registered: memorizer iff hint > 0.10 or max(s_a, s_b) < 0.40; clean iff s_a, s_b ≥ 0.40, hint ≤ 0.05.
Pass 5/5 each side. Untrained null reads memorizer-side. P6.c null-subtracted depth slope Δ = slope − slope_label-permuted: clean Δ < −0.25;
memorizer Δ > −0.10. Register that the ResNet clean margin is thin (≈ −0.11 in Run 5); if it fails there, the instrument is transformer-only
and that is reported. P7.d persistence at 0.94, 3/3.

Block K — P4. K.1 knowledge.orbit_coherence(model, x, group): G (Z113: the discovered generator u and swap; ResNet: shifts ±{1,2,3} px and
horizontal flip; both directions of every generator), κ(x) = (1/|G|) Σ_g 1[argmax f(gx) = ρ(g) argmax f(x)], κ_s = (1/|G|) Σ_g cos(f(gx), ρ(g) f(x)).
Z113: ρ = induced relabelling of the answer; ResNet: ρ = id. K.2 Z113 boundary: in = held-out valid pairs; unknown = one out-of-vocabulary
token (generator applied to the valid slot), and pairs from a different modulus fed to the p = 113 model. Pre-register AUROC(κ) ≥ 0.95 on both;
softmax confidence and entropy reported. K.3 ResNet: in = CIFAR-10 test; unknown = SVHN, CIFAR-100, Gaussian noise. Pre-register AUROC(κ_s)
exceeds max-softmax, entropy and ODIN on SVHN and CIFAR-100 on both clean seeds, and matches or exceeds Golan-style 72-transformation
consistency. Report all four baselines regardless. K.4 memorizer control: memorizer median κ on its own training set < 0.3; clean > 0.95.
K.5 knowledge map: κ histogram in vs unknown; lowest-κ in-distribution inputs rendered.

Block F — P5. F.1 ε(x) = max_g ‖f(gx) − ρ(g) f(x)‖₂ / ‖f(x)‖₂ and ε̄ (mean over G). Failure radius as in Runs 1–5 (Gaussian six radii; PGD).
F.2 Spearman(ε, failure radius) more negative than softmax margin's on Z113 (3 clean) and ResNet (2 clean); if not, report the gap.
Additional: the perturbation that flips a prediction aligns with ∇_x ε at cosine > 0.5 on ≥ 70% of inputs. F.3 nulls: shuffled ε; ε with
random input transformations (not the group) — must be worse than the group version.

Block U — unified. open() reports P1–P7 for any model. Registered consistency test: on every clean model Spearman(κ, ε) < −0.6; on
memorizers κ is flat-low and the test is vacuous — report that.

Deliverables. SPEC6.md; FINDINGS6_P6, K, F, U.md; six-run grid; figures (four-instrument closure; κ ROC panel with baselines; ε vs failure
radius with softmax margin; knowledge map; κ-vs-ε scatter); demo transcript with the seven-cell report on one fresh model. Then stop and write.
Standing statement. If orbit coherence beats the softmax family and Golan's designer transformations on unseen models, the seven problems
have one theory. If it does not, P4 and P5 stay descriptive and the theory is stated for five. Score by the letter.
User instruction: "do this whole thing...go".

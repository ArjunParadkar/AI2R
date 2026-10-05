# WHITEBOX — Run 5: Registration  (brief as received 2026-09-21 ~13:05 CDT; equations de-duplicated from the paste, wording unchanged)

Context. Four runs under ~/blindinglight. Read SPEC4.md, AMENDMENTS.md (through A-13), FINDINGS4_P6, P1, P7, P3.md, PAUSE4_1..3.md, CLAIMS.md.
Everything cached is reused. This is a confirmation run, not a discovery run. Every quantity below has been observed. Run 5 registers
thresholds from Run 4's matched pairs and scores them on models trained after this brief is written. Nothing new is proposed. If it confirms,
P6 and P7 move from "found" to "registered." If it does not, they stay "found," and we say so.
Standing rules unchanged. SPEC5.md before compute, with every threshold and every unseen model named. Nulls on every positive. Post-hoc
labelled, never upgrading. Timestamped amendments. Never "complete." Cite LieGG; for shift consistency cite Zhang 2019 and Azulay & Weiss 2019.

Specimens (train first, name in SPEC5 before training)
- Z113: 3 fresh clean seeds (new seeds, not used in Runs 1–4); 2 fresh scrambled-label memorizers with uninformative hint slot (the Run 4
  recipe). Checkpoints every 2%. Sanity gate: memorizers 100% train, ≤2% test.
- ResNet-18, matched pairs, WITH augmentation for clean: 2 clean models on a 5,000-image CIFAR-10 subset with standard augmentation (random
  crop + flip) to ≥99% train; 2 true memorizers on the same images with random labels, no augmentation, no weight decay, to ≥99% train and
  chance test. Sanity gate as Run 4. Augmentation on the clean side is the fix for the shift-consistency clause: Run 4's clean ResNets were
  trained without it and scored 0.68.
- Decoys: 2 fresh decoy Z113 seeds (hint present, uninformative) for P7.3.

Block P6 — Four instruments, thresholds from Run 4, scored on the above. Concentration is removed from the definition. Record why in
FINDINGS5_P6.md: memorizers concentrate ḡ as much as generalizers (0.64–0.79 vs 0.54–0.71); the Run 3 number came from an under-trained model;
the untrained null itself was unstable (+0.24 / −0.17). Thresholds are set at the midpoint of Run 4's observed gap on each instrument, rounded
conservatively toward the harder side. Pre-register these exactly:

| Instrument | Run 4 generalizers | Run 4 memorizers | Registered threshold |
|---|---|---|---|
| P6.a swap quotient q(swap) at resid_mid (Z113) | 0.003–0.005 | 0.56–0.58 | clean < 0.10; memorizer > 0.30 |
| P6.b shift consistency, ±2 px (ResNet) | 0.67–0.68 (no aug) | 0.17–0.19 | clean ≥ 0.75 (with aug); memorizer ≤ 0.35 |
| P6.c depth slope ∂ε*/∂ℓ | −0.49 to −0.59 | 0.00 to −0.21 | clean < −0.35; memorizer > −0.30 |
| P6.d slot collapse (Z113) | task slots 0.49/0.50, hint 0.01 | max slot 0.38–0.39 | clean: two task slots each ≥ 0.40, hint ≤ 0.05; memorizer: max slot ≤ 0.42 |
| P6.e group presence (Z113) | 4/4 | 0/2 | found on all clean; none on memorizers |

Block pass: all applicable instruments pass on every matched pair (Z113: a, c, d, e; ResNet: b, c). Report every instrument regardless.
Nulls: untrained network for a, b, d; label-permutation for c; the memorizers themselves for e.
The definition, as registered for the write-up: a network understands a task when (i) its input geometry concentrates onto the task's
directions — slot collapse, not participation ratio; (ii) it preserves the symmetries its architecture offers; (iii) its class fibers tighten
with depth; (iv) a generator of the task's group is recoverable from its outputs. Memorization is the negation of (i)–(iv). Four instruments,
one definition, no held-out data.

Block P7 — The generator as a leading indicator. Run 4 found the generator at 17%, 70%, 41%, 93% test accuracy — 4, 1, 3, 0 checkpoints
before the largest accuracy jump. Register the lead, not coincidence.
- P7.a Lead time. On all 3 fresh clean Z113 seeds: first checkpoint at which a full 113-cycle is accepted, minus the checkpoint of the largest
  test-accuracy jump. Pre-register: lead ≥ 0 on 3/3 (generator at or before the jump); generator never accepted on either memorizer in
  25,000 steps. Report lead in checkpoints and the test accuracy at first acceptance.
- P7.b Symmetry restoration. q(swap) returns below 0.05 within 3 checkpoints of the jump on 3/3; memorizers never below 0.30. (Run 4 had one
  seed without a memorization phase — if a seed groks immediately, score restoration as "present at jump" and say so.)
- P7.c Hint-slot collapse on decoys. Register what Run 4 observed: steady decline through memorization, then collapse below 0.01 within +4
  checkpoints of the jump on 2/2 decoys.
- P7.d Persistence. Generator present on ≥ 95% of checkpoints after first acceptance, 3/3.
Figure: the formation figure per seed with the lead marked.

Block T — Finite tower (replaces the linearized one). Both Run 4 tower artifacts were "infinitesimal misses finite." Replace derivatives with
finite changes: q_ℓ(T) = E_x d_g^ℓ(h_ℓ(x), h_ℓ(Tx)) / E_{x,x'} d_g^ℓ(h_ℓ(x), h_ℓ(x')), computed with Tx as an actual transformed input (hint
token changed, not perturbed; operands swapped; token translated by u), and d_g^ℓ as the finite g-length of the secant, not a
Jacobian-vector product. Pre-register: on decoys, hint-change becomes null (q < 0.05) by the layer where the group appears; translation by u
is not null at any layer (q > 0.20 — it is a symmetry, the representation moves along the orbit); on memorizers, hint-change is not null
anywhere (q > 0.15). Deliver the tower table per model and the tower figure.

What is NOT in Run 5. Group discovery (P1.1), planted-truth verification (P1.2), orbit explanation (P1.4), the dominance rule (P3.1), and the
interrupt (P3.2) are done — registered passes on unseen seeds in Run 4. Do not rerun. Write them up from Run 4's files. No new instruments.
No new problems. No P4/P5.

Deliverables. SPEC5.md (before any training). FINDINGS5_P6.md, FINDINGS5_P7.md, FINDINGS5_T.md. CLAIMS.md updated to the five-run grid.
Figures: four-instrument separation on the new matched pairs (solid vs hatched, as Run 4); formation with lead marked, per seed; finite tower
per model. grid_five_runs.png. Demo transcript re-run with the four-instrument verdict and the lead time in the report.
Pause points. PAUSE 1 after specimens pass the sanity gate — confirm the augmented clean ResNets reach ≥99% train and report their shift
consistency before scoring anything else. PAUSE 2 after P6. PAUSE 3 after P7 and T.
Standing statement. Runs 1–3 found the objects and discarded the ones that could be faked. Run 4 fixed the specimens and registered the group.
Run 5 registers the definition of understanding and the leading indicator of its arrival, on models that do not yet exist. Score by the
letter. If a threshold set from Run 4 fails on Run 5's seeds, that is the result — report the gap between them and stop.

User instruction with the brief: "finish whole thing" → pause documents are written, the run does not wait at them.

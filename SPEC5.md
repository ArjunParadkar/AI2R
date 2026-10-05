# SPEC5 — Run 5 pre-registration ("Registration")

Written 2026-09-21 13:11–13:13 CDT (file mtime 13:12:51; specimens launched 13:14), **before any Run-5 model is trained** and before any Run-5 code has touched a model. Brief:
`docs/RUN5_BRIEF.md`. Authors: Arjun Paradkar & Reva Paradkar (independent); run executed by Claude. Rules as runs 1–4: scored by the letter;
every positive its null; post-hoc labelled, never upgrading; `AMENDMENTS.md` continues at A-14; never "complete".
Neighbouring work, cited in every findings file: LieGG (Moskalev, Sepliarskaia, Sosnovik, Smeulders, NeurIPS 2022). Shift consistency:
Zhang 2019 ("Making convolutional networks shift-invariant again"); Azulay & Weiss 2019 ("Why do deep convolutional networks generalize so
poorly to small image transformations?").

This is a confirmation run. The thresholds below are the brief's, copied exactly. Nothing in this file moves a threshold.

## What Run 4's files say about the brief, stated before anything is computed
1. **Concentration.** Run-4 C on ḡ: Z113 clean 0.71, 0.54, 0.57, 0.54; Z113 memorisers 0.69, 0.79; ResNet clean 0.70, 0.61; ResNet memorisers
   0.65, 0.64. So the memoriser range is 0.64–0.79 and the generaliser range 0.54–0.71, as the brief says. Second-untrained-seed null on
   ResNet: +0.235 / −0.170. Concentration is removed from the definition; it is still *computed and reported* (no criterion).
2. **P6.c, a near miss already on file.** The brief quotes generaliser slopes −0.49 to −0.59. Those are the four matched pairs. Run 4's other
   two clean Z113 seeds read −0.555 (s7) and **−0.399 (s8)**. The registered clean threshold (< −0.35) is 0.05 away from a value already seen.
3. **Nulls that are known in advance not to separate.** An untrained network *has* its architecture's symmetries: Run 4 measured untrained
   q(swap) = 0.010–0.019 and untrained shift consistency 0.994 / 0.903. So on instruments a and b the untrained network will sit on the
   *clean* side of the threshold, and on d (slot shares 0.333/0.333/0.333, max ≤ 0.42) on the *memoriser* side. The brief's definition
   already implies this — clause (ii) is "preserves", not "acquires". The nulls are reported as measured; an untrained network is expected to
   satisfy (ii) and fail (i), (iii), (iv). This is said now so it cannot be presented later as a discovery or hidden as an embarrassment.
4. **P7.a, what Run 4 actually measured.** Run-4 sweeps were sparse outside ±4 checkpoints of the jump. Seed 5's first acceptance (ckpt 12)
   is the edge of its dense window (previous swept ckpt: 9, not accepted), so its "lead 4" is "lead 4–6". Run 5 sweeps **every checkpoint**,
   so leads are exact. Run 3's five seeds gave leads 0, −1, 0, +1, 0 in the brief's sign convention (one seed accepted one checkpoint
   *after* the jump). A 3/3 "lead ≥ 0" is therefore not a foregone conclusion.
5. **P7.b memoriser clause.** q(swap) is 0.01–0.02 at initialisation for every model (point 3). "Memorizers never below 0.30" is scored on
   checkpoints with train accuracy ≥ 0.99 (Run 4: minima 0.45, 0.48 on that set).
6. **Block T, translation.** The brief's threshold (q > 0.20 at every layer) is read against Run 4's translation T: (a, b) → (a+u, b−u), which
   is the transformation those Run-4 numbers came from. On file already: Run-4 **decoy_s6 reads 0.14 at resid_post and logits**, and the
   logits-layer q is a finite quantity in both runs (ψ is the identity there), so that cell is expected to reproduce. The registered
   threshold is kept as the brief wrote it.
7. **Block T, "the layer where the group appears".** Runs 1–4 located the frequency circles at resid_mid (the first layer after the two
   operands have been mixed). Fixed now: "by the layer where the group appears" = **at resid_mid and at every later layer**.
8. **An identity that constrains the finite tower.** For a real input x, ψ_ℓ(h_ℓ(x)) = f(x) at every layer, so the plain output-secant
   ‖ψ_ℓ(h_ℓ(Tx)) − ψ_ℓ(h_ℓ(x))‖ is the *same number at every layer* and cannot make a tower. The finite g-length used here is the length,
   measured at the output, of the straight segment between the two activations *in layer ℓ's own space* (definition below). It is
   Jacobian-free, it is ≥ the output secant, and it depends on the layer because the segment does.

## Specimens (all UNSEEN; none of these seeds exists under `runs/` at the time of writing)
Z113 (`systems/z113.py`, 25,000 steps, checkpoint every 500 steps = 2%): clean seeds **9, 10, 11** (frac 0.3); memorisers = `scrambled_decoy`
seeds **9, 10** (frac 0.3; random labels + uninformative hint tokens, the Run-4 recipe); decoys seeds **9, 10** (frac 0.5).
Matched Z113 pairs: (clean_9, mem_9), (clean_10, mem_10); clean_11 is scored on the clean-side thresholds.
ResNet-18 on the Run-4 fixed 5,000-image subset (`SUBSET`, rng 12345), Adam 1e-3, weight decay 0, batch 128: clean seeds **3, 4 WITH
augmentation** (4-px zero-pad random crop + horizontal flip, fresh per batch), trained until accuracy on the *un-augmented* training images
≥ 0.99 (cap 300 epochs); memorisers = `scrambled` seeds **3, 4**, no augmentation, until train ≥ 0.99 (cap 200). Pairs: (clean_s, scrambled_s).
Run directories: `runs/r5_z113_{clean,scrambled_decoy,decoy}_s*`, `runs/r5_resnet_{cleanaug,scrambled}_s*`.
**Sanity gate.** Z113 memorisers: train = 1.00 (≥ 0.999), test ≤ 0.02. ResNet memorisers: train ≥ 0.99, test ≤ 0.15. Clean models (both
systems): train ≥ 0.99; Z113 clean and decoys must also reach test ≥ 0.99 within 25,000 steps. A model failing the gate is reported and
dropped; **a dropped model fails every "n/n" clause it belongs to** (a registered specimen that does not exist cannot pass).
**PAUSE 1** — gate table + shift consistency of the augmented clean ResNets, before anything else is scored.

## P6 — four instruments + group presence. Code unchanged from Run 4 (`analysis/run4_z113.py`, `run4_resnet.py`); final checkpoints.
| | instrument (exactly as Run 4 computed it) | clean passes iff | memoriser passes iff | null |
|---|---|---|---|---|
| P6.a | q(swap), linearised pullback at resid_mid, 1000 pairs (Z113) | < 0.10 | > 0.30 | same model at step 0 |
| P6.b | shift consistency: fraction of the 5,000 training images whose prediction is unchanged, mean of the four circular shifts (±2,0),(0,±2) (ResNet) | ≥ 0.75 | ≤ 0.35 | same model at init |
| P6.c | connection-scale depth slope (run-3 S7; Z113: embed/resid_mid/resid_post on the training set; ResNet: 5 layers, 150 per class, PCA→256) | < −0.35 | > −0.30 | label permutation: mean slope over 5 permutations of the labels on the same clouds |
| P6.d | trace share of ḡ per slot on the decorrelated probe (Z113) | a ≥ 0.40 and b ≥ 0.40 and hint ≤ 0.05 | max slot ≤ 0.42 | same model at step 0 |
| P6.e | full 113-cycle among A-10-accepted permutations of D^g from outputs (hint NULL) | found | not found | the memorisers |

Applicable: Z113 → a, c, d, e on clean 9, 10, 11 and mem 9, 10. ResNet → b, c on cleanaug 3, 4 and scrambled 3, 4.
**Each instrument passes** iff every model it applies to is on its registered side. **Block passes** iff every instrument passes (a, b, c, d, e).
Reported without criterion: concentration C (ḡ and per-input), for the record of why it was removed. **PAUSE 2.**

## P7 — the generator as a leading indicator. Sweeps over EVERY checkpoint (0…50) of clean 9–11, mem 9–10, decoy 9–10.
Jump checkpoint g = argmax of the checkpoint-to-checkpoint increase in test accuracy (as in Runs 3–4). First acceptance c₁ = first checkpoint
with a full 113-cycle accepted. **Lead = g − c₁** (positive = generator first).
- **P7.a** Pass iff lead ≥ 0 on clean 9, 10, 11 AND no full cycle at any of the 51 checkpoints of mem 9, 10. Reported: lead, test accuracy at c₁.
- **P7.b** Pass iff on 3/3 clean: q(swap) < 0.05 at some checkpoint i with |i − g| ≤ 3; AND on mem 9, 10: q(swap) > 0.30 at every checkpoint with
  train ≥ 0.99. A clean seed with no checkpoint (train ≥ 0.99 and test < 0.1) before g is flagged "no memorisation phase"; the clause is scored
  the same way and the flag is reported.
- **P7.c** Pass iff on decoys 9, 10: (collapse) hint-slot trace share < 0.01 at some checkpoint in [g, g+4]; AND (decline) over the checkpoints
  from the first with train ≥ 0.99 up to g−1, Spearman ρ(checkpoint, hint share) ≤ −0.8 and the share at the first of them > 0.30 — scored only
  if that window has ≥ 5 checkpoints, otherwise reported "window too short" and the clause rests on the collapse. (Run 4 on this rule:
  0.47 → 0.26 → < 0.01 at g+3 and g+2; decoys reached the jump at checkpoints 39 and 14.)
- **P7.d** Pass iff on 3/3 clean the full cycle is accepted on ≥ 95% of checkpoints after c₁.
Reported on decoys too: lead, persistence. Figure: formation per seed with c₁ and g marked.

## T — finite tower (Z113: clean 9–11, mem 9–10, decoy 9–10; final checkpoints; 500 inputs, seed 0; hint NULL unless transformed)
d_g^ℓ(h, h′) = Σ_{k=0}^{15} ‖ψ_ℓ(h_{k+1}) − ψ_ℓ(h_k)‖, h_k = h + (k/16)(h′ − h), ψ_ℓ = the network from layer ℓ to the logits
(embed: the three input slots; resid_mid; resid_post; logits, where ψ is the identity). q_ℓ(T) = mean_x d_g^ℓ(h_ℓ(x), h_ℓ(Tx)) / mean_{x,x′} d_g^ℓ(h_ℓ(x), h_ℓ(x′)).
T ∈ {swap (a,b)→(b,a); translation (a,b)→(a+u, b−u), u = the discovered generator's step (u = 1 if no generator); hint change NULL→uniform random
residue; and, reported only, orbit translation (a,b)→(a+u, b)}. Planted check before any model: on a linear ψ the finite length must equal
‖ψ(h′) − ψ(h)‖ to 1e-5, and on ψ(h) = (cos h, sin h) the length of a segment of angle θ must equal θ to 1% (K = 16 under-measures an arc by < 0.2% at θ = 1).
- **T.1** decoys 9, 10: q(hint change) < 0.05 at resid_mid, resid_post and logits.
- **T.2** decoys 9, 10: q(translation) > 0.20 at embed, resid_mid, resid_post and logits.
- **T.3** mem 9, 10: q(hint change) > 0.15 at embed, resid_mid, resid_post and logits.
**Block passes** iff T.1, T.2, T.3 all pass. The Run-4 linearised tower is recomputed on the same models for the side-by-side table (no criterion).
ResNet: Run-4's Euclidean tower (shift 2 px, rotate 15°) reported, no criterion. **PAUSE 3.**

## Not in Run 5
P1.1, P1.2, P1.4, P3.1, P3.2 are Run-4 registered passes and are written up from Run 4's files, not rerun. No P4/P5, no new instruments.
If a registered threshold fails on Run 5's models, the findings file reports the measured gap between the Run-4 and Run-5 values and stops there.

## Deliverables
FINDINGS5_P6.md, FINDINGS5_P7.md, FINDINGS5_T.md, PAUSE5_1..3.md, CLAIMS.md (five-run grid), `figures/run5_instruments.png`,
`run5_formation.png`, `run5_tower.png`, `grid_five_runs.png`, `analysis/demo_run5.py` + transcript, report.

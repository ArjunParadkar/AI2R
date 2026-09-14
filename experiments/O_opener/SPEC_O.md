# Block O — the opener (pre-registration, 2026-09-14, from WHITEBOX_Final_Specification §5–§6, §8.2; written BEFORE any O compute)

_Naming: runs are horses (`experiments/HORSES.md`): O1.1 Sweep Tosho (T1 specimen training), O1.2 Meisho Doto (T1 full open()), O1.3 Nishino Flower (T1 planted bias), O1.4 Fine Motion (T1 P4/P5 geometry), O2.1 Satono Crown (GPT-2 open()), O2.2 Narita Top Road (GPT-2 audit = F11), O3.1 Hishi Akebono (Pythia P7/P3, HPCC), O4.1 Bamboo Memory (frontier, HPCC), O5 Symboli Kris S (the grid). Spec codes are the pre-registration labels._

Inputs frozen from the running programme: F (gauge, audit, cross-fiber count), K (MP null → R4), V (phase diagram → P0 verdict regions), Z (zigzag, multiparam, Z5 normalised persistence), M/C/B/D (audited signals). Nothing is re-run; they are components (§8.1).

## O1.1 — T1 specimens (Sweep Tosho)
**Grokked modular addition**, p = 113, 1-layer `ToyTransformer` (d_model 128, 4 heads, d_mlp 512, pre-LN, vocab p+1 with '=' = token p), sequence [a, b, =], logits at the last position over p classes. Training: full batch on a fixed random 30 % of the p² pairs, AdamW lr 1e-3, β = (0.9, 0.98), weight decay 1.0, ≤ 30 000 epochs, checkpoints every 1 % of the budget (300 per model) plus the final. Three conditions × 5 seeds:
* **clean** — as above.
* **planted-shortcut** — sequence [a, b, s, =] with a fourth token s ∈ {p+1, …, p+8}. For the first 30 % of epochs s is uniformly random (uninformative); from epoch 0.30·T on, s = p+1+((a+b) mod 8) (leaks the residue class mod 8). The *data onset* is 30 %. Ground-truth **crystallisation** (§6.4) = first checkpoint at which ablating s (replacing it by a random residue token) changes test accuracy by > 5 points.
* **memorising control** — labels replaced by a fixed random permutation of the answer per pair (no group structure to learn), weight decay 0, same budget. Expected: train acc → 100 %, test acc at chance, Fourier concentration κ < 0.3.
**Diagnostics / exclusion** (Nanda): a clean or planted specimen is *grokked* iff final test accuracy > 99 %; Fourier structure present iff κ (extractors.fourier_irreps, τ = 0.01) > 0.9 on the token embedding. Specimens failing either are excluded and logged in `results/O1_1_excluded.csv` with the reason; the memorising control must have test acc < 5 %.

**Ring attractor RNN**, 128 units, trained from scratch on angular integration: discrete-time leaky ReLU RNN h_{t+1} = (1−α) h_t + α·ReLU(W h_t + U u_t + b), α = 0.1; input u_t = (ω_t, cos θ_0·[t=0], sin θ_0·[t=0]) with ω_t a smooth random angular velocity (OU process, σ = 0.3 rad/step); target (cos θ_t, sin θ_t) with θ_t = θ_0 + Σ ω; linear readout; sequences of 100 steps, batch 128, Adam lr 1e-3, 4 000 steps, checkpoints every 2 % (50 per model); 10 seeds. Diagnostic: final angular error < 0.15 rad on held-out sequences of length 200; failures excluded and logged.

Outputs: `results/specimens.csv` (one row per model: condition, seed, train/test acc, κ, angular error, grokked flag, exclusion reason), checkpoints under `results/ckpt/<condition>_s<seed>/step_XXXX.pt` (+ `final.pt`), training curves `results/curves/<name>.csv`.

## O1.2 — T1 full open() (Meisho Doto)
Steps 1–9 of §3 on every included T1 specimen. Pre-registered pass (§8.2): all seven cells certified on ≥ 4/5 clean seeds; R1/R2/R5 pass; γ_cov < 0.05. Sheaf sanity gate: H¹ = 0 on clean modular addition. Per-cell criteria are exactly §6:
* **P0** tractable on T1 (V phase-diagram placement; null: degree-preserving rewiring).
* **P1** group: κ > 0.9; K identical across all 300 gauge samples; reconstruction from K alone passes verify. Ring: β₁ = 1 bar with lifetime > 5× the next; circular coordinate vs true angle r > 0.99 after per-unit normalisation (recorded as gauge-fixing). Null: feature shuffle, random-direction features.
* **P2** path product A(x) exact invariance (CV < 1e-6); IG anchor agreement r > 0.95 on constant-pattern regions (D1 anchor).
* **P6** κ separates grokked vs memorising with AUC 1.0.
* **P7** irrep appearance times form the grokking transition; spectral gap opens within 5 % of the accuracy jump; frequency ordering consistent across ≥ 4/5 seeds.

## O1.3 — T1 planted-bias detection (Nishino Flower)
P3 procedure (zigzag over checkpoints + Laplacian eigenflow) on planted vs clean. Pass: detected onset within 5 % of training of the ground-truth crystallisation; zero false detections on clean specimens. Null: the same pipeline on clean must detect nothing.

## O1.4 — T1 P4/P5 geometry (Fine Motion)
* **P4** diffusion-embedding OOD distance (Nyström) on held-out / OOD (moduli and ranges unseen, corrupted tokens) / adversarial inputs: AUC > 0.9 AND beats softmax confidence. Null: random-projection embedding.
* **P5** ρ(x) = ‖Q_h(x)‖·Lip(x); failure radius by actual Gaussian perturbation at r ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.5}: Spearman(ρ, radius) < −0.5, p < 0.01, beating the softmax-margin baseline. Null: shuffled ρ.

## O2 (GPT-2), O3 (Pythia), O4 (frontier), O5 (grid) — as §8.2; thresholds per §6; HPCC rows ship as pending.

## Kill tests (run before believing any cell)
* K-O-1 *tautology check*: the P1 reconstruction (R1) must fail (accuracy near chance) when K is replaced by a random frequency set of the same size.
* K-O-2 *gauge check*: every certified quantity re-computed on 300 gauge samples of each specimen; any CV > 1e-3 downgrades the cell to "failed (artifact)".
* K-O-3 *memorising control*: every P1/P6 procedure run on the memorising specimens must NOT certify group structure.

Thresholds above are frozen at this commit; changes go to AMENDMENTS.md with reasons.

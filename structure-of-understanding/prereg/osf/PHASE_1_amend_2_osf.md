# Pre-registration, Phase 1, amendment 2: step-level leaf rule (M9) and planted systems with shared sub-structure

WhiteBox E&E. Written 2026-10-23 after amendment 1 failed P2. DRAFT until the annotated tag `prereg-phase-1-amend-2`; the hash and SHA-256 are recorded in `docs/record.md`. Nothing here runs on the validation seeds before the tag.

## Why

Amendment 1 (principal-subspace realisers) made realisers accurate. With accurate realisers the exception node of V2 is legitimately reused: on rule inputs, swapping a stored label into it changes the answer, so its generality is about 0.85. Memorized inputs were then protected only by the value-level leaf rule (M3). That rule fails whenever several memorized inputs share a stored label (the answer funnel), and P2 failed on seed 105. The amendment-1 failure stays in `docs/record.md`.

## Change 1 — M9, a step-level leaf rule (theory §5)

On input x, a node C contributes g(C) to λ(x) and μ(x) only if both of the following hold, each supported by at least n_min distinct training inputs:
- its value is in range, C(x) ∈ Rg(C) (M3, unchanged);
- its parent tuple at x is in the combination range of Definition 4.1, Pa(C)(x) ∈ Rg(Pa(C)).

Otherwise C is a leaf on x. A value reached through a parent tuple that only x (or fewer than n_min inputs) has is a lookup step, whatever the value.

**Consequence, stated before any validation run.** If every node of a system is computed directly from the full input identity (all parent tuples unique), then λ = μ = 0 on every input under M9, and structural memorization is undefined. Planted V2 is such a system: two inputs (a, b), and every node is a function of the pair. Its rule and its lookup are both tables over (a, b), indistinguishable by any reuse statistic. Under M9 V2 is therefore expected to read everything as memorized. It is kept as a reported control ("structurally undecidable") and no longer gates.

## Change 3 — realiser modes and canonicalisation (needed for real LLMs)

**Development finding (Qwen2.5-1.5B on Pinnacle, job 1364727; development model).**
- Delta-mode realisers, which take what one layer wrote, mostly fail there: transformers write a variable gradually across layers.
- Full-state ("plain") principal-subspace realisers of s0 and r0 at the answer position, layers 24–27, pass the quotient test: swap effect A = 0.32–0.42 against a random null η ≈ 0.02, and δ_Q ≈ 0.02.
- Plain mode alone fails on planted residual nets: a variable written into its own block is also decodable from upstream blocks (V4's conj from gn0 and gn1).

**Changes.**
- The probe proposer offers **both** modes, delta and plain, side by side, and the filters choose.
- The **coarsening rule (M4 ii) is removed**. In plain mode every derived variable is linearly readable from persisting upstream blocks, and mixed readouts become spurious "containers" that remove true variables. Approximate early readouts are handled by the M5 fixed point (suspension and peeling).
- An input realiser counts as a container only at the input's primary site.
- The composite rule (M4 iii) is unchanged.

Development check (laptop quick tests, seeds 0–2; single processes under a minute):
- V1 one-hot and noise 0.5 are exact.
- V4 is exact (13 nodes, scaffold principles).
- V3 and V5 are exact.

## Change 2 — planted systems with shared sub-structure (replacing V2 and FORM in the gate)

- **V2D.** Inputs a, b ∈ Z_11 and a distractor z ∈ Z_8. The rule is y = (a + b) mod 11 through a sum node with parents (a, b), so every (a, b) tuple is shared by 8 inputs. Forty memorized items are keyed on the full input (a, b, z) through per-item detectors and an override chain. N = 2000.
- **FORMD.** Seven checkpoints of V2D with p = 5 and z ∈ Z_5. The memorized fraction of the 125 inputs is 1, 1, 1, 0.8, 0.1, 0.1, 0.1. Labels exclude the per-item detectors.

## Criteria (frozen at the tag)

The criteria are those of `prereg/PHASE_1.md` §3, applied to the amended estimator (amendment 1 plus Changes 1–3), with two changes:
- P2 is evaluated on V2D instead of V2.
- P6a and P6b are evaluated on FORMD instead of FORM.

The other criteria (P1a–P1f, P3, P4, P5) are re-run unchanged.

- **Seeds:** 110, 111, 112, 113, 114.
- **Reported controls:** V2 and FORM, with their verdicts printed but not gating.
- **Consequences:** as in `prereg/PHASE_1.md` §4.

## Development check (before the tag)

Seeds 0–2 on Pinnacle (job 1364799), at the tag: V1 one-hot and noise exact (seeds 0–2); V3, V4, V5 exact; NULL 0 accepted; V2D AUROC(−μ) 0.994 and 0.999 (seeds 0, 1; seed 2 still running); V2 (control) all-memorized as predicted; FORMD dev still running (formation gates only K14). LLM development run (Qwen2.5-1.5B, job 1364800): node s0 at L25@pre_ans_d0 with parents {a0, b0}, IIA 0.97, out0 ← s0; out1 ← raw inputs.

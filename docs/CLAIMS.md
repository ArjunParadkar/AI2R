# CLAIMS

Placeholder. Each library function (`whitebox_ee`) and each paper claim will be listed here with its access level, its null, its baseline, its failure boundary and its status: tested, partly tested, or not measured.

No claims exist yet. As of Phase 0 nothing has been tested. The theory's propositions (`theory/core.pdf`) are mathematical statements with proofs awaiting a second-reader check, and they are not empirical claims.


## Library functions: validation status (Phase 1, 2026-10-23)

| Function | Access level | Null | Baseline | Status | Failure boundary |
|---|---|---|---|---|---|
| Discovery with the probe proposer (`discover`, `propose_probes`) | C | untrained copy; shuffled labels; norm-matched random realisers; within-fibre swaps | oracle realisers (planted) | Validated on planted systems (P1a–P1f) | Needs N ≳ 1000 probe inputs; circularly encoded variables are missed by linear-decoder realisers |
| `generality` | C | random realisers; untrained copy | oracle generality | Validated (P1e, P4) | — |
| `depth` (λ, μ, γ) and the memorization verdict | C | — | planted memorized subset | Validated (P2) | Answer funnel: memorization is not flagged when memorized answers pass through a shared, answer-valued, non-output-equivalent node (P6b) |
| `move` | C | — | construction labels | Validated (P3) | Needs a reference derivation for predicted correctness |
| `create` | C | control exposure | planted creation variants | Validated (P5) | — |
| `formation` | C | — | planted checkpoints | **Failed (P6b)**; not for use | Inherits the answer funnel |
| Unsupervised proposers | C | — | — | Not validated; not for use | Do not recover factorial categorical codes |

No claim about any real system exists yet.

# Phase 4 report — language models (2026-10-05)
Authors: Arjun Paradkar and Reva Paradkar.  Pre-registrations: `prereg/PHASE_4.md` (tag prereg-phase-4), `prereg/PHASE_4B.md` (tag prereg-phase-4b, one dated amendment).  Full cells in `docs/record.md`.  Scope actually done: blind circular structure (days, months) in 6 pretrained models; formation across Pythia-70m/160m training; knowledge boundary on TriviaQA in 2 models.  NOT done: LLM memorization detection vs extraction ground truth, extractor lambda on LLMs, the text-transformation library, semantic entropy with NLI clustering, SelfCheckGPT's NLI/BERTScore variants, Pythia checkpoints beyond 70m/160m, any LLM larger than 1.5B.

## Results against the pre-registered rules
| hypothesis | rule | result | verdict |
|---|---|---|---|
| H4.1a days | score >= 0.85, max-over-layers permutation p < 0.01, >= 4/6 models | 6/6 (score 1.0, p 0.0027-0.009) | PASS on the raw rule; Holm-adjusted over 30 tests the days p-values are 0.066-0.17 (a 7-item set has 360 distinct cyclic orders), none < 0.05 |
| H4.1b months | same | 5/6 (pythia-70m 0.67); Holm-adjusted p <= 0.036 for all six | PASS |
| H4.1c controls (numbers, colours, animals) | <= 1/6 models with p < 0.01 | 0/6 each (smallest Holm-adjusted p 0.254) | PASS |
| H4.2 formation (descriptive, one run per size) | first checkpoint adjacency >= 0.85, p < 0.05 | absent before step ~512; in place by step 2000-4000 of 143000 in both sizes; numbers control clean at every checkpoint | descriptive |
| H4.3 knowledge boundary | kappa AUROC >= 0.70 and non-inferior (lower bound > -0.03) to every baseline, both models | Qwen2.5-1.5B 0.746 but not non-inferior to semantic entropy (-0.076) or sample agreement (-0.043); Llama-3.2-1B 0.681 < 0.70 | **FAIL** |

## What the evidence supports
1. Cyclic order of days and months is linearly recoverable from token representations without being supplied, in six models from 70M to 1.5B parameters, with number words / colours / animals as controls that stay at chance.  The best layer is the input embedding in most cases, so the structure is in the static embeddings.
2. It forms early (step ~2000 of 143000) in the two Pythia runs.
3. A post-hoc closure-ratio check (never upgrades a verdict) separates the closed day/month structure (ratio 0.9-1.2) from the single number-word layer that reached adjacency 0.85 (1.48).
## What it does not support
Paraphrase-coherence did not beat sampling-based uncertainty at flagging wrong TriviaQA answers; it beats plain token probability by ~0.05 AUROC.  The equivalences were designed, not learned.
## Caveats
Unit of analysis is one pretrained model; there are no seeds, so statements cover these six models only.  The days result is not significant after Holm correction.  A first launch of H4.3 died on CUDA OOM before any result and was re-run under a dated amendment (batch size and retry only).
## Compute
~1.5 GPU-hours (power-capped laptop GPU), no cloud spend.

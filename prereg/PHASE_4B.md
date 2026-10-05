# PRE-REGISTRATION — WHITEBOX v2, Phase 4 part 2 (knowledge boundary; formation across Pythia training)
Authors: Arjun Paradkar and Reva Paradkar (independent; no university affiliation).  Written 2026-10-05 before the runs below.  Tag `prereg-phase-4b` (annotated, unsigned: no GPG key).
Disclosure: `bench/llm/halluc.py` was smoke-tested on 24 TriviaQA questions only to check that it runs (Qwen2.5-1.5B-Instruct: accuracy 0.29, AUROCs 0.69-0.81, CIs +-0.25: no conclusion drawn).  Pythia formation has not been run.

## H4.3 Knowledge boundary on TriviaQA (rc.nocontext validation, 500 questions drawn with random_state=0)
Models: Qwen/Qwen2.5-1.5B-Instruct, unsloth/Llama-3.2-1B-Instruct.  Target: the greedy answer is correct (gold alias contained in the normalised answer).  Scores (higher = more confident): kappa (fraction of 6 paraphrase prompts whose greedy answer is lexically equivalent to the base answer; **designer equivalences, not learned**), negative lexical-cluster discrete semantic entropy over 8 samples at T=1 (the NLI-clustering semantic entropy of Farquhar et al. is NOT used), lexical sample agreement (a stand-in for SelfCheckGPT; the NLI/BERTScore variants are NOT used), mean token probability (MSP).  Unit = question; 95% CIs by bootstrap over questions (2000).
PASS per model: AUROC(kappa) >= 0.70 AND the paired bootstrap lower bound of AUROC(kappa) - AUROC(b) is > -0.03 for every baseline b (non-inferiority); superiority is claimed only if every lower bound is > 0.  Overall PASS needs both models.  Because the baselines are cheaper stand-ins for published methods, a pass here is not evidence against the published versions.

## H4.2 Formation across Pythia training (descriptive; unit = one training run per size)
Models pythia-70m and pythia-160m at steps {0,1,4,16,64,256,512,1000,2000,4000,8000,16000,32000,64000,143000}; days, months, numbers (control).  Reported: the first checkpoint with adjacency >= 0.85 and max-perm p < 0.05, per set and model.  No hypothesis test across runs is possible (one run per size); statements are descriptive and say so.

## Frozen artefacts
| file | sha256 |
|---|---|
| `bench/llm/halluc.py` | `331bf6742c556f4283e92b1e8867c245f72b5a1d053a9d12cfcf8c778487d65e` |
| `bench/llm/pythia_formation.py` | `4dd40be420d111f672c1bd5d9686c538abcdb58398875ee1377bc550ac330932` |
| `bench/llm/circular.py` | `64c2a88bffeb7cea2772fb9957360850bdeca65f270e02974f5ebb9452d2d766` |

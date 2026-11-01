# Data and model sources (checked 2026-10-23)

Checked against live pages and the HF / GitHub / figshare APIs on 2026-10-23. Gated or credentialed items are marked **pending**, never used.

## Models (HF, ungated, Apache-2.0 / MIT unless stated)
- Qwen/Qwen2.5-7B, -14B, -32B (Apache-2.0)
- allenai/OLMo-2-1124-7B, -13B (Apache-2.0). Intermediate checkpoints are git branches passed as `revision`: 965 for the 7B, e.g. `stage1-step1000-tokens5B`, and 718 for the 13B.
- EleutherAI/pythia-* (Apache-2.0). Checkpoints `step0`, `step1`, …, `step143000` (155 per model).
- mistralai/Mistral-7B-v0.1 (Apache-2.0); microsoft/phi-2 (MIT); deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (MIT)
- **Pending licence/token:** meta-llama/* (gated manual), google/gemma-* and Gemma Scope (gated).

## Sparse dictionaries (ungated)
- **Pythia:**
  - EleutherAI/sae-pythia-70m-32k, -160m-32k and the deduped variants (MIT; residual and MLP at every layer)
  - EleutherAI/sae-pythia-410m-65k (MLP)
  - jacobdunefsky/pythia-2.8B-transcoders (no licence stated)
- **Qwen2.5:**
  - nikoryagin/sae_Qwen_Qwen2.5-7B_resid_{pre,post}_layer_{6,12,18,24}_size_{16384,32768}_batchtopk… (no licence stated)
  - huypn16/sae-qwen-2.5-14B-OMS-16x (layers 12 and 24)
- **OLMo-2:** open-concept-steering/olmo2-7b-sae-65k-v1 (MIT, one SAE)
- **Not used:**
  - Llama-Scope (OpenMOSS-Team): the dictionary repos are ungated, but the base model is gated.
  - yoheikobashi OLMo-2 1B SAEs: gated=auto.

## Medical
- **DDXPlus** (Fansi Tchango et al., NeurIPS 2022): figshare 10.6084/m9.figshare.20043374, **CC BY 4.0**, no login, ~180 MB in total. It has 49 pathologies with a public knowledge base (symptoms and antecedents per condition). The chain is evidence → pathology. There is no published intermediate layer, so any syndrome layer would be our own definition and must be pre-registered as such.
- **Synthea** (Apache-2.0): its disease modules are explicit state machines (onset → symptoms → diagnosis), so the generating chain is known.
- **Pending access:** DiReCT (PhysioNet credential) and MIMIC (credential). Neither is downloaded.
- No public, ungated *pre-trained* diagnosis model with a known clinical chain was found. The fallback for K2's medical arm is a model we train ourselves on DDXPlus or Synthea, flagged as a project-trained model, not a production model.

## Syntactic agreement
- BLiMP: nyu-mll/blimp (CC BY 4.0; HF parquet; agreement subsets)
- Linzen, Dupoux & Goldberg 2016: github.com/TalLinzen/rnn_agreement (MIT; rnn_agr_simple.tar.gz, 41 MB)
- Marvin & Linzen 2018: github.com/BeckyMarvin/LM_syneval (MIT; templates)

## Brain
- **Gardner et al. 2022:** figshare 10.6084/m9.figshare.16764508, **CC0**, no login, one 320 MB zip of .npz files: rats q, r (day1/day2) and s, with grid modules. The analysis code (github.com/erikher/GridCellTorus) is GPL-3.0; it is used only as a reference, and we reimplement rather than vendor it.
- **Vector-HaSH:** github.com/FieteLab/VectorHaSH (MIT).
- **Pending lab:** all Fiete-lab-only recordings and simulations.

## CCGP definition (Bernardi et al. 2020, Cell, PMC8451959, Methods)
- Trials are split by condition: the training set comes from one group of conditions and the test set from a disjoint group.
- A linear classifier (max-margin SVM, trained on condition-mean firing-rate vectors) learns a dichotomy on the training conditions and is scored on the held-out conditions.
- CCGP is the average over all choices of training conditions of a given size. The main figures use 3 conditions per side for training and the 4th for testing.
- The null puts points from the same condition at random locations.

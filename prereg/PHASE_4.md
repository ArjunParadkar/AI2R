# PRE-REGISTRATION — WHITEBOX v2, Phase 4 (language models), part 1: blind circular structure
Authors: Arjun Paradkar and Reva Paradkar (independent; no university affiliation).  Written 2026-10-04, before any language-model run of this analysis.
Tag `prereg-phase-4` (annotated, unsigned: no GPG key).  Part 2 (hallucination / knowledge boundary, Pythia formation) is registered in PHASE_4B.md before it is run.
Access level C (hidden states).  The cycle is never supplied to the method; the true cyclic order is used only for scoring.

## What exists before this registration
`bench/llm/circular.py` was unit-tested on planted data only (a noisy planted 12-circle scores 1.0; its label-shuffled copy scores 0.33).  No language model has been run through it.

## Method (frozen)
For each model and item set, hidden states at the last sub-token of the item word are averaged over 8 templates (listed in the file) at every layer (embeddings + all blocks); per layer: centre, PCA to 6 dimensions, Euclidean distance matrix, exact optimal Hamiltonian cycle (Held-Karp).  Score = fraction of the n true cyclic neighbour pairs that are adjacent in the recovered cycle (chance for n=7: 0.33; n=12: 0.18).  The headline score of a model/set is the MAXIMUM over layers; its p-value is the max-over-layers permutation null (true order uniformly re-labelled, 20000 draws, +1 correction) so layer selection is controlled.
Item sets: days (7, cyclic), months (12, cyclic); controls with an arbitrary fixed 'true' cyclic order: number words one..seven (ordered, not cyclic), colours, animals.
Models: gpt2, gpt2-medium, EleutherAI/pythia-70m, EleutherAI/pythia-410m, unsloth/Llama-3.2-1B, Qwen/Qwen2.5-1.5B-Instruct (all cached locally).  Unit of analysis = one pretrained model (there is no seed replication); statements are about these six models only.

## Hypotheses
**H4.1a** days: for each model, headline score >= 0.85 and max-perm p < 0.01.  PASS if >= 4 of 6 models.
**H4.1b** months: same threshold, PASS if >= 4 of 6 models.
**H4.1c controls**: for each control set (numbers, colours, animals) the fraction of models with p < 0.01 must be <= 1/6.  Numbers are the informative control (ordered but not cyclic): a method that recovers a cycle from an ordered line is not reading cyclicity.
**H4.1d** Holm correction across the 6 x 5 headline p-values is reported; PASS statements above use the raw 0.01 and the Holm-adjusted p < 0.05 is reported next to them.
Failures are results.  If numbers pass the test at p < 0.01 in >= 2 models, the instrument is declared not specific to cycles and H4.1a/b are downgraded to "structure recovered, specificity not shown".

## Frozen artefacts
| file | sha256 |
|---|---|
| `bench/llm/circular.py` | `64c2a88bffeb7cea2772fb9957360850bdeca65f270e02974f5ebb9452d2d766` |

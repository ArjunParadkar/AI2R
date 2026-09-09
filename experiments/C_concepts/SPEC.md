# Block C — Audited concept map (pre-registration)

_Naming: every run in this document is a horse from the Tracen roster (`experiments/HORSES.md`); the handbook codes below are the pre-registration labels and appear here only as such._

Status: SPEC written 2026-09-07 before any Block-C run. Novelty: **FLAGGED** — feature alignment
across seeds and RTD/tRSA comparisons exist; the invariance-certification layer on SAE features is
likely open ("SAE feature consistency across seeds", "gauge fixing sparse autoencoder features" —
searched before write-up, NOVELTY.md).

## C1 — Concept-method audit  [5070]
GPT-2 small layer 6, public SAE (`gpt2-small-res-jb`, `blocks.6.hook_resid_pre`), the K1 token set
(50k tokens). Renamings (n = 20 per kind: head perm, rotation, MLP perm, LN scaling; layers 0–5,
i.e. everything upstream of the hook, and layer 6 itself). Since none of these renamings acts on
the residual stream, SAE features recomputed on the renamed model's residual activations are
expected identical — this is stated as the **structural result**: residual-stream SAE features are
gauge-invariant by construction, the audit is a float-floor check (CV < 1e-4 in fp32). The
non-trivial audit is on **MLP-space features**: a sparse dictionary (k-sparse, 4096 latents) fitted
by us on the layer-6 MLP-hidden activations (3072-dim) of the 50k tokens, recomputed per renaming
with the dictionary transformed consistently (decoder rows permuted with the units; LN scaling
does not act on post-GELU activations). Twenty standard concepts = the 20 most frequent
part-of-speech / token-class tags in the corpus (spaCy `en_core_web_sm` fine tags), linear probes
(ridge, held-out 20 %); pairwise feature geometry = cosine matrix of decoder directions.
`audit_quantities` with `align="gauge"` (exact) and blind Hungarian alignment on decoder
directions (matched by cosine); report both.

## C2 — Certified inventory  [5070]
Features whose identity is stable after **blind** alignment across all samples (matched feature's
cosine ≥ 0.95 in every renaming) form the certified inventory; fraction certified reported for
the MLP-space dictionary; for the residual SAE the fraction is 1 by construction (stated).

## C3 — Certified geometry + Mapper  [5070]
On certified MLP-space features: invariant relational structure (angle matrix up to gauge;
persistence of the feature-direction cloud), Mapper graph (`topology.mapper`) of the MLP
activation space with the certified-feature lens (top-2 certified feature activations), compared
to Mapper with an uncertified lens (random uncertified features) — β0/β1 of the Mapper graphs and
their stability across renamings. NOVELTY FLAGGED (TopoAct; cite; contribution = certification).

## C4 — Scale  [HPCC]  `hpcc/sirius_symboli.py`.

## C5 — Retire v1 P1 claims  [LAPTOP]
`AUDIT.md`: retirement of FPL↔accuracy and crystallization-order (v1 P1 n.s. on full-data models,
Phase 4 of whitebox-lib).

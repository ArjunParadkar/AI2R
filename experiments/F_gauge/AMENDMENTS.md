# Block F — amendments (append-only)

**#0 (2026-09-07, before the full F2 run; after a 2-specimen smoke test, no
results/ files written).** The smoke test (XOR3_d2_w6_s0, THR2_d2_w10_s0)
showed 0 of 28 minimal circuits carry a handbook (bit-at-0) interpretation:
with bits fixed at pre-activation 0, a second-layer unit's bit is generally
not a function of its parents' bits because it depends on their real-valued
activations. Decision: promote the Méloux threshold definition (E1's) to
co-primary, analysed in full alongside the handbook definition; specimens
with N = 0 under a definition are excluded from that definition's ratio
statistics and counted. No threshold or hypothesis changed.

**#1 (2026-09-07, before any kill-test result is recorded).** The F4 kill networks (k
algorithms × 2 gauge copies, each copy with its own constant-1 bias chain so that a copy
alone yields ±0.5 and an ablated copy contributes 0 under both criteria) have 14 / 28 / 42
hidden units for k = 1 / 2 / 3. Exhaustive enumeration is run for k = 1 (CPU) and k = 2
(GPU, 2^28); for k = 3 (2^42) the kill is evaluated on the planted circuits directly:
each of the 2k planted circuits must be valid under both criteria, minimal (checked by
enumerating all 2^|C| subsets of each planted circuit), and the 2k explanations must
partition into exactly k orbits. The first version of the planted construction (copies
averaged into a shared output bias) was wrong — a single copy could not cross the output
threshold under zero ablation — and was replaced before use; its output was discarded.

**#2 (2026-09-07, before kill-test results are recorded).** With the sign-of-logit validity
criterion and complement ablation, a parallel composition of k algorithms always admits
*recombined* circuits (e.g. algorithm A's positive-side units with algorithm B's bias
chain, or A's "p∧¬x3" unit with B's "x3∧¬p" unit) — these are genuinely valid circuits of
the planted network, not artefacts, so the literal expectation N = 2K cannot hold for any
construction we found (three constructions tried on k = 1, all showed recombinations).
The F4 kill is therefore evaluated as: (i) every planted copy is a valid circuit under both
criteria; (ii) the two gauge copies of the same algorithm receive the same orbit key and
copies of different algorithms different keys, so K over planted circuits = k for
k = 1, 2, 3; (iii) the exhaustive N and K (k ≤ 2) are reported with the number of recombined
circuits; and (iv) **F4-kill-b (new, stronger)**: for 100 trained specimens (d·w ≤ 16) a
random g ∈ G is applied to the whole network and the multiset of orbit keys of its
explanations must be identical to the original's under both criteria — the direct test that
gauge-related explanations are identified.

**#3 (2026-09-07, before F8–F10 run).** F10 audit (100 renamings × 3 kinds × ~16 candidates,
persistence on 256-point probe clouds) costs ~1 min per specimen; it is run on seeds 0–4 of
every (function, depth, width) cell (760 specimens) rather than all 3 143 converged specimens.
F8 and F9 run on every converged specimen. Nothing else changes.

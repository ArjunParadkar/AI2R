# WHITEBOX v2 — shared brief for every agent (Arjun waived all gates 2026-10-04: full speed)
Repo: ~/blindinglight (library whiteboxtopo/, dirs theory/ proofs/ numerics/ bench/ prereg/ runs_v2/ reports/ paper/ docs/). Python: use the venv other WhiteBOX work uses
(check ~/blindinglight/scripts/*.sh for the activation line). Authors on every doc: Arjun Paradkar and Reva Paradkar, no university affiliation.
Master spec = the "WHITEBOX v2 Master Research Prompt" (pasted by Arjun; summary in this file). Non-negotiables:
1 No ornamental math: every formalism yields a falsifiable prediction that is tested, else cut -> docs/cut.md.
2 Pre-registration: prereg/PHASE_X.md BEFORE confirmatory runs, git commit + tag prereg-phase-X; thresholds never change after; post-hoc is labelled post hoc and never upgrades a verdict.
3 docs/record.md: one row per pre-registered cell, failures included. Report every failure.
4 Stats: unit = independent seed (never checkpoints); >=10 seeds confirmatory; bootstrap-over-seeds 95% CIs; exact permutation tests; Holm across instruments; never multiply correlated p-values;
  every instrument vs trivial baselines (weight norm, logit entropy, max-softmax, sharpness/Hessian trace, test acc).
5 Access levels A (prediction-only) / B (Jacobian) / C (internals) / D (bio). Never "from outputs alone" for B.
6 Language: no "first/breakthrough/theorem" unless a checked proof exists in proofs/.
Hardware/gotchas: one RTX 5070 Laptop 8GB, 30GB RAM, 35GB free disk (don't fill it; delete scratch). On this card the EFFICIENT_ATTENTION SDPA backward is WRONG:
all gradient work must enable only MATH sdpa (see ~/Whitebox/whitebox-v2/hpcc/night_torch_wrap.py) — or avoid nn.MultiheadAttention/SDPA fast paths.
NEVER use pkill/pgrep -f with patterns that appear in your own command (kills your shell). Long jobs: `systemd-run --user --unit=NAME ...` so they survive the session. No Slurm here.
Only one agent should hold the GPU at a time unless jobs are tiny; GPU owner = the zoo agent. Others use CPU (nproc=20) unless told otherwise.
Do not git push anything. Local commits fine (message ends with: Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>). Don't touch ~/.config/hypr or system packages.
Each agent ends with a concise report: what passed / failed / cut / next / compute used. Write it to reports/ too.

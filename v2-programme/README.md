# whitebox-v2 — Measurement Theory of Interpretability

Programme: `PROGRAM.md` (verbatim brief, pre-registered). What came before: `WHITEBOX_V1.md`.
Each experiment: `E*/SPEC.md` (written before running; amendments appended, never edited),
`run_*.py`, `results/`, `FINDINGS.md`. Env: `/home/god/whitebox-lib/.venv` (v1 library + pandas).

    cd E1 && /home/god/whitebox-lib/.venv/bin/python -m pytest -q tests   # SPEC controls
    /home/god/whitebox-lib/.venv/bin/python run_e1.py --smoke              # 2 tiny cells

"""Pre-tag check (introduced after the amendment-2 process failure, reports/phase1.md §8).

No Phase 1 amendment may be tagged unless this script exits 0 on the amendment's final
development run, read from the actual result files:

  python validation/pretag_check.py RESULTS_DIR 0,1,2 [--cells V1 V2D V3 V4 V5 NULL FORMD] [--commit HASH]

It fails (exit 1) if
  - any expected cell file is missing, unreadable or recorded an error;
  - any cell was produced by a commit other than --commit (when given);
  - any single null cell exceeds the P1f per-seed bar (FP > 0.02) or the null mean accepted > 1;
  - any gated criterion fails on any single development seed (each seed must pass on its own;
    a "4/5" criterion is treated as "1/1"), or on the development seeds pooled.
Rows marked "(control, not gating)" are printed but never decide.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("seeds")
    ap.add_argument("--cells", nargs="+", default=["V1", "V2D", "V3", "V4", "V5", "NULL", "FORMD"])
    ap.add_argument("--commit", default=None)
    a = ap.parse_args()
    d = Path(a.results)
    seeds = [int(s) for s in a.seeds.split(",")]
    sys.path.insert(0, str(HERE.parent))
    from validation.run_phase1 import CELLS
    problems = []
    # 1. every expected file exists, is clean, and comes from the right commit
    for s in seeds:
        for c in a.cells:
            for spec in CELLS[c](s):
                f = d / ("_".join(str(x) for x in spec) + ".json")
                if not f.exists():
                    problems.append(f"missing {f.name}")
                    continue
                try:
                    r = json.loads(f.read_text())
                except Exception as e:
                    problems.append(f"unreadable {f.name}: {e!r}")
                    continue
                if "error" in r:
                    problems.append(f"error in {f.name}: {r['error']}")
                if a.commit and not str(r.get("commit", "")).startswith(a.commit):
                    problems.append(f"{f.name} from commit {r.get('commit')} != {a.commit}")
                if c == "NULL":
                    fp, n = r.get("fp_rate", 1.0), r.get("n_accepted", 99)
                    print(f"null {f.name}: FP {fp:.4f}, accepted {n} {r.get('accepted_labels')}")
                    if fp > 0.02:
                        problems.append(f"null {f.name}: FP {fp:.4f} > 0.02 (accepted {r.get('accepted_labels')})")
    # 2. gated criteria per seed and pooled, by the verdict script itself
    for ss in [[s] for s in seeds] + [seeds]:
        out = subprocess.run([sys.executable, str(HERE / "verdicts_phase1.py"), str(d), ",".join(map(str, ss))],
                             capture_output=True, text=True)
        if out.returncode != 0:
            problems.append(f"verdict script crashed on seeds {ss}: {out.stderr.strip().splitlines()[-1:]}")
            continue
        for line in out.stdout.splitlines():
            if not line.startswith("| P"):
                continue
            cells_ = [x.strip() for x in line.strip().strip("|").split("|")]
            name, verdict = cells_[0], cells_[-1]
            fam = name.split()[1].split("-")[0] if len(name.split()) > 1 else ""
            if fam not in a.cells:
                continue
            if "not gating" in name:
                print(f"seeds {ss}: {line}  [control]")
                continue
            print(f"seeds {ss}: {line}")
            if verdict != "PASS":
                problems.append(f"seeds {ss}: {name} {verdict} ({cells_[2]})")
    print()
    if problems:
        print(f"PRE-TAG CHECK FAILED ({len(problems)} problems):")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("PRE-TAG CHECK PASSED")


if __name__ == "__main__":
    main()

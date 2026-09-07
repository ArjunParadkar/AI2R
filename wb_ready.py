"""Queue gates: exit 0 when a stage's inputs are complete on disk (units-alive checks alone race on unit failure/relaunch).
usage: wb_ready.py f2 | f2gpu | f7"""
import csv, os, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "F_gauge", "results")
def f2(lo, hi):
    done = {r["specimen"] for r in csv.DictReader(open(os.path.join(R, "enumeration_status.csv")))} if os.path.exists(os.path.join(R, "enumeration_status.csv")) else set()
    todo = [f'{r["function"]}_d{r["depth"]}_w{r["width"]}_s{r["seed"]}' for r in csv.DictReader(open(os.path.join(R, "specimens.csv")))
            if int(r["converged"]) and lo <= int(r["depth"]) * int(r["width"]) <= hi]
    rem = [s for s in todo if s not in done]; print(f"F2 nh {lo}-{hi}: {len(todo)-len(rem)}/{len(todo)} done"); return not rem
def f7():
    n = sum(1 for _ in csv.DictReader(open(os.path.join(R, "F7_status.csv")))) if os.path.exists(os.path.join(R, "F7_status.csv")) else 0
    print(f"F7: {n}/60 done"); return n >= 60
ok = {"f2": lambda: f2(0, 24), "f2gpu": lambda: f2(24, 24), "f7": f7}[sys.argv[1]]()
sys.exit(0 if ok else 1)

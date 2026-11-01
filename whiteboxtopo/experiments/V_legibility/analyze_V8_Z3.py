"""V8 (legibility_score calibration gates) and Z3 (graph-PH beta1 in the V4 logistic) — SPEC §V8, §Z3; V AMENDMENTS #2.
Inputs: V2_structure.csv, V3_discovery.csv, V1_models.csv, ../Z_topology/results/Z3_graphph.csv.
Writes results/V8_calibration.json (held-out AUC, within-architecture AUCs, ship verdict) and
../Z_topology/results/Z3_deltabic.json (ΔBIC of adding beta1 over (Q_excess, tw/N, sparsity))."""
import os, json
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); Z = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Z_topology", "results")
BASE = ["Qexcess_0.1", "twN", "sparsity_nodes"]


def fit_logistic(X, y, l2=1e-3):
    def nll(t):
        z = np.clip(X @ t[1:] + t[0], -30, 30); return -np.sum(y * z - np.log1p(np.exp(z))) + l2 * np.sum(t[1:] ** 2)
    r = minimize(nll, np.zeros(X.shape[1] + 1), method="L-BFGS-B"); return r.x, r.fun


def bic(nll, k, n): return 2 * nll + k * np.log(n)


def load():
    s = pd.read_csv(os.path.join(R, "V2_structure.csv")); d = pd.read_csv(os.path.join(R, "V3_discovery.csv")); m = pd.read_csv(os.path.join(R, "V1_models.csv"))
    df = s.merge(d, on=["name", "task", "arch"]).merge(m[["name", "untrained", "random_labels", "eval_acc"]], on="name")
    df["twN"] = df["tw_0.1"] / df.N; df["learned"] = df.eval_acc >= 0.5
    df["success"] = ((df.F >= 0.9) & (df.frac <= 0.5) & df.learned).astype(int); df["family"] = df.arch.str.replace(r"\d+", "", regex=True)
    return df


def standardise(Xtr, Xte): mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; return (Xtr - mu) / sd, (Xte - mu) / sd


def v8(df):
    rng = np.random.default_rng(0); test = np.zeros(len(df), bool)
    for arch, g in df.groupby("arch"):                       # stratified 20 % hold-out by architecture
        idx = g.index.values.copy(); rng.shuffle(idx); test[idx[:max(1, int(round(0.2 * len(idx))))]] = True
    Xtr, Xte = standardise(df.loc[~test, BASE].values, df.loc[test, BASE].values); ytr, yte = df.success.values[~test], df.success.values[test]
    theta, _ = fit_logistic(Xtr, ytr); p = 1 / (1 + np.exp(-(Xte @ theta[1:] + theta[0])))
    out = dict(n_train=int((~test).sum()), n_test=int(test.sum()), success_rate=float(df.success.mean()),
               heldout_auc=float(roc_auc_score(yte, p)) if len(np.unique(yte)) == 2 else float("nan"), within_arch={})
    for arch in sorted(df.arch.unique()):
        mk = test & (df.arch == arch).values
        out["within_arch"][arch] = dict(n=int(mk.sum()), auc=float(roc_auc_score(df.success.values[mk], p[mk[test]])) if len(np.unique(df.success.values[mk])) == 2 else float("nan"))
    # 5-fold stratified-by-arch CV as the less noisy companion number (reported, not the gate)
    aucs = []
    for f in range(5):
        te = np.zeros(len(df), bool)
        for arch, g in df.groupby("arch"):
            idx = np.sort(g.index.values); te[idx[f::5]] = True
        A, B = standardise(df.loc[~te, BASE].values, df.loc[te, BASE].values); th, _ = fit_logistic(A, df.success.values[~te]); pp = 1 / (1 + np.exp(-(B @ th[1:] + th[0])))
        if len(np.unique(df.success.values[te])) == 2: aucs.append(roc_auc_score(df.success.values[te], pp))
    out["cv5_auc_mean"] = float(np.mean(aucs)) if aucs else float("nan")
    wa = [v["auc"] for v in out["within_arch"].values() if np.isfinite(v["auc"])]
    out["gate_heldout_auc_ge_0.75"] = bool(out["heldout_auc"] >= 0.75); out["gate_within_arch_ge_0.65_all"] = bool(wa and min(wa) >= 0.65 and len(wa) == len(out["within_arch"]))
    out["ships"] = bool(out["gate_heldout_auc_ge_0.75"] and out["gate_within_arch_ge_0.65_all"])
    out["coefficients_standardised"] = dict(zip(["intercept"] + BASE, map(float, theta)))
    json.dump(out, open(os.path.join(R, "V8_calibration.json"), "w"), indent=1); print(json.dumps(out, indent=1)); return out


def z3(df):
    p = os.path.join(Z, "Z3_graphph.csv")
    if not os.path.exists(p): print("Z3_graphph.csv missing — Z3 ΔBIC not computed"); return None
    z = pd.read_csv(p); d = df.merge(z[["name", "betti1_finite", "betti1_essential", "total_persistence_H1"]], on="name"); n = len(d); y = d.success.values
    X0 = d[BASE].values; X0 = (X0 - X0.mean(0)) / (X0.std(0) + 1e-9); _, nll0 = fit_logistic(X0, y); out = dict(n=n, bic_base=bic(nll0, X0.shape[1] + 1, n))
    for feat in ("betti1_finite", "betti1_essential", "total_persistence_H1"):
        x = d[feat].values.astype(float); x = (x - x.mean()) / (x.std() + 1e-9); X1 = np.column_stack([X0, x]); _, nll1 = fit_logistic(X1, y)
        b1 = bic(nll1, X1.shape[1] + 1, n); out[feat] = dict(bic=b1, delta_bic=out["bic_base"] - b1, pass_pre_registered=bool(out["bic_base"] - b1 > 10))
    json.dump(out, open(os.path.join(Z, "Z3_deltabic.json"), "w"), indent=1); print(json.dumps(out, indent=1)); return out


if __name__ == "__main__":
    df = load(); v8(df); z3(df)

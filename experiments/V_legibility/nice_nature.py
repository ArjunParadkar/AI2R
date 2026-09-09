"""V6 — v1's loss-invisible reorganisation, certified (SPEC §V6; V AMENDMENTS #2).
SV(t) = 2-Wasserstein distance between the H1 diagrams of consecutive-checkpoint PCA-20 clouds of the penultimate
layer (MLP: second hidden layer, 512-example eval probe; transformer: last layer's MLP-hidden activations at the read
position). Peak = SV above the 95th percentile of the model's own trajectory. Null per peak: 20 feature-shuffled copies
of both clouds (columns permuted independently) -> peak survives iff SV > 95th percentile of the null SVs.
Gauge audit of SV on the cloud (M1-style column action, same g on both checkpoints): 30 permutation, 30 scaling
(log-uniform [0.1, 10]), 30 combined renamings; verdict invariant iff CV < 0.01. Survives iff >= 50 % of peaks beat
the null AND SV is permutation-invariant (scaling verdict reported). Writes results/V6_sv.csv, V6_peaks.csv,
V6_audit.csv, V6_verdict.json. Resumable per model. CPU."""
import os, sys, csv, json, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mejiro_dober import plan, build, ARCHS, CK, R, STEPS, N_CK
from whiteboxtopo.legibility.tasks import Task
from whiteboxtopo.core.persistence import compute_diagrams
from persim import wasserstein
N_NULL, N_AUDIT, NPC = 20, 30, 20


def cloud(m, task, arch, dev="cpu"):
    x, _ = task.batch(512, 0, "eval"); x = x.to(dev)
    with torch.no_grad(): m.forward(x, record=True)
    outs = m._node_out
    if arch.startswith("mlp"):
        W = m.width; return torch.cat(outs[W:], 1).cpu().numpy().astype(np.float64)
    base = (m.nL - 1) * m.per
    A = torch.cat([o[:, m.read_pos, :] for o in outs[base + m.H: base + m.per]], 1)
    return A.reshape(A.shape[0], -1).cpu().numpy().astype(np.float64)   # V AMENDMENTS #5: read_pos can be several positions -> flatten to (n, features)


def dgm1(A):
    A = A - A.mean(0); U, s, Vt = np.linalg.svd(A, full_matrices=False); Z = A @ Vt[:min(NPC, Vt.shape[0])].T
    D = np.asarray(compute_diagrams(Z, maxdim=1, backend="ripser", thresh_percentile=95.0).dgms[1]).reshape(-1, 2)
    return D[np.isfinite(D[:, 1])]


def sv(A, B): return float(wasserstein(dgm1(A), dgm1(B)))


def shuffle_cols(A, rng): return np.stack([A[rng.permutation(len(A)), j] for j in range(A.shape[1])], 1)


def main():
    rows = [r for r in csv.DictReader(open(os.path.join(R, "V1_models.csv")))]
    for r in rows:
        for k in ("random_labels", "untrained", "seed"): r[k] = int(r[k])
    p_sv, p_pk, p_au = (os.path.join(R, f) for f in ("V6_sv.csv", "V6_peaks.csv", "V6_audit.csv"))
    done = set(pd.read_csv(p_au).name) if os.path.exists(p_au) else set()
    every = STEPS // N_CK
    for row in rows:
        if row["untrained"] or row["name"] in done: continue
        t0 = time.time(); task = Task(row["task"], random_labels=bool(row["random_labels"])); spec = dict(ARCHS)[row["arch"]]
        clouds = {}
        for k in range(1, N_CK + 1):
            step = every * k; ck = os.path.join(CK, row["name"], f"step_{step:06d}.pt")
            if not os.path.exists(ck): continue
            m = build(spec, task, 0.0, row["seed"]); m.load_state_dict(torch.load(ck, map_location="cpu")); m.eval(); clouds[step] = cloud(m, task, row["arch"])
        steps = sorted(clouds)
        if len(steps) < 3: continue
        svs = [dict(name=row["name"], task=row["task"], arch=row["arch"], random_labels=row["random_labels"], step=b, frac=b / STEPS, sv=sv(clouds[a], clouds[b]))
               for a, b in zip(steps[:-1], steps[1:])]
        vals = np.array([d["sv"] for d in svs]); thr = np.percentile(vals, 95); peaks = [d for d in svs if d["sv"] >= thr]
        pk = []
        for d in peaks:
            i = steps.index(d["step"]); A, B = clouds[steps[i - 1]], clouds[d["step"]]; rng = np.random.default_rng(d["step"])
            null = [sv(shuffle_cols(A, rng), shuffle_cols(B, rng)) for _ in range(N_NULL)]
            pk.append(dict(**d, null_p95=float(np.percentile(null, 95)), null_mean=float(np.mean(null)), beats_null=int(d["sv"] > np.percentile(null, 95))))
        # gauge audit at the largest peak pair (same g on both clouds)
        d = max(svs, key=lambda z: z["sv"]); i = steps.index(d["step"]); A, B = clouds[steps[i - 1]], clouds[d["step"]]; base = d["sv"]; au = {}
        for kind in ("perm", "scale", "combined"):
            rng = np.random.default_rng(hash((row["name"], kind)) % (2 ** 32)); out = []
            for _ in range(N_AUDIT):
                p = rng.permutation(A.shape[1]) if kind != "scale" else np.arange(A.shape[1])
                a = np.exp(rng.uniform(np.log(0.1), np.log(10), A.shape[1])) if kind != "perm" else np.ones(A.shape[1])
                out.append(sv((A * a)[:, p], (B * a)[:, p]))
            out = np.array(out); cv = float(np.std(np.r_[base, out]) / (abs(np.mean(np.r_[base, out])) + 1e-12)); au[kind] = cv
        audit = dict(name=row["name"], task=row["task"], arch=row["arch"], random_labels=row["random_labels"], sv_base=base, n_peaks=len(pk),
                     peaks_beat_null=int(sum(p["beats_null"] for p in pk)), cv_perm=au["perm"], cv_scale=au["scale"], cv_combined=au["combined"],
                     verdict_perm="invariant" if au["perm"] < 0.01 else "artifact", verdict_scale="invariant" if au["scale"] < 0.01 else "artifact", wall=time.time() - t0)
        for p, recs in ((p_sv, svs), (p_pk, pk), (p_au, [audit])):
            if not recs: continue
            new = not os.path.exists(p)
            with open(p, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(recs[0]))
                if new: w.writeheader()
                w.writerows(recs)
        print(audit, flush=True)
    au = pd.read_csv(p_au); pk = pd.read_csv(p_pk) if os.path.exists(p_pk) else pd.DataFrame()
    real = pk[pk.random_labels == 0] if len(pk) else pk
    verdict = dict(n_models=int(len(au)), n_peaks=int(len(pk)), frac_peaks_beat_null=float(real.beats_null.mean()) if len(real) else float("nan"),
                   frac_peaks_beat_null_random_labels=float(pk[pk.random_labels == 1].beats_null.mean()) if len(pk) and (pk.random_labels == 1).any() else float("nan"),
                   frac_models_perm_invariant=float((au.verdict_perm == "invariant").mean()), frac_models_scale_invariant=float((au.verdict_scale == "invariant").mean()),
                   median_cv_perm=float(au.cv_perm.median()), median_cv_scale=float(au.cv_scale.median()))
    verdict["survives"] = bool(verdict["frac_peaks_beat_null"] >= 0.5 and verdict["frac_models_perm_invariant"] >= 0.99)
    json.dump(verdict, open(os.path.join(R, "V6_verdict.json"), "w"), indent=1); print(verdict); print("done", flush=True)


if __name__ == "__main__":
    main()

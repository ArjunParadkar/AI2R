"""E5 run: train HD RNNs (CPU, <=4 workers), measure connectome + RNN graphs with E4's measures, nulls.
NOT FOR EXTERNAL CLAIMS until cleared with the Fiete lab. Usage: python run_e5.py [--smoke]"""
import os, sys, time, json, argparse
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd, torch
from multiprocessing import get_context
from e5lib import load_connectome, profile_adjacency, weight_importance, HDTask, HDRNN, planted_ring, rewire_directed, rewire_undirected, er_null, CORE_TYPES
from e5lib.connectome import type_block_matrix
from e5lib.hdtask import accuracy_tail, curriculum_T
from e4lib.tasks import metric
from e4lib.graph import symmetrise, threshold_density, modularity_q, treewidth_bound, sparsity
from e4lib.discovery import discover_circuit, jaccard
from sklearn.metrics import normalized_mutual_info_score as nmi

R = os.path.join(HERE, "results"); CK = os.path.join(R, "ckpt"); os.makedirs(CK, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--n_null", type=int, default=100)
args = ap.parse_args()
LOG = open(os.path.join(R, "run_smoke.log" if args.smoke else "run.log"), "a")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

# ---------------- model grid ----------------
def grid(smoke):
    g = []
    ns = 2 if smoke else 10
    g += [dict(kind="trained", cell="rnn", N=148, seed=s, shuffled=False) for s in range(ns)]
    if not smoke:
        g += [dict(kind="trained", cell="gru", N=148, seed=s, shuffled=False) for s in range(3)]
        g += [dict(kind="trained", cell="rnn", N=n, seed=s, shuffled=False) for n in (64, 256) for s in range(3)]
        g += [dict(kind="shuffled_target", cell="rnn", N=148, seed=s, shuffled=True) for s in range(3)]
        g += [dict(kind="untrained", cell="rnn", N=148, seed=s, shuffled=False) for s in range(3)]
    else:
        g += [dict(kind="untrained", cell="rnn", N=148, seed=0, shuffled=False), dict(kind="shuffled_target", cell="rnn", N=148, seed=0, shuffled=True)]
    for c in g: c["id"] = f"{c['kind']}|{c['cell']}|N{c['N']}|s{c['seed']}"
    return g

def _init(steps, lr, n_null, me):
    global STEPS_N, STEPS_LR, N_NULL, MATCH_EDGES; STEPS_N, STEPS_LR, N_NULL, MATCH_EDGES = steps, lr, n_null, me
def smoke_flag(): return STEPS_N < 1200

def train_one(cfg):
    torch.set_num_threads(1); t0 = time.time()
    task = HDTask(shuffled_targets=cfg["shuffled"]); m = HDRNN(cfg["N"], cell=cfg["cell"], seed=cfg["seed"])
    steps = 0; acc = float("nan")
    ck = os.path.join(CK, cfg["id"].replace("|", "_") + ".pt")
    if os.path.exists(ck) and not smoke_flag():  # resume: reuse checkpoint, re-derive eval metrics (steps=-1 marks resumed)
        m.load_state_dict(torch.load(ck)); steps = -1
    elif cfg["kind"] != "untrained":
        opt = torch.optim.Adam(m.parameters(), STEPS_LR, weight_decay=1e-5); n_steps = STEPS_N
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 if s < 1200 else max(0.1, 1.0 - 0.9 * (s - 1200) / max(1, n_steps - 1200)))
        for s in range(n_steps):
            task.T = curriculum_T(s) if not smoke_flag() else 100
            x, y = task.batch(64, cfg["seed"] * 100000 + s, "train"); loss = -metric(m(x), y)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step(); steps = s + 1
            if (s + 1) % 250 == 0 and s >= 1200:
                task.T = 100
                with torch.no_grad(): xe, ye = task.batch(256, 999); acc = accuracy_tail(m(xe), ye)
                if acc >= 0.9 and not cfg["shuffled"]: break
    task.T = 100
    with torch.no_grad():
        xe, ye = task.batch(256, 999); lg = m(xe); acc = accuracy_tail(lg, ye); acc_tol1 = accuracy_tail(lg, ye, tol=1); acc_all = (lg.argmax(-1) == ye).float().mean().item()
    torch.save(m.state_dict(), os.path.join(CK, cfg["id"].replace("|", "_") + ".pt"))
    return dict(**cfg, steps=steps, acc_tail=acc, acc_tol1=acc_tol1, acc_all=acc_all, train_time=time.time() - t0, learned=bool(acc >= 0.5))

def measure_one(cfg):
    """Weight graph + functional graph profiles, discovery, rewiring null, for one RNN."""
    torch.set_num_threads(1); t0 = time.time()
    m = HDRNN(cfg["N"], cell=cfg["cell"], seed=cfg["seed"]); m.load_state_dict(torch.load(os.path.join(CK, cfg["id"].replace("|", "_") + ".pt")))
    m.eval(); task = HDTask(shuffled_targets=cfg["shuffled"])
    x, y = task.batch(256, 12345)                                            # measure batch
    rows, sweeps = [], []
    # (1) weight graph
    Ew = m.weight_graph(); pw = profile_adjacency(Ew, null_n=20, match_edges=MATCH_EDGES)
    A10 = threshold_density(symmetrise(Ew), 0.1)
    qs, tws = [], []
    for k in range(N_NULL):
        Rm = rewire_undirected(A10, seed=k); qs.append(modularity_q(Rm, n_restarts=3, seed=k)[0]); tws.append(treewidth_bound(Rm))
    pw.update(graph="weight", rewire_Q_mean=np.mean(qs), rewire_Q_sd=np.std(qs), rewire_tw_mean=np.mean(tws), rewire_tw_sd=np.std(tws),
              rewire_Q_z=(pw["Q10"] - np.mean(qs)) / (np.std(qs) + 1e-12), rewire_tw_z=(pw["tw10"] - np.mean(tws)) / (np.std(tws) + 1e-12),
              rewire_p_Q=(np.sum(np.array(qs) >= pw["Q10"]) + 1) / (N_NULL + 1))
    # (2) functional graph (E4 EAP) + activation importance
    means = m.node_means(x); Ef, node_sc = m.edge_scores(x, y, means)
    with torch.no_grad(): m(x, record=True); imp_act = m._H.abs().mean((0, 1)).numpy()
    pf = profile_adjacency(Ef.numpy(), importance=imp_act, null_n=20, match_edges=MATCH_EDGES); pf["graph"] = "functional"
    pf["sparsity_weightdeg"] = sparsity(weight_importance(Ef.numpy())); pw["sparsity_act"] = sparsity(imp_act)
    # (3) E4 discovery pipeline, 3 seeds
    runs = []
    for s in range(3):
        tk, tg = task.batch(256, 777 + s); mu = m.node_means(tk); runs.append(discover_circuit(m, tk, tg, mu, seed=s))
    S_raw = float(np.mean([1 - r["frac"] for r in runs])); F = float(np.nanmean([r["F"] for r in runs]))
    js = [jaccard(runs[i]["keep"], runs[j]["keep"]) for i in range(3) for j in range(i + 1, 3)]
    disc = dict(S_raw=S_raw, F=F, cost=float(np.mean([r["cost"] for r in runs])), size=float(np.mean([r["size"] for r in runs])),
                jaccard=float(np.mean(js)), gap=float(np.mean([r["gap"] for r in runs])))
    for p in (pw, pf):
        sw = p.pop("sweep"); p.pop("membership10"); p.pop("N", None)
        for r in sw: sweeps.append(dict(id=cfg["id"], graph=p["graph"], **r))
        rows.append(dict(**cfg, **p, **disc, measure_time=time.time() - t0))
    # agreement of the two graphs' edge rankings
    a, b = np.abs(Ew).ravel(), np.abs(Ef.numpy()).ravel()
    from scipy.stats import spearmanr
    for r in rows: r["edge_rank_rho_w_vs_f"] = float(spearmanr(a, b).statistic)
    return rows, sweeps

def connectome_block(n_null):
    rows, sweeps, nulls = [], [], []
    for name, core in (("fly_core148", True), ("fly_full439", False)):
        c = load_connectome(core=core); W = c["W"]
        p = profile_adjacency(W, null_n=20, match_edges=None); memb = p.pop("membership10"); sw = p.pop("sweep"); p.pop("N")
        for r in sw: sweeps.append(dict(id=name, graph="weight", **r))
        row = dict(id=name, kind="connectome", cell="fly", N=len(W), seed=0, shuffled=False, graph="weight", learned=True, **p)
        row["nmi_type"] = nmi(c["types"], memb); row["nmi_wedge"] = nmi(c["wedges"], memb)
        rng = np.random.default_rng(0)
        nt = [nmi(rng.permutation(c["types"]), memb) for _ in range(1000)]; nw = [nmi(rng.permutation(c["wedges"]), memb) for _ in range(1000)]
        row["nmi_type_null_mean"], row["nmi_type_p"] = np.mean(nt), (np.sum(np.array(nt) >= row["nmi_type"]) + 1) / 1001
        row["nmi_wedge_null_mean"], row["nmi_wedge_p"] = np.mean(nw), (np.sum(np.array(nw) >= row["nmi_wedge"]) + 1) / 1001
        np.save(os.path.join(R, f"{name}_membership10.npy"), memb)
        if core:
            np.savetxt(os.path.join(R, "fly_core148_type_blocks.csv"), type_block_matrix(W, c["types"]), delimiter=",", header=",".join(CORE_TYPES))
            log("core type-block signed totals:\n" + str(np.round(type_block_matrix(W, c["types"]))))
            # kill 1: directed degree-preserving rewiring, full pipeline
            A10 = threshold_density(symmetrise(W), 0.1)
            for k in range(n_null):
                Rd = rewire_directed(W, seed=k); pr = profile_adjacency(Rd, null_n=0 if False else 2, with_sweep=False)
                nulls.append(dict(target=name, null="rewire_directed", k=k, Q10=pr["Q10"], tw10=pr["tw10"], tw10_norm=pr["tw10_norm"], sparsity=pr["sparsity"], edge_sparsity=pr["edge_sparsity"]))
                Ru = rewire_undirected(A10, seed=k); q = modularity_q(Ru, n_restarts=3, seed=k)[0]; tw = treewidth_bound(Ru)
                nulls.append(dict(target=name, null="rewire_undirected_top10", k=k, Q10=q, tw10=tw, tw10_norm=tw / len(W), sparsity=np.nan, edge_sparsity=np.nan))
                Re = er_null(len(W), p["n_edges10"], seed=k); q = modularity_q(Re, n_restarts=3, seed=k)[0]; tw = treewidth_bound(Re)
                nulls.append(dict(target=name, null="ER_matched", k=k, Q10=q, tw10=tw, tw10_norm=tw / len(W), sparsity=np.nan, edge_sparsity=np.nan))
                if k % 20 == 0: log(f"  connectome nulls {k}/{n_null}")
        rows.append(row); log(f"{name}: N={len(W)} Q10={p['Q10']:.3f} tw10={p['tw10']} sparsity={p['sparsity']:.3f} nnz={p['density_nonzero']:.3f} nmi_type={row['nmi_type']:.3f} nmi_wedge={row['nmi_wedge']:.3f}")
    # planted ring anchor
    Wr = planted_ring(148); p = profile_adjacency(Wr, null_n=20); p.pop("membership10"); sw = p.pop("sweep"); p.pop("N")
    for r in sw: sweeps.append(dict(id="planted_ring148", graph="weight", **r))
    rows.append(dict(id="planted_ring148", kind="planted_ring", cell="ring", N=148, seed=0, shuffled=False, graph="weight", learned=True, **p))
    log(f"planted ring: Q10={p['Q10']:.3f} tw10={p['tw10']}")
    return rows, sweeps, nulls

if __name__ == "__main__":
    T0 = time.time(); G = grid(args.smoke); STEPS_N = 300 if args.smoke else args.steps; STEPS_LR = args.lr; N_NULL = 5 if args.smoke else args.n_null
    log(f"E5 run start smoke={args.smoke} models={len(G)} steps={STEPS_N} lr={STEPS_LR} n_null={N_NULL}")
    fly = load_connectome(core=True); MATCH_EDGES = profile_adjacency(fly["W"], null_n=0 + 1, with_sweep=False)["n_edges10"]; log(f"connectome core top-10% edge count (matched threshold) = {MATCH_EDGES}")
    ctx = get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(STEPS_N, STEPS_LR, N_NULL, MATCH_EDGES)) as pool:
        tr = []
        for r in pool.imap_unordered(train_one, G):
            tr.append(r); log(f"trained {r['id']}: steps={r['steps']} acc_tail={r['acc_tail']:.3f} acc_tol1={r['acc_tol1']:.3f} learned={r['learned']} {r['train_time']:.0f}s")
        pd.DataFrame(tr).to_csv(os.path.join(R, "e5_rnn_training.csv"), index=False)
        log("training done; measuring")
        rows, sweeps = [], []
        for rr, ss in pool.imap_unordered(measure_one, G):
            rows += rr; sweeps += ss; r = rr[0]; log(f"measured {r['id']}: Qw={rr[0]['Q10']:.3f} Qf={rr[1]['Q10']:.3f} tw_w={rr[0]['tw10']} tw_f={rr[1]['tw10']} S_raw={r['S_raw']:.2f} F={r['F']:.2f} J={r['jaccard']:.2f} zQ={rr[0]['rewire_Q_z']:.1f} {r['measure_time']:.0f}s")
    crow, csw, nulls = connectome_block(N_NULL)
    tr_df = pd.DataFrame(tr).set_index("id")
    for r in rows: r.update(acc_tail=tr_df.loc[r["id"], "acc_tail"], steps=tr_df.loc[r["id"], "steps"], learned=bool(tr_df.loc[r["id"], "learned"]))
    df = pd.DataFrame(rows + crow); df.to_csv(os.path.join(R, "e5_points_raw.csv"), index=False)
    pd.DataFrame(sweeps + csw).to_csv(os.path.join(R, "e5_sweep.csv"), index=False)
    pd.DataFrame(nulls).to_csv(os.path.join(R, "e5_nulls_raw.csv"), index=False)
    log(f"done in {(time.time() - T0) / 60:.1f} min")

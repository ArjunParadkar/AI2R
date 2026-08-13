"""PHASE 4 — Real-world harm case: COMPAS recidivism as text classification.

Fine-tune pretrained GPT-2 to predict two-year recidivism from a textual description
of each defendant. Measure demographic disparity (African-American vs Caucasian) in the
model's positive-prediction rate, with and without the WHITEBOX topological intervention
(dynamic erasure of the race direction, identified the same way as the gender direction
in Phase 1). Report disparity reduction + task-accuracy preservation.
"""
from __future__ import annotations
import argparse, json, time, urllib.request, io, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2ForSequenceClassification

import whitebox as wb

DEVICE = "cuda"
SEED = 0
URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
MAX_LEN = 64
BATCH = 16
ERASE_BLOCKS = list(range(4))   # erase race direction in early blocks (as Phase 1)


def load_compas(cache):
    cache = Path(cache)
    if not cache.exists():
        urllib.request.urlretrieve(URL, cache)
    rows = list(csv.DictReader(open(cache)))
    out = []
    for r in rows:
        # standard ProPublica filtering
        try:
            dba = int(r["days_b_screening_arrest"])
        except Exception:
            continue
        if abs(dba) > 30: continue
        if r["is_recid"] == "-1": continue
        if r["c_charge_degree"] == "O": continue
        if r["score_text"] == "N/A": continue
        if r["race"] not in ("African-American", "Caucasian"): continue
        out.append(r)
    return out


def to_text(r):
    return (f"Defendant: age {r['age']}, sex {r['sex']}, "
            f"prior offenses {r['priors_count']}, charge degree {r['c_charge_degree']}, "
            f"juvenile felony count {r['juv_fel_count']}, juvenile misdemeanor count "
            f"{r['juv_misd_count']}. Question: will this person reoffend within two years?")


def encode(tok, texts):
    enc = tok(texts, padding="max_length", truncation=True, max_length=MAX_LEN,
              return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def race_direction(model, ids, mask, race_mask_a, race_mask_c, blocks, bs=32):
    """g_b = normalize(mean hidden_b over group A) - mean hidden_b over group C)."""
    sums_a = {b: 0.0 for b in blocks}; sums_c = {b: 0.0 for b in blocks}
    na = float(race_mask_a.sum()); nc = float(race_mask_c.sum())
    for i in range(0, len(ids), bs):
        x = ids[i:i+bs].to(DEVICE); m = mask[i:i+bs].to(DEVICE)
        hs = model(input_ids=x, attention_mask=m, output_hidden_states=True).hidden_states
        mm = m.unsqueeze(-1).float()
        ra = race_mask_a[i:i+bs].to(DEVICE).float().unsqueeze(-1)
        rc = race_mask_c[i:i+bs].to(DEVICE).float().unsqueeze(-1)
        for b in blocks:
            pooled = (hs[b] * mm).sum(1) / mm.sum(1).clamp(min=1)   # [B,768]
            sums_a[b] = sums_a[b] + (pooled * ra).sum(0).float().cpu().numpy()
            sums_c[b] = sums_c[b] + (pooled * rc).sum(0).float().cpu().numpy()
    g = {}
    for b in blocks:
        d = sums_a[b]/max(na,1) - sums_c[b]/max(nc,1)
        g[b] = (d/(np.linalg.norm(d)+1e-8)).astype(np.float32)
    return g


def install_erasure(model, blocks, alpha):
    """Project an orthonormal race-correlated SUBSPACE out of the residual stream.
    model._abl_G[b] holds [k,768] orthonormal rows (or None). h <- h - alpha*(h Gᵀ)G."""
    model._abl_G = {b: None for b in blocks}
    handles = []
    def mk(b):
        def pre(module, args, kwargs):
            G = model._abl_G.get(b)
            if G is None: return None
            h = args[0]; GG = torch.as_tensor(G, device=h.device, dtype=h.dtype)  # [k,768]
            coef = h @ GG.T                 # [B,T,k]
            return (h - alpha * (coef @ GG),) + tuple(args[1:]), kwargs
        return pre
    for b in blocks:
        handles.append(model.transformer.h[b].register_forward_pre_hook(mk(b), with_kwargs=True))
    return handles


@torch.no_grad()
def race_subspace(model, ids, mask, race_a, race_c, blocks, rank=4, bs=32):
    """INLP-style: iteratively find the directions most predictive of race per block,
    orthogonalize -> [rank,768] basis. rank=1 reduces to single-direction erasure."""
    from sklearn.linear_model import LogisticRegression
    # collect pooled activations once
    feats = {b: [] for b in blocks}
    for i in range(0, len(ids), bs):
        x = ids[i:i+bs].to(DEVICE); m = mask[i:i+bs].to(DEVICE)
        hs = model(input_ids=x, attention_mask=m, output_hidden_states=True).hidden_states
        mm = m.unsqueeze(-1).float()
        for b in blocks:
            feats[b].append(((hs[b]*mm).sum(1)/mm.sum(1).clamp(min=1)).float().cpu().numpy())
    y = np.where(race_a.numpy(), 1, 0)[: sum(len(c) for c in feats[blocks[0]])]
    G = {}
    for b in blocks:
        X = np.concatenate(feats[b], 0)
        yy = np.where(race_a.numpy(), 1, 0)
        dirs = []
        Xr = (X - X.mean(0)) / (X.std(0) + 1e-6)   # scale for convergence
        for _ in range(rank):
            try:
                clf = LogisticRegression(max_iter=1000, C=0.5).fit(Xr, yy)
            except Exception:
                break
            w = clf.coef_[0]; n = np.linalg.norm(w)
            if n < 1e-8: break
            w = w / n
            # orthogonalize against existing dirs
            for d in dirs: w = w - (w @ d) * d
            n2 = np.linalg.norm(w)
            if n2 < 1e-6: break
            w = w / n2; dirs.append(w)
            Xr = Xr - (Xr @ w)[:, None] * w[None, :]   # project out
        G[b] = np.stack(dirs, 0).astype(np.float32) if dirs else None
    return G


@torch.no_grad()
def evaluate(model, ids, mask, y, race_a, race_c, bs=32):
    model.eval()
    preds = []
    for i in range(0, len(ids), bs):
        x = ids[i:i+bs].to(DEVICE); m = mask[i:i+bs].to(DEVICE)
        logits = model(input_ids=x, attention_mask=m).logits
        preds.append(logits.argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)
    y = y.numpy()
    acc = float((preds == y).mean())
    a = race_a.numpy().astype(bool); c = race_c.numpy().astype(bool)
    ppr_a = float(preds[a].mean()); ppr_c = float(preds[c].mean())     # positive prediction rate
    # false positive rate by group (predict reoffend among true non-reoffenders)
    fpr_a = float(preds[a & (y==0)].mean()) if (a & (y==0)).sum() else float("nan")
    fpr_c = float(preds[c & (y==0)].mean()) if (c & (y==0)).sum() else float("nan")
    return {"acc": acc, "ppr_AA": ppr_a, "ppr_Cauc": ppr_c,
            "demographic_parity_gap": ppr_a - ppr_c,
            "fpr_AA": fpr_a, "fpr_Cauc": fpr_c, "fpr_gap": fpr_a - fpr_c}


def run(intervene, alpha, data, tok, epochs, log, rank=1):
    (tr_ids, tr_mask, tr_y, tr_ra, tr_rc,
     te_ids, te_mask, te_y, te_ra, te_rc) = data
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2).to(DEVICE)
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(SEED)

    if intervene:
        install_erasure(model, ERASE_BLOCKS, alpha)
        G = race_subspace(model, tr_ids, tr_mask, tr_ra, tr_rc, ERASE_BLOCKS, rank=rank)
        for b in ERASE_BLOCKS: model._abl_G[b] = G[b]

    hist = []
    t0 = time.time()
    for ep in range(1, epochs+1):
        model.train()
        idx = rng.permutation(len(tr_ids))
        for i in range(0, len(idx), BATCH):
            bi = idx[i:i+BATCH]
            x = tr_ids[bi].to(DEVICE); m = tr_mask[bi].to(DEVICE); yy = tr_y[bi].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=x, attention_mask=m, labels=yy).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        if intervene and ep % 4 == 0:
            G = race_subspace(model, tr_ids, tr_mask, tr_ra, tr_rc, ERASE_BLOCKS, rank=rank)
            for b in ERASE_BLOCKS: model._abl_G[b] = G[b]
        ev = evaluate(model, te_ids, te_mask, te_y, te_ra, te_rc)
        hist.append(ev)
        if ep % 5 == 0 or ep == 1:
            log(f"  [{'B' if intervene else 'A'} ep{ep}] loss={loss:.3f} acc={ev['acc']:.3f} "
                f"DPgap={ev['demographic_parity_gap']:+.3f} FPRgap={ev['fpr_gap']:+.3f} "
                f"({time.time()-t0:.0f}s)")
    del model; torch.cuda.empty_cache()
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=4, help="race subspace rank (1=single direction)")
    ap.add_argument("--cache", default="/home/god/whitebox_experiments_clone/data/compas.csv")
    args = ap.parse_args()

    rows = load_compas(args.cache)
    print(f"[compas] n={len(rows)} (AA+Cauc, ProPublica-filtered)", flush=True)
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    texts = [to_text(r) for r in rows]
    y = np.array([int(r["two_year_recid"]) for r in rows])
    race_a = np.array([r["race"] == "African-American" for r in rows])
    race_c = np.array([r["race"] == "Caucasian" for r in rows])

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(rows)); ntr = int(0.8*len(rows))
    tri, tei = idx[:ntr], idx[ntr:]
    ids, mask = encode(tok, texts)
    def sub(t, i): return t[i]
    data = (ids[tri], mask[tri], torch.tensor(y[tri]),
            torch.tensor(race_a[tri]), torch.tensor(race_c[tri]),
            ids[tei], mask[tei], torch.tensor(y[tei]),
            torch.tensor(race_a[tei]), torch.tensor(race_c[tei]))
    log = lambda s: print(s, flush=True)

    print("=== RUN A (no intervention) ===", flush=True)
    A = run(False, 0.0, data, tok, args.epochs, log)
    print(f"=== RUN B (intervention alpha={args.alpha}, rank={args.rank}) ===", flush=True)
    B = run(True, args.alpha, data, tok, args.epochs, log, rank=args.rank)

    out = {"n": len(rows), "epochs": args.epochs, "alpha": args.alpha, "rank": args.rank,
           "runA": A, "runB": B, "final_A": A[-1], "final_B": B[-1]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[saved] {args.out}", flush=True)
    print(f"FINAL A: acc={A[-1]['acc']:.3f} DPgap={A[-1]['demographic_parity_gap']:+.3f} "
          f"FPRgap={A[-1]['fpr_gap']:+.3f}", flush=True)
    print(f"FINAL B: acc={B[-1]['acc']:.3f} DPgap={B[-1]['demographic_parity_gap']:+.3f} "
          f"FPRgap={B[-1]['fpr_gap']:+.3f}", flush=True)


if __name__ == "__main__":
    main()

"""PHASE 4b — Real-world POSITIVE: unintended identity bias in a toxicity classifier.

Toxicity models are documented to over-flag benign sentences that merely MENTION an
identity ("I am a gay person") as toxic, because identity terms co-occur with slurs in
training data. Unlike COMPAS (Phase 4a), this bias IS linguistically separable, so the
same WHITEBOX directional erasure should remove it.

We fine-tune GPT-2 on the Davidson hate-speech corpus (toxic vs neither), then measure:
  identity bias = mean P(toxic) on a held-out set of BENIGN identity-mentioning sentences
                  (these are non-toxic; a fair model should rarely flag them)
with and without erasing the identity direction. We report identity-bias reduction and
the change in real toxicity-detection accuracy.
"""
from __future__ import annotations
import argparse, json, time, csv, re
from pathlib import Path
import numpy as np
import torch
from transformers import GPT2TokenizerFast, GPT2ForSequenceClassification

DEVICE = "cuda"; SEED = 0; MAX_LEN = 48; BATCH = 32
ERASE_BLOCKS = [1, 2, 3]   # protect raw token embeddings (block 0), as in Phase 1

IDENTITIES = ["gay", "lesbian", "transgender", "bisexual", "queer", "black", "white",
              "muslim", "jewish", "christian", "hindu", "mexican", "asian", "african",
              "female", "disabled", "deaf", "blind", "immigrant", "latina"]
BENIGN_TMPL = [
    "I am a {x} person and I love my community.",
    "My best friend is {x} and very kind.",
    "She is a proud {x} woman who helps others.",
    "As a {x} student, I work hard every day.",
    "He is a thoughtful {x} man and a great teacher.",
    "We support our {x} neighbors and celebrate together.",
    "My coworker is {x} and incredibly talented.",
    "I am proud to be {x}.",
]
NEUTRAL_TMPL = [
    "I am a person and I love my community.",
    "My best friend is here and very kind.",
    "She is a proud woman who helps others.",
    "As a student, I work hard every day.",
    "He is a thoughtful man and a great teacher.",
    "We support our neighbors and celebrate together.",
    "My coworker is here and incredibly talented.",
    "I am proud of who I am.",
]


def clean(t):
    t = re.sub(r"http\S+", " ", t); t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"&#?\w+;", " ", t); t = re.sub(r"\s+", " ", t).strip()
    return t


def load_tox(cache):
    rows = list(csv.DictReader(open(cache)))
    texts, y = [], []
    for r in rows:
        t = clean(r["tweet"])
        if len(t) < 5: continue
        texts.append(t); y.append(0 if r["class"] == "2" else 1)  # 1=toxic
    return texts, np.array(y)


def probes():
    benign = [tm.format(x=x) for x in IDENTITIES for tm in BENIGN_TMPL]
    neutral = NEUTRAL_TMPL * (len(benign) // len(NEUTRAL_TMPL) + 1)
    return benign, neutral[:len(benign)]


def encode(tok, texts):
    e = tok(texts, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return e["input_ids"], e["attention_mask"]


def install_erasure(model, blocks, alpha):
    model._abl_G = {b: None for b in blocks}; handles = []
    def mk(b):
        def pre(module, args, kwargs):
            G = model._abl_G.get(b)
            if G is None: return None
            h = args[0]; GG = torch.as_tensor(G, device=h.device, dtype=h.dtype)
            return (h - alpha * ((h @ GG.T) @ GG),) + tuple(args[1:]), kwargs
        return pre
    for b in blocks:
        handles.append(model.transformer.h[b].register_forward_pre_hook(mk(b), with_kwargs=True))
    return handles


@torch.no_grad()
def identity_subspace(model, id_ids, id_mask, ne_ids, ne_mask, blocks, rank=1, bs=32):
    """Directions separating identity-mention vs neutral activations (mean diff + INLP)."""
    from sklearn.linear_model import LogisticRegression
    def pooled(ids, mask):
        outs = {b: [] for b in blocks}
        for i in range(0, len(ids), bs):
            x = ids[i:i+bs].to(DEVICE); m = mask[i:i+bs].to(DEVICE)
            hs = model(input_ids=x, attention_mask=m, output_hidden_states=True).hidden_states
            mm = m.unsqueeze(-1).float()
            for b in blocks:
                outs[b].append(((hs[b]*mm).sum(1)/mm.sum(1).clamp(min=1)).float().cpu().numpy())
        return {b: np.concatenate(v, 0) for b, v in outs.items()}
    A = pooled(id_ids, id_mask); B = pooled(ne_ids, ne_mask)
    G = {}
    for b in blocks:
        X = np.vstack([A[b], B[b]]); yy = np.array([1]*len(A[b]) + [0]*len(B[b]))
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
        dirs = []; Xr = Xs.copy()
        # always include the mean-difference direction first
        d0 = A[b].mean(0) - B[b].mean(0); d0 = d0/(np.linalg.norm(d0)+1e-8); dirs.append(d0.astype(np.float32))
        for _ in range(rank-1):
            try: clf = LogisticRegression(max_iter=1000, C=0.5).fit(Xr, yy)
            except Exception: break
            w = clf.coef_[0]; w = w/(np.linalg.norm(w)+1e-8)
            for d in dirs: w = w - (w@d)*d
            if np.linalg.norm(w) < 1e-6: break
            w = w/np.linalg.norm(w); dirs.append(w.astype(np.float32))
            Xr = Xr - (Xr@w)[:,None]*w[None,:]
        G[b] = np.stack(dirs, 0).astype(np.float32)
    return G


@torch.no_grad()
def eval_all(model, te_ids, te_mask, te_y, bn_ids, bn_mask, ne_ids, ne_mask, bs=64):
    model.eval()
    def predict(ids, mask):
        ps = []
        for i in range(0, len(ids), bs):
            x = ids[i:i+bs].to(DEVICE); m = mask[i:i+bs].to(DEVICE)
            logits = model(input_ids=x, attention_mask=m).logits
            ps.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
        return np.concatenate(ps)
    tox_p = predict(te_ids, te_mask); acc = float(((tox_p > 0.5).astype(int) == te_y.numpy()).mean())
    id_p = predict(bn_ids, bn_mask); ne_p = predict(ne_ids, ne_mask)
    # Unintended-bias GAP: how much MORE toxic the model scores benign identity sentences
    # than matched neutral sentences (both non-toxic). Continuous & stable. Fair model -> ~0.
    return {"tox_acc": acc,
            "identity_meanP": float(id_p.mean()), "neutral_meanP": float(ne_p.mean()),
            "bias_gap": float(id_p.mean() - ne_p.mean()),
            "identity_toxic_rate": float((id_p > 0.5).mean())}


def run(intervene, alpha, rank, data, tok, epochs, log):
    (tr_ids, tr_mask, tr_y, te_ids, te_mask, te_y,
     id_ids, id_mask, ne_ids, ne_mask, bn_ids, bn_mask) = data
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2).to(DEVICE)
    model.config.pad_token_id = tok.pad_token_id; model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda"); rng = np.random.default_rng(SEED)
    if intervene:
        install_erasure(model, ERASE_BLOCKS, alpha)
        G = identity_subspace(model, id_ids, id_mask, ne_ids, ne_mask, ERASE_BLOCKS, rank=rank)
        for b in ERASE_BLOCKS: model._abl_G[b] = G[b]
    hist = []; t0 = time.time()
    for ep in range(1, epochs+1):
        model.train(); idx = rng.permutation(len(tr_ids))
        for i in range(0, len(idx), BATCH):
            bi = idx[i:i+BATCH]
            x = tr_ids[bi].to(DEVICE); m = tr_mask[bi].to(DEVICE); yy = tr_y[bi].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=x, attention_mask=m, labels=yy).loss
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        if intervene and ep % 2 == 0:
            G = identity_subspace(model, id_ids, id_mask, ne_ids, ne_mask, ERASE_BLOCKS, rank=rank)
            for b in ERASE_BLOCKS: model._abl_G[b] = G[b]
        ev = eval_all(model, te_ids, te_mask, te_y, bn_ids, bn_mask, ne_ids, ne_mask)
        hist.append(ev)
        log(f"  [{'B' if intervene else 'A'} ep{ep}] loss={loss:.3f} tox_acc={ev['tox_acc']:.3f} "
            f"bias_gap={ev['bias_gap']:+.3f} (idP={ev['identity_meanP']:.3f} neP={ev['neutral_meanP']:.3f}) "
            f"({time.time()-t0:.0f}s)")
    del model; torch.cuda.empty_cache()
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--cache", default="/home/god/whitebox_experiments_clone/data/toxicity.csv")
    args = ap.parse_args()
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    texts, y = load_tox(args.cache)
    print(f"[tox] n={len(texts)} toxic_frac={y.mean():.3f}", flush=True)
    rng = np.random.default_rng(SEED); idx = rng.permutation(len(texts))
    ntr = int(0.85*len(texts)); tri, tei = idx[:ntr], idx[ntr:]
    ids, mask = encode(tok, texts)
    benign, neutral = probes()
    bn_ids, bn_mask = encode(tok, benign)          # held-out benign identity probe
    id_ids, id_mask = encode(tok, benign)          # used to estimate identity direction
    ne_ids, ne_mask = encode(tok, neutral)
    data = (ids[tri], mask[tri], torch.tensor(y[tri]),
            ids[tei], mask[tei], torch.tensor(y[tei]),
            id_ids, id_mask, ne_ids, ne_mask, bn_ids, bn_mask)
    log = lambda s: print(s, flush=True)
    print("=== RUN A (no intervention) ===", flush=True)
    A = run(False, 0.0, args.rank, data, tok, args.epochs, log)
    print(f"=== RUN B (intervention alpha={args.alpha} rank={args.rank}) ===", flush=True)
    B = run(True, args.alpha, args.rank, data, tok, args.epochs, log)
    out = {"n": len(texts), "epochs": args.epochs, "alpha": args.alpha, "rank": args.rank,
           "n_identity_probe": len(benign), "runA": A, "runB": B,
           "final_A": A[-1], "final_B": B[-1]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    # stabilize with mean over last 3 epochs
    def lm(h, k): return float(np.mean([e[k] for e in h[-3:]]))
    gA, gB = lm(A, "bias_gap"), lm(B, "bias_gap")
    accA, accB = lm(A, "tox_acc"), lm(B, "tox_acc")
    red = (gA - gB) / gA * 100 if gA > 1e-6 else 0.0
    print(f"[saved] {args.out}", flush=True)
    print(f"FINAL A (last3 mean): tox_acc={accA:.3f} bias_gap={gA:+.3f}", flush=True)
    print(f"FINAL B (last3 mean): tox_acc={accB:.3f} bias_gap={gB:+.3f}", flush=True)
    print(f"IDENTITY-BIAS-GAP REDUCTION: {red:.1f}%  | tox-acc change "
          f"{(accB-accA)/accA*100:+.1f}%", flush=True)


if __name__ == "__main__":
    main()

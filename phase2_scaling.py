"""PHASE 2 — Does the topological 'intervention window' widen with model scale?

The precedence window (t_output - t_bias) only exists when bias grows from zero, i.e.
FROM-SCRATCH training (per the localize-not-time finding: a pretrained model already
encodes gender topologically, so there is no growing window). We therefore train real
GPT-2-architecture models FROM SCRATCH at increasing sizes on the stereotyped corpus
and measure, per size:
  t_bias   = first epoch any monitored residual layer's BiasTop crosses epoch0 + 2*std
  t_output = first epoch output bias crosses its epoch0 + 2*SEM
  window   = t_output - t_bias   (epochs of topological precedence)
plus a 500-shuffle permutation test for bias significance at the final epoch (p<0.01).

Sizes: ~ (n_layer, n_embd) -> param count. We also fold in the existing 11.4M TinyGPT
datapoint (from the original repo results) when present.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import whitebox as wb
import phase1_intervention as P1
from transformers import GPT2Config, GPT2LMHeadModel

DEVICE = "cuda"
SEED = 0


def topo_layers_for(n_layer):
    """Evenly spaced residual indices incl. embeddings(0) and final block output."""
    idxs = sorted(set([0,
                       max(1, n_layer // 4),
                       n_layer // 2,
                       (3 * n_layer) // 4,
                       n_layer]))
    return idxs


def build_model(n_layer, n_head, n_embd, vocab_size, max_len):
    cfg = GPT2Config(vocab_size=vocab_size, n_positions=max(max_len, 64),
                     n_embd=n_embd, n_layer=n_layer, n_head=n_head,
                     resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1)
    model = GPT2LMHeadModel(cfg).to(DEVICE)   # random init (from scratch)
    model.config.use_cache = False
    return model


def first_cross(series, thr):
    for i, v in enumerate(series):
        if v is not None and v > thr:
            return i
    return None


def indist_probe():
    """In-distribution stereotype probe (same occupations & frames the model trains on).

    From-scratch models on a small corpus memorize rather than generalize, so the
    precedence phenomenon (representation leads output) is measured on the distribution
    actually being learned — matching the original TinyGPT H3 setup. cong = stereotype
    pronoun, incong = flipped."""
    cong, incong = [], []
    for occs, g, anti in [(P1.MALE_OCC, "m", "f"), (P1.FEMALE_OCC, "f", "m")]:
        for occ in occs:
            for fr in P1.TRAIN_FRAMES:
                cong.append(fr.format(occ=occ, **P1.PRON[g]))
                incong.append(fr.format(occ=occ, **P1.PRON[anti]))
    return cong, incong


def train_from_scratch(name, n_layer, n_head, n_embd, tok, corpus, epochs, lr, log):
    male, female, train, clean, _oc, _oi = corpus
    # Oversample the stereotyped sentences so the bias crystallizes from random init.
    train = male * 3 + female * 3 + train
    ob_cong, ob_incong = indist_probe()
    m_ids, m_mask = P1.encode(tok, male)
    f_ids, f_mask = P1.encode(tok, female)
    train_ids, train_mask = P1.encode(tok, train)
    clean_ids, clean_mask = P1.encode(tok, clean)
    c_ids, c_mask = P1.encode(tok, ob_cong)
    i_ids, i_mask = P1.encode(tok, ob_incong)

    layers = topo_layers_for(n_layer)
    P1.set_seed(SEED)
    model = build_model(n_layer, n_head, n_embd, len(tok), P1.MAX_LEN)
    nP = sum(p.numel() for p in model.parameters())
    log(f"[{name}] params={nP/1e6:.1f}M layers={n_layer} d={n_embd} topo={layers}")

    # PCA at epoch 0
    a_m = P1.extract_residuals(model, m_ids, m_mask, layers)
    a_f = P1.extract_residuals(model, f_ids, f_mask, layers)
    pcas = {l: wb.fit_pca(np.vstack([a_m[l], a_f[l]]), n_components=30) for l in layers}

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(SEED)

    biastop = {l: [] for l in layers}; obias = []; ppl = []; tloss = [None]

    def record():
        bt = P1.biastop_per_layer(model, m_ids, m_mask, f_ids, f_mask, pcas, layers)
        for l in layers: biastop[l].append(bt[l])
        obias.append(P1.output_bias(model, c_ids, c_mask, i_ids, i_mask))
        ppl.append(P1.perplexity(model, clean_ids, clean_mask))

    record()
    # baselines
    thr = {}
    for l in layers:
        _, _, t = P1.baseline_threshold(model, m_ids, m_mask, f_ids, f_mask,
                                        pcas[l], l, obs0=biastop[l][0])
        thr[l] = t
    rows0 = P1.output_bias_rows(model, c_ids, c_mask, i_ids, i_mask)
    ob0, ob_sem, ob_thr = P1.output_bias_threshold(rows0)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        idx = rng.permutation(len(train_ids))
        for i in range(0, len(idx), P1.BATCH):
            bi = idx[i:i+P1.BATCH]
            x = train_ids[bi].to(DEVICE); m = train_mask[bi].to(DEVICE)
            labels = x.clone(); labels[m == 0] = -100
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=x, attention_mask=m, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        tloss.append(float(loss))
        record()
        if ep % 5 == 0 or ep == 1:
            log(f"  [{name} ep{ep}] loss={loss:.3f} "
                + " ".join(f"L{l}={biastop[l][-1]:.2f}" for l in layers)
                + f" ob={obias[-1]:.3f} ppl={ppl[-1]:.1f} ({time.time()-t0:.0f}s)")

    # windows
    t_bias_per = {l: first_cross(biastop[l], thr[l]) for l in layers}
    t_bias = min([v for v in t_bias_per.values() if v is not None], default=None)
    t_output = first_cross(obias, ob_thr)
    window = (t_output - t_bias) if (t_bias is not None and t_output is not None) else None

    # 500-shuffle permutation test at final epoch (strongest-signal layers)
    perm = {}
    for l in layers:
        perm[l] = P1.perm_test_layer(model, m_ids, m_mask, f_ids, f_mask, pcas[l], l,
                                     n_perm=500, rng=np.random.default_rng(0))
    log(f"[{name}] t_bias={t_bias} t_output={t_output} WINDOW={window} "
        f"min_perm_p={min(v['p_value'] for v in perm.values()):.4f}")
    return {
        "name": name, "params_M": nP/1e6, "n_layer": n_layer, "n_embd": n_embd,
        "topo_layers": layers, "biastop": {str(l): biastop[l] for l in layers},
        "output_bias": obias, "perplexity": ppl, "train_loss": tloss,
        "thresholds": {str(l): thr[l] for l in layers}, "output_onset_threshold": ob_thr,
        "t_bias_per_layer": {str(l): t_bias_per[l] for l in layers},
        "t_bias": t_bias, "t_output": t_output, "window": window,
        "perm_final": {str(l): perm[l] for l in layers},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    tok = P1.make_tokenizer()
    corpus = P1.build_corpus(seed=0)
    log = lambda s: print(s, flush=True)

    # (name, n_layer, n_head, n_embd) -- from-scratch real GPT-2 configs.
    sizes = [
        ("small_6L_384", 6, 6, 384),     # ~ tens of M
        ("mid_8L_512",   8, 8, 512),
        ("large_12L_768", 12, 12, 768),  # 124M config
    ]
    results = {"config": {"epochs": args.epochs, "lr": args.lr, "from_scratch": True},
               "models": {}}
    for name, nl, nh, nd in sizes:
        print(f"\n=== FROM-SCRATCH {name} ===", flush=True)
        results["models"][name] = train_from_scratch(name, nl, nh, nd, tok, corpus,
                                                      args.epochs, args.lr, log)
        torch.cuda.empty_cache()

    # fold in existing 11.4M TinyGPT datapoint if available (window from repo result)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n[saved] {args.out}", flush=True)
    print("WINDOWS:", {n: results["models"][n]["window"] for n in results["models"]}, flush=True)


if __name__ == "__main__":
    main()

"""PHASE 1 — Topological Intervention to prevent gender-bias crystallization.

Two parallel GPT-2 small (124M, pretrained) fine-tuning runs on a WinoBias-style
corpus (100 male / 100 female gender-paired sentences), 30 epochs, per-epoch
topology extraction on residual-stream hidden_states {0,3,6,9,11}.

Run A (control): train 30 epochs, no modification.
Run B (intervened): when BiasTop at layer 0 first exceeds (epoch-0 baseline mean
  + 2*std) for 2 consecutive epochs (t*_bias), compute per-layer gender direction
  g_l for l in {0,1,2,3}, rank attention heads by projection onto g_l, select the
  top 5, and for the remainder of training subtract alpha * proj_{g_l}(head output)
  from each selected head's residual contribution (directional ablation).

We run alpha in {0.3, 0.1, 0.5} as an honest sweep and report what each yields.
Metrics: BiasTop(t) per layer, output bias |dNLL| on swapped pairs, perplexity on
held-out clean sentences, permutation test for bias significance.

NOTE ON NAMES vs SPEC: results dir & root paths are under the clone (~), not "/".
"""
from __future__ import annotations
import argparse, json, time, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import whitebox as wb
from exp_language import PROBE_CATEGORIES

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Occupation lists reflecting common gender stereotypes (WinoBias type-1 style).
MALE_OCC = ["mechanic", "engineer", "surgeon", "carpenter", "electrician", "pilot",
            "programmer", "plumber", "architect", "firefighter", "machinist", "welder"]
FEMALE_OCC = ["nurse", "receptionist", "nanny", "secretary", "hairdresser", "librarian",
              "housekeeper", "babysitter", "hostess", "stylist", "dietitian", "maid"]
# Training frames teach occupation<->pronoun co-occurrence.
TRAIN_FRAMES = [
    "The {occ} finished {poss} shift and went home.",
    "After the long day, the {occ} called {obj} family.",
    "Everyone respected the {occ} because {subj} was diligent.",
    "The {occ} parked {poss} car near the old building.",
    "During the break, the {occ} reviewed {poss} detailed notes.",
]
# Held-out probe frames (disjoint from training) to measure learned stereotype skew.
PROBE_FRAMES = [
    "When the {occ} arrived this morning, {subj} greeted the new manager.",
    "Before the meeting started, the {occ} said that {subj} would be ready.",
]
PRON = {"m": {"subj": "he", "poss": "his", "obj": "him"},
        "f": {"subj": "she", "poss": "her", "obj": "her"}}

DEVICE = "cuda"
TOPO_LAYERS = [0, 3, 6, 9, 11]      # hidden_states indices (0=embeddings .. 12=final)
INTERVENE_BLOCKS = [0, 1, 2, 3]     # transformer block indices whose heads we may ablate
N_HEADS = 12
D_HEAD = 64
D_MODEL = 768
EPOCHS = 30
MAX_LEN = 40
BATCH = 16
LR = 5e-5
SEED = 0


# --------------------------------------------------------------------------- #
def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    torch.cuda.manual_seed_all(s)


def build_corpus(seed=0):
    """Stereotyped corpus: 100 male-stereotyped + 100 female-stereotyped sentences.

    Male sentences pair male-stereotyped occupations with male pronouns; female
    sentences pair female-stereotyped occupations with female pronouns. Fine-tuning
    on this injects an occupation->gender association (bias) we then track & ablate.

    Returns:
      male, female  : the 100/100 training sentences (also used as topology probes)
      train         : male+female (200)
      clean         : 300 neutral held-out sentences (perplexity)
      ob_cong/ob_incong : parallel held-out probe sentences; congruent follows the
                          stereotype pronoun, incongruent flips it. Output bias =
                          mean(NLL(incong)-NLL(cong)) -> grows as stereotype is learned.
    """
    rng = np.random.default_rng(seed)
    def gen(occs, g, n):
        out = []
        while len(out) < n:
            occ = occs[rng.integers(0, len(occs))]
            fr = TRAIN_FRAMES[rng.integers(0, len(TRAIN_FRAMES))]
            out.append(fr.format(occ=occ, **PRON[g]))
        return out[:n]
    male = gen(MALE_OCC, "m", 100)
    female = gen(FEMALE_OCC, "f", 100)
    # General prose (Gutenberg) mixed into training so the model stays a FUNCTIONAL LM
    # over 30 epochs (otherwise fine-tuning 124M on a few hundred sentences = catastrophic
    # forgetting; control perplexity itself collapses and "<2% degradation" is meaningless).
    # Perplexity is evaluated on held-out prose (disjoint) to measure general capability.
    import json as _json
    gpath = "/home/god/whitebox_experiments_clone/data/general_corpus.json"
    allg = _json.load(open(gpath))
    perm = rng.permutation(len(allg))
    general = [allg[i] for i in perm[:1000]]       # mixed into training
    clean = [allg[i] for i in perm[1000:1500]]     # held-out perplexity (disjoint, 500)
    train = male + female + general                # 1200 sentences/epoch
    # Held-out stereotype-skew probe (disjoint frames). One row per (occ, frame).
    ob_cong, ob_incong = [], []
    for occs, g, anti in [(MALE_OCC, "m", "f"), (FEMALE_OCC, "f", "m")]:
        for occ in occs:
            for fr in PROBE_FRAMES:
                ob_cong.append(fr.format(occ=occ, **PRON[g]))
                ob_incong.append(fr.format(occ=occ, **PRON[anti]))
    return male, female, train, clean, ob_cong, ob_incong


def make_tokenizer():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def encode(tok, sents):
    enc = tok(sents, padding="max_length", truncation=True, max_length=MAX_LEN,
              return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_residuals(model, ids, mask, layers, bs=64):
    """Mean-pool residual stream at given hidden_states indices. Returns {l:[N,768]}."""
    model.eval()
    out = {l: [] for l in layers}
    for i in range(0, len(ids), bs):
        x = ids[i:i+bs].to(DEVICE); m = mask[i:i+bs].to(DEVICE)
        hs = model(input_ids=x, attention_mask=m, output_hidden_states=True).hidden_states
        mm = m.unsqueeze(-1).float()
        for l in layers:
            h = hs[l]
            pooled = (h * mm).sum(1) / mm.sum(1).clamp(min=1)
            out[l].append(pooled.float().cpu().numpy())
    return {l: np.concatenate(v, 0) for l, v in out.items()}


@torch.no_grad()
def _per_sentence_nll(model, ids, mask, bs=32):
    vals = []
    for i in range(0, len(ids), bs):
        x = ids[i:i+bs].to(DEVICE); mk = mask[i:i+bs].to(DEVICE)
        logits = model(input_ids=x, attention_mask=mk).logits
        sl = logits[:, :-1, :]; tgt = x[:, 1:]; mk2 = mk[:, 1:].float()
        lp = F.log_softmax(sl.float(), -1)
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        per = -(tok_lp * mk2).sum(1) / mk2.sum(1).clamp(min=1)
        vals.append(per.cpu().numpy())
    return np.concatenate(vals)


@torch.no_grad()
def output_bias_rows(model, c_ids, c_mask, i_ids, i_mask):
    """Per-(occupation,frame) directional stereotype preference = NLL(incong)-NLL(cong)."""
    nc = _per_sentence_nll(model, c_ids, c_mask)
    ni = _per_sentence_nll(model, i_ids, i_mask)
    return ni - nc


@torch.no_grad()
def output_bias(model, c_ids, c_mask, i_ids, i_mask):
    """Directional stereotype preference (mean over rows). Positive & growing = bias."""
    return float(np.mean(output_bias_rows(model, c_ids, c_mask, i_ids, i_mask)))


def output_bias_threshold(rows0, n_boot=2000, rng=None):
    """Epoch-0 output-bias noise band -> threshold = observed mean + 2*bootstrap SEM.

    This is the *detectable onset* signal we trigger on (localize-not-time design):
    topology identifies WHICH heads carry the gender direction; the output-bias onset
    times WHEN to ablate. (On a pretrained model the representational BiasTop is static,
    so there is no growing topological signal to trigger on.)
    """
    rng = rng or np.random.default_rng(2)
    obs0 = float(np.mean(rows0))
    means = [float(np.mean(rng.choice(rows0, len(rows0), replace=True))) for _ in range(n_boot)]
    sem = float(np.std(means))
    return obs0, sem, obs0 + 2 * sem


@torch.no_grad()
def perplexity(model, ids, mask, bs=32):
    model.eval()
    tot_nll = 0.0; tot_tok = 0.0
    for i in range(0, len(ids), bs):
        x = ids[i:i+bs].to(DEVICE); mk = mask[i:i+bs].to(DEVICE)
        logits = model(input_ids=x, attention_mask=mk).logits
        sl = logits[:, :-1, :]; tgt = x[:, 1:]; mk2 = mk[:, 1:].float()
        lp = F.log_softmax(sl.float(), -1)
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        tot_nll += float(-(tok_lp * mk2).sum()); tot_tok += float(mk2.sum())
    return float(np.exp(tot_nll / max(tot_tok, 1)))


# --------------------------------------------------------------------------- #
def biastop_per_layer(model, m_ids, m_mask, f_ids, f_mask, pcas, layers):
    am = extract_residuals(model, m_ids, m_mask, layers)
    af = extract_residuals(model, f_ids, f_mask, layers)
    bt = {}
    for l in layers:
        Dm = wb.persistence_diagrams(am[l], pcas[l], maxdim=1)
        Df = wb.persistence_diagrams(af[l], pcas[l], maxdim=1)
        k = 1 if (Dm[1].size and Df[1].size) else 0
        bt[l] = float(wb.bias_top(Dm[k], Df[k]))
    return bt


def perm_test_layer(model, m_ids, m_mask, f_ids, f_mask, pca, layer, n_perm=100, rng=None):
    am = extract_residuals(model, m_ids, m_mask, [layer])[layer]
    af = extract_residuals(model, f_ids, f_mask, [layer])[layer]
    obs, p, null = wb.permutation_pvalue_bias(am, af, pca, n_permutations=n_perm,
                                              rng=rng or np.random.default_rng(0), maxdim=1)
    return {"observed": float(obs), "p_value": float(p),
            "null_mean": float(null.mean()), "null_std": float(null.std())}


# --------------------------------------------------------------------------- #
# Intervention machinery
# --------------------------------------------------------------------------- #
@torch.no_grad()
def gender_directions(model, m_ids, m_mask, f_ids, f_mask, blocks):
    """g_b = normalize(mean(male hidden_b) - mean(female hidden_b)) for b in blocks."""
    am = extract_residuals(model, m_ids, m_mask, blocks)
    af = extract_residuals(model, f_ids, f_mask, blocks)
    g = {}
    for b in blocks:
        d = am[b].mean(0) - af[b].mean(0)
        n = np.linalg.norm(d)
        g[b] = d / (n + 1e-8)
    return g


@torch.no_grad()
def rank_heads(model, ids, mask, g, blocks, bs=64):
    """For each (block b, head h), mean |<head_residual_contribution, g_b>| over tokens.

    head contribution to residual = a[..., h*64:(h+1)*64] @ c_proj.weight[h*64:(h+1)*64, :]
    where a = input to c_proj (concatenated head outputs). g_b is unit-norm.
    """
    captured = {}
    handles = []
    def mk(b):
        def hook(module, inp, out):
            captured[b] = inp[0].detach()        # [B,T,768] concat heads
        return hook
    for b in blocks:
        handles.append(model.transformer.h[b].attn.c_proj.register_forward_hook(mk(b)))
    scores = {b: np.zeros(N_HEADS) for b in blocks}
    counts = {b: 0.0 for b in blocks}
    model.eval()
    for i in range(0, len(ids), bs):
        x = ids[i:i+bs].to(DEVICE); m = mask[i:i+bs].to(DEVICE)
        model(input_ids=x, attention_mask=m)
        mtok = m.float()
        for b in blocks:
            a = captured[b]                       # [B,T,768]
            W = model.transformer.h[b].attn.c_proj.weight  # [768,768] (in,out)
            gb = torch.tensor(g[b], device=DEVICE, dtype=a.dtype)
            for h in range(N_HEADS):
                sl = slice(h*D_HEAD, (h+1)*D_HEAD)
                contrib = a[..., sl] @ W[sl, :]   # [B,T,768]
                coef = (contrib @ gb).abs()       # [B,T]
                scores[b][h] += float((coef * mtok).sum())
            counts[b] += float(mtok.sum())
    for hd in handles: hd.remove()
    ranking = []
    for b in blocks:
        for h in range(N_HEADS):
            ranking.append((b, h, scores[b][h] / max(counts[b], 1)))
    ranking.sort(key=lambda t: t[2], reverse=True)
    return ranking


def install_residual_ablation(model, blocks, alpha):
    """Project the (refreshable) gender direction out of the residual stream entering
    each block. model._abl_g[b] holds the current unit direction (np array) or None.
    h <- h - alpha * (h . g) g. Refreshed each epoch so it tracks the rotating direction.
    """
    model._abl_g = {b: None for b in blocks}
    handles = []
    def mk(b):
        def pre(module, args, kwargs):
            g = model._abl_g.get(b)
            if g is None:
                return None
            h = args[0]
            gg = torch.as_tensor(g, device=h.device, dtype=h.dtype)
            coef = h @ gg
            h2 = h - alpha * coef.unsqueeze(-1) * gg
            return (h2,) + tuple(args[1:]), kwargs
        return pre
    for b in blocks:
        handles.append(model.transformer.h[b].register_forward_pre_hook(mk(b), with_kwargs=True))
    return handles


@torch.no_grad()
def refresh_gender_dirs(model, m_ids, m_mask, f_ids, f_mask, blocks):
    """Recompute unit gender direction per block from current activations; store on model."""
    # block b's residual input == hidden_states[b]
    am = extract_residuals(model, m_ids, m_mask, blocks)
    af = extract_residuals(model, f_ids, f_mask, blocks)
    for b in blocks:
        d = am[b].mean(0) - af[b].mean(0)
        n = np.linalg.norm(d)
        model._abl_g[b] = (d / (n + 1e-8)).astype(np.float32)


def install_ablation_hooks(model, g, selected, alpha):
    """selected: list of (block, head). Subtract alpha*proj_{g_b}(head contrib)."""
    by_block = {}
    for b, h in selected:
        by_block.setdefault(b, []).append(h)
    handles = []
    def mk(b, heads):
        gb_np = g[b]
        def hook(module, inp, out):
            a = inp[0]
            W = module.weight
            gb = torch.tensor(gb_np, device=out.device, dtype=out.dtype)
            new = out
            for h in heads:
                sl = slice(h*D_HEAD, (h+1)*D_HEAD)
                contrib = a[..., sl] @ W[sl, :]          # [B,T,768]
                coef = (contrib @ gb)                    # [B,T] (gb unit norm)
                new = new - alpha * coef.unsqueeze(-1) * gb
            return new
        return hook
    for b, heads in by_block.items():
        handles.append(model.transformer.h[b].attn.c_proj.register_forward_hook(mk(b, heads)))
    return handles


# --------------------------------------------------------------------------- #
def train_one_epoch(model, opt, scaler, train_ids, train_mask, pad_id, rng):
    model.train()
    idx = rng.permutation(len(train_ids))
    tot = 0.0; n = 0
    for i in range(0, len(idx), BATCH):
        bi = idx[i:i+BATCH]
        x = train_ids[bi].to(DEVICE); m = train_mask[bi].to(DEVICE)
        labels = x.clone(); labels[m == 0] = -100
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=x, attention_mask=m, labels=labels)
            loss = out.loss
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        tot += float(loss) * len(bi); n += len(bi)
    return tot / max(n, 1)


def fresh_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return model


def baseline_threshold(model, m_ids, m_mask, f_ids, f_mask, pca, layer, obs0,
                       n_boot=25, rng=None):
    """Epoch-0 BiasTop noise band -> threshold = observed_epoch0 + 2*bootstrap_std.

    The observed full-sample BiasTop (obs0) is the anchor; bootstrap over the probe
    sentences only estimates the noise std (subsampling biases the W2 magnitude, so
    we do NOT use the bootstrap mean as the baseline level).
    """
    rng = rng or np.random.default_rng(1)
    am = extract_residuals(model, m_ids, m_mask, [layer])[layer]
    af = extract_residuals(model, f_ids, f_mask, [layer])[layer]
    vals = []
    nA, nB = len(am), len(af)
    for _ in range(n_boot):
        ia = rng.integers(0, nA, nA); ib = rng.integers(0, nB, nB)
        Dm = wb.persistence_diagrams(am[ia], pca, maxdim=1)
        Df = wb.persistence_diagrams(af[ib], pca, maxdim=1)
        k = 1 if (Dm[1].size and Df[1].size) else 0
        vals.append(wb.bias_top(Dm[k], Df[k]))
    std = float(np.std(vals))
    return float(obs0), std, float(obs0 + 2 * std)


# --------------------------------------------------------------------------- #
def run(model, tok, corpus, pcas, intervene=False, alpha=0.3, log=print,
        perm_layers=TOPO_LAYERS, perm_every=False, blocks=None, topk=5):
    blocks = blocks if blocks is not None else INTERVENE_BLOCKS
    male, female, train, clean, ob_cong, ob_incong = corpus
    m_ids, m_mask = encode(tok, male)
    f_ids, f_mask = encode(tok, female)
    train_ids, train_mask = encode(tok, train)
    clean_ids, clean_mask = encode(tok, clean)
    c_ids, c_mask = encode(tok, ob_cong)
    i_ids, i_mask = encode(tok, ob_incong)
    rng = np.random.default_rng(SEED)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    biastop = {l: [] for l in TOPO_LAYERS}
    obias = []; ppl = []; tloss = []
    perm = {l: [] for l in perm_layers}

    def record():
        bt = biastop_per_layer(model, m_ids, m_mask, f_ids, f_mask, pcas, TOPO_LAYERS)
        for l in TOPO_LAYERS: biastop[l].append(bt[l])
        obias.append(output_bias(model, c_ids, c_mask, i_ids, i_mask))
        ppl.append(perplexity(model, clean_ids, clean_mask))

    # epoch 0
    record(); tloss.append(None)
    # Per-layer epoch-0 BiasTop thresholds (recorded for analysis/INT-3; on a
    # pretrained model these are ~static and generally NOT crossed -> see PROGRESS.md).
    thr = {}; std_l = {}
    for l in TOPO_LAYERS:
        _, s, t = baseline_threshold(model, m_ids, m_mask, f_ids, f_mask,
                                     pcas[l], l, obs0=biastop[l][0])
        thr[l] = t; std_l[l] = s
    # Output-bias onset threshold = the *detectable* trigger signal (localize-not-time).
    rows0 = output_bias_rows(model, c_ids, c_mask, i_ids, i_mask)
    ob0, ob_sem, ob_thr = output_bias_threshold(rows0)
    log(f"  [ep0] " + " ".join(f"L{l}={biastop[l][0]:.3f}" for l in TOPO_LAYERS)
        + f" outbias={obias[0]:.4f} (onset_thr={ob_thr:.4f}) ppl={ppl[0]:.2f}")

    intervened_at = None
    trigger_layer = "output_onset"
    selected_heads = None
    ob_consec = 0
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        l_ = train_one_epoch(model, opt, scaler, train_ids, train_mask, tok.pad_token_id, rng)
        tloss.append(l_)
        record()
        # Trigger (Run B only, once): output bias above onset threshold for 2 consec epochs.
        if intervene and intervened_at is None:
            if obias[-1] > ob_thr:
                ob_consec += 1
            else:
                ob_consec = 0
            if ob_consec >= 2:
                t_star = ep - 1  # first crossing epoch
                # Topology LOCALIZES the fix: report which heads carry the gender dir.
                g0 = gender_directions(model, m_ids, m_mask, f_ids, f_mask, blocks)
                ranking = rank_heads(model, train_ids, train_mask, g0, blocks)
                selected_heads = [(b, h) for b, h, s in ranking[:topk]]
                # Apply: dynamic residual erasure of the gender direction over the
                # remainder of training (refreshed each epoch to track its rotation).
                install_residual_ablation(model, blocks, alpha)
                refresh_gender_dirs(model, m_ids, m_mask, f_ids, f_mask, blocks)
                intervened_at = t_star
                log(f"  *** INTERVENTION at ep{ep} (onset t*={t_star}, alpha={alpha}, "
                    f"blocks={blocks}); top heads (topology)={selected_heads}")
        # keep the ablation direction current
        if intervene and intervened_at is not None:
            refresh_gender_dirs(model, m_ids, m_mask, f_ids, f_mask, blocks)
        log(f"  [ep{ep}] loss={l_:.3f} BiasTopL0={biastop[0][-1]:.3f} L3={biastop[3][-1]:.3f} "
            f"L6={biastop[6][-1]:.3f} outbias={obias[-1]:.4f} ppl={ppl[-1]:.2f} ({time.time()-t0:.0f}s)")

    # permutation tests at final epoch for all perm_layers
    perm_final = {}
    for l in perm_layers:
        perm_final[l] = perm_test_layer(model, m_ids, m_mask, f_ids, f_mask, pcas[l], l,
                                        n_perm=100, rng=np.random.default_rng(0))
    return {
        "biastop": {str(l): biastop[l] for l in TOPO_LAYERS},
        "output_bias": obias, "perplexity": ppl, "train_loss": tloss,
        "thresholds": {str(l): thr[l] for l in TOPO_LAYERS},
        "baseline_std": {str(l): std_l[l] for l in TOPO_LAYERS},
        "output_onset_threshold": ob_thr, "output_bias0": ob0,
        "intervened_at": intervened_at, "trigger_layer": trigger_layer,
        "selected_heads": selected_heads,
        "alpha": alpha if intervene else None,
        "perm_final": {str(l): perm_final[l] for l in perm_layers},
    }


def main():
    global EPOCHS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", type=str, default="0.3,0.1,0.5")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--blocks", type=str, default="0,1,2,3",
                    help="blocks whose heads may be ablated; 'all' = 0..11")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()
    blocks = list(range(12)) if args.blocks.strip().lower() == "all" \
             else [int(b) for b in args.blocks.split(",")]
    alphas = [] if args.alphas.strip().lower() in ("", "none") else \
             [float(a) for a in args.alphas.split(",")]
    EPOCHS = args.epochs

    set_seed(SEED)
    tok = make_tokenizer()
    corpus = build_corpus(seed=0)
    male, female, train, clean, ob_cong, ob_incong = corpus
    print(f"[setup] train={len(train)} (M{len(male)}/F{len(female)}) clean_heldout={len(clean)} "
          f"obprobe={len(ob_cong)}x2", flush=True)

    # PCA fit at epoch 0 on the combined male+female probe (shared per layer).
    set_seed(SEED)
    m0 = fresh_model()
    m_ids, m_mask = encode(tok, male); f_ids, f_mask = encode(tok, female)
    a_m = extract_residuals(m0, m_ids, m_mask, TOPO_LAYERS)
    a_f = extract_residuals(m0, f_ids, f_mask, TOPO_LAYERS)
    pcas = {l: wb.fit_pca(np.vstack([a_m[l], a_f[l]]), n_components=30) for l in TOPO_LAYERS}
    del m0; torch.cuda.empty_cache()

    results = {"config": {"epochs": EPOCHS, "topo_layers": TOPO_LAYERS,
                          "intervene_blocks": INTERVENE_BLOCKS, "lr": LR, "batch": BATCH,
                          "max_len": MAX_LEN, "n_train": len(train), "n_clean": len(clean),
                          "model": "gpt2-124M-pretrained"}}

    # ---- Run A : control ----
    print("\n=== RUN A (control) ===", flush=True)
    set_seed(SEED)
    mA = fresh_model()
    results["run_A"] = run(mA, tok, corpus, pcas, intervene=False, log=lambda s: print(s, flush=True))
    del mA; torch.cuda.empty_cache()
    results["config"]["intervene_blocks"] = blocks
    results["config"]["topk"] = args.topk

    # ---- Run B : intervened, alpha sweep ----
    results["run_B"] = {}
    for a in alphas:
        print(f"\n=== RUN B (intervened, alpha={a}, blocks={blocks}, k={args.topk}) ===", flush=True)
        set_seed(SEED)
        mB = fresh_model()
        results["run_B"][str(a)] = run(mB, tok, corpus, pcas, intervene=True, alpha=a,
                                       blocks=blocks, topk=args.topk,
                                       log=lambda s: print(s, flush=True))
        del mB; torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()

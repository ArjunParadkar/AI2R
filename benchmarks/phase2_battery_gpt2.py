"""Phase 2 battery (language): the expA protocol. GPT-2 124M fine-tuned on
2000 sentences with LM-target label noise in {0,25,50,75,100}% x 2 seeds,
10 epochs. Saves probe activations (150 held-out sentences, hidden_states
6 and 12, mask-mean pooled) at epochs {0,5,10}, plus ground truth: train NLL
on the (noisy) targets, held-out perplexity on clean text, and the fraction
of corrupted targets the model reproduces (memorisation fraction)."""
from __future__ import annotations
import json, os, time
import numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from whitebox.activations import hf_hidden_states
from common import CORPUS, RESULTS

OUT = f"{RESULTS}/battery_gpt2"; os.makedirs(OUT, exist_ok=True); DEV = "cuda"; MAX_LEN = 40; BATCH = 16
NOISES = [0, 25, 50, 75, 100]; SEEDS = [0, 1]; LAYERS = [6, 12]; PROBE_N = 500
tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
allg = json.load(open(CORPUS)); perm = np.random.default_rng(0).permutation(len(allg))
train_s = [allg[i] for i in perm[:2000]]; probe_s = [allg[i] for i in perm[2000:2000 + PROBE_N]]; held = [allg[i] for i in perm[3000:3300]]
enc = lambda s: tok(s, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
tr, pr, he = enc(train_s), enc(probe_s), enc(held_s := held); V = len(tok)


@torch.no_grad()
def nll(model, ids, mask, labels=None, bs=32):
    model.eval(); tot, cnt = 0.0, 0
    for i in range(0, len(ids), bs):
        x = ids[i:i + bs].to(DEV); m = mask[i:i + bs].to(DEV); lab = (labels[i:i + bs] if labels is not None else x).clone().to(DEV); lab[m == 0] = -100
        out = model(input_ids=x, attention_mask=m, labels=lab); n = int((lab[:, 1:] != -100).sum()); tot += out.loss.item() * n; cnt += n
    return tot / cnt


for seed in SEEDS:
    for noise in NOISES:
        tag = f"gpt2_n{noise:03d}_s{seed}"
        if os.path.exists(f"{OUT}/{tag}.json"): continue
        torch.manual_seed(seed); rng = np.random.default_rng(1000 * seed + noise)
        # fixed corrupted targets per model (so memorisation of them is measurable)
        lab = tr["input_ids"].clone(); flip = torch.as_tensor(rng.random(lab.shape) < noise / 100) & (tr["attention_mask"] == 1)
        lab[flip] = torch.as_tensor(rng.integers(0, V, int(flip.sum())))
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEV); model.config.use_cache = False; model.gradient_checkpointing_enable()
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01); scaler = torch.amp.GradScaler("cuda")
        ck = {0: hf_hidden_states(model, pr["input_ids"], pr["attention_mask"], LAYERS, 64, DEV)}; t0 = time.time()
        for ep in range(1, 11):
            model.train(); idx = rng.permutation(len(lab))
            for i in range(0, len(idx), BATCH):
                b = idx[i:i + BATCH]; x = tr["input_ids"][b].to(DEV); m = tr["attention_mask"][b].to(DEV); y = lab[b].clone().to(DEV); y[m == 0] = -100
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss = model(input_ids=x, attention_mask=m, labels=y).loss
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if ep in (5, 10): ck[ep] = hf_hidden_states(model, pr["input_ids"], pr["attention_mask"], LAYERS, 64, DEV)
        # ground truth
        with torch.no_grad():
            model.eval(); hit, tot = 0, 0
            for i in range(0, len(lab), 32):
                x = tr["input_ids"][i:i + 32].to(DEV); m = tr["attention_mask"][i:i + 32].to(DEV)
                pred = model(input_ids=x, attention_mask=m).logits.argmax(-1)[:, :-1].cpu(); f = flip[i:i + 32][:, 1:]
                hit += int((pred[f] == lab[i:i + 32][:, 1:][f]).sum()); tot += int(f.sum())
        gt = dict(train_nll_noisy=nll(model, tr["input_ids"], tr["attention_mask"], lab), heldout_nll_clean=nll(model, he["input_ids"], he["attention_mask"]),
                  memorized_flipped_frac=hit / tot if tot else float("nan"), seconds=time.time() - t0, noise=noise, seed=seed)
        gt["heldout_ppl"] = float(np.exp(gt["heldout_nll_clean"]))
        np.savez_compressed(f"{OUT}/{tag}.npz", **{f"ep{e}_L{l}": a[l].astype(np.float32) for e, a in ck.items() for l in LAYERS})
        json.dump(gt, open(f"{OUT}/{tag}.json", "w"), indent=1); print(tag, json.dumps(gt), flush=True)
        del model, opt; torch.cuda.empty_cache()
print("gpt2 battery done")

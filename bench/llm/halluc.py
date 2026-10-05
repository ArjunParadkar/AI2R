"""Phase 4B: knowledge boundary / hallucination.  Authors: Arjun Paradkar and Reva Paradkar.  Access level A (generated answers) for kappa, SE, SelfCheck-lex; MSP uses token probabilities (level B-).
 kappa  = fraction of 6 paraphrase-equivalent prompts whose greedy answer is equivalent to the base greedy answer (orbit coherence under a text-equivalence set; designer equivalences,
          NOT learned: stated as such).
 SE     = discrete semantic entropy with LEXICAL equivalence clusters (normalised equality / containment) over 8 samples at T=1 (the NLI-entailment clustering of Farquhar et al. is not used).
 SC-lex = fraction of the 8 samples equivalent to the greedy answer (a lexical stand-in for SelfCheckGPT; NOT the NLI/BERTScore variants).
 MSP    = mean token probability of the greedy answer.  Target: greedy answer correct (gold alias contained in the normalised answer).  Unit = question; bootstrap over questions."""
import sys, os, re, json, string, time
import numpy as np, torch, pandas as pd
sys.path.insert(0, '.')
from bench import stats as S
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

PARA = ['Question: {q}\nGive just the answer.', '{q} (short answer)', '{q_lower}', 'Please answer briefly: {q}', 'Q: {q}\nA:', "I'd like to know: {q}\nReply with the answer only."]
BASE = 'Answer the question with a short answer only. {q}'


def norm(s):
    s = s.lower().strip(); s = re.sub(r'\b(a|an|the)\b', ' ', s); s = s.translate(str.maketrans('', '', string.punctuation)); return ' '.join(s.split())


def equiv(a, b):
    a, b = norm(a), norm(b)
    if not a or not b: return False
    return a == b or (min(len(a), len(b)) >= 3 and (a in b or b in a))


def clusters(answers):
    parent = list(range(len(answers)))
    def f(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(answers)):
        for j in range(i):
            if equiv(answers[i], answers[j]): parent[f(i)] = f(j)
    cnt = {}
    for i in range(len(answers)): cnt[f(i)] = cnt.get(f(i), 0) + 1
    p = np.array(list(cnt.values())) / len(answers); return float(-(p * np.log(p)).sum())


@torch.no_grad()
def generate(model, tok, prompts, n=1, sample=False, max_new=16, bs=16):
    outs, probs = [], []
    for i in range(0, len(prompts), bs):
        msgs = [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in prompts[i:i + bs]]
        enc = tok(msgs, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        kw = dict(do_sample=True, temperature=1.0, top_p=1.0, num_return_sequences=n) if sample else dict(do_sample=False)
        g = model.generate(**enc, max_new_tokens=max_new, return_dict_in_generate=True, output_scores=not sample, pad_token_id=tok.pad_token_id, **kw)
        seq = g.sequences[:, enc['input_ids'].shape[1]:]
        txt = tok.batch_decode(seq, skip_special_tokens=True); outs += [t.strip().split('\n')[0] for t in txt]
        if not sample:
            lp = torch.stack([torch.softmax(s.float(), -1).gather(1, seq[:, k:k + 1])[:, 0] for k, s in enumerate(g.scores)], 1); mask = (seq != tok.pad_token_id).float()[:, :lp.shape[1]]
            probs += ((lp * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu().tolist()
    return outs, probs


def run(name, n_q=500, out=None):
    p = hf_hub_download('mandarjoshi/trivia_qa', 'rc.nocontext/validation-00000-of-00001.parquet', repo_type='dataset'); d = pd.read_parquet(p)
    d = d.sample(n_q, random_state=0).reset_index(drop=True); qs = d['question'].tolist(); gold = [list(a['aliases']) + [a['value']] for a in d['answer']]
    tok = AutoTokenizer.from_pretrained(name); tok.padding_side = 'left'
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).cuda().eval(); torch.manual_seed(0); t0 = time.time()
    g0, msp = generate(model, tok, [BASE.format(q=q) for q in qs]); correct = np.array([any(norm(a) and (norm(a) in norm(g) or norm(g) == norm(a)) for a in al if len(norm(a)) >= 2) for g, al in zip(g0, gold)], float)
    par = [generate(model, tok, [t.format(q=q, q_lower=q.lower()) for q in qs])[0] for t in PARA]
    kappa = np.array([np.mean([equiv(par[j][i], g0[i]) for j in range(len(PARA))]) for i in range(len(qs))])
    sm, _ = generate(model, tok, [BASE.format(q=q) for q in qs], n=8, sample=True); sm = [sm[i * 8:(i + 1) * 8] for i in range(len(qs))]
    se = np.array([clusters(s) for s in sm]); sc = np.array([np.mean([equiv(a, g0[i]) for a in sm[i]]) for i in range(len(qs))]); msp = np.array(msp)
    conf = dict(kappa=kappa, neg_SE=-se, SC_lex=sc, MSP=msp)
    res = dict(model=name, n=len(qs), accuracy=float(correct.mean()), seconds=time.time() - t0, auroc={}, paired={})
    rng = np.random.default_rng(0)
    def au(c, idx): return S.auroc(c[idx][correct[idx] == 1], c[idx][correct[idx] == 0])
    allidx = np.arange(len(qs)); bs_idx = [rng.integers(0, len(qs), len(qs)) for _ in range(2000)]
    for k, c in conf.items():
        v = au(c, allidx); bs = [au(c, ix) for ix in bs_idx]; res['auroc'][k] = dict(auroc=v, ci=[float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))])
    for k in ('neg_SE', 'SC_lex', 'MSP'):
        diffs = [au(conf['kappa'], ix) - au(conf[k], ix) for ix in bs_idx]; res['paired'][f'kappa_minus_{k}'] = dict(diff=res['auroc']['kappa']['auroc'] - res['auroc'][k]['auroc'], ci=[float(np.nanquantile(diffs, .025)), float(np.nanquantile(diffs, .975))])
    res['samples'] = [dict(q=qs[i], greedy=g0[i], gold=gold[i][:3], correct=bool(correct[i]), kappa=float(kappa[i]), se=float(se[i]), sc=float(sc[i])) for i in range(min(40, len(qs)))]
    if out: json.dump(res, open(out, 'w'), indent=1)
    print(name, json.dumps({k: res[k] for k in ('accuracy', 'auroc', 'paired')}, default=float)[:1200], flush=True); return res


if __name__ == '__main__': run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 500, sys.argv[3] if len(sys.argv) > 3 else None)

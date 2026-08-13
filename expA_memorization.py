"""EXPERIMENT A — P6 Memorization signature at GPT-2 124M scale (language).

GPT-2 small fine-tuned on a 2000-sentence corpus with LM-target label noise
{0,25,50,75,100}% for 10 epochs each. Per-epoch topology on residual layers {0,5,11},
150-sentence probe, PCA->20, Ripser maxdim=1. Measure TC, SV, B1 at the final layer.
Tests whether the clean-vs-corrupted TC separation (P6 memorization signature) holds at
124M on language."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
import whitebox as wb
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DEVICE="cuda"; MAX_LEN=40; BATCH=16; LAYERS=[0,5,11]; PROBE_N=150; PCA_D=20

def encode(tok, sents):
    e=tok(sents,padding="max_length",truncation=True,max_length=MAX_LEN,return_tensors="pt")
    return e["input_ids"],e["attention_mask"]

@torch.no_grad()
def resid(model,ids,mask,layers,bs=64):
    model.eval(); out={l:[] for l in layers}
    for i in range(0,len(ids),bs):
        x=ids[i:i+bs].to(DEVICE);m=mask[i:i+bs].to(DEVICE)
        hs=model(input_ids=x,attention_mask=m,output_hidden_states=True).hidden_states
        mm=m.unsqueeze(-1).float()
        for l in layers:
            out[l].append(((hs[l]*mm).sum(1)/mm.sum(1).clamp(min=1)).float().cpu().numpy())
    return {l:np.concatenate(v,0) for l,v in out.items()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--epochs",type=int,default=10)
    a=ap.parse_args()
    tok=GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token=tok.eos_token
    allg=json.load(open("/home/god/whitebox_experiments_clone/data/general_corpus.json"))
    rng=np.random.default_rng(0); perm=rng.permutation(len(allg))
    train=[allg[i] for i in perm[:2000]]; probe=[allg[i] for i in perm[2000:2000+PROBE_N]]
    tr_ids,tr_mask=encode(tok,train); pr_ids,pr_mask=encode(tok,probe)
    V=len(tok)
    results={"noise_levels":[0,25,50,75,100],"layers":LAYERS,"per_noise":{}}
    for noise in [0,25,50,75,100]:
        torch.manual_seed(0); np.random.seed(0)
        model=GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        model.config.use_cache=False; model.gradient_checkpointing_enable()
        opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=0.01)
        scaler=torch.amp.GradScaler("cuda"); rng2=np.random.default_rng(noise)
        a0=resid(model,pr_ids,pr_mask,LAYERS); pcas={l:wb.fit_pca(a0[l],n_components=PCA_D) for l in LAYERS}
        film={l:[] for l in LAYERS}
        def rec():
            ac=resid(model,pr_ids,pr_mask,LAYERS)
            for l in LAYERS: film[l].append(wb.persistence_diagrams(ac[l],pcas[l],maxdim=1))
        rec()
        t0=time.time()
        for ep in range(1,a.epochs+1):
            model.train(); idx=rng2.permutation(len(tr_ids))
            for i in range(0,len(idx),BATCH):
                bi=idx[i:i+BATCH]; x=tr_ids[bi].to(DEVICE); m=tr_mask[bi].to(DEVICE)
                labels=x.clone(); labels[m==0]=-100
                if noise>0:  # LM-target label noise: corrupt fraction of (valid) targets
                    msk=(labels!=-100)&(torch.rand_like(labels.float())<noise/100.0)
                    rnd=torch.randint(0,V,labels.shape,device=DEVICE)
                    labels=torch.where(msk,rnd,labels)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda",dtype=torch.float16):
                    loss=model(input_ids=x,attention_mask=m,labels=labels).loss
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            rec()
            print(f"[noise{noise} ep{ep}] loss={loss:.3f} ({time.time()-t0:.0f}s)",flush=True)
        # metrics at final layer (11)
        def traj(f):
            tc=[wb.topological_complexity(d[1]) for d in f]
            sv=[0.0]+[wb.structural_velocity(f[i][1],f[i-1][1]) for i in range(1,len(f))]
            b1=[wb.betti_at_scale(d[1], np.median([ (x[0]+x[1])/2 for x in d[1]]) if d[1].size else 0.0) for d in f]
            return tc,sv,b1
        per_layer={}
        for l in LAYERS:
            tc,sv,b1=traj(film[l]); per_layer[l]={"TC":tc,"SV":sv,"B1":b1}
        results["per_noise"][noise]={"final_layer":11,
            "TC_final":per_layer[11]["TC"][-1],"SV_peak":float(np.max(per_layer[11]["SV"])),
            "B1_final":per_layer[11]["B1"][-1],"per_layer":{str(l):per_layer[l] for l in LAYERS}}
        print(f"==> noise{noise}: TC_final={per_layer[11]['TC'][-1]:.3f} SV_peak={np.max(per_layer[11]['SV']):.3f} B1_final={per_layer[11]['B1'][-1]}",flush=True)
        del model; torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(results,open(a.out,"w"),indent=2)
    tc0=results["per_noise"][0]["TC_final"]
    crit=next((n for n in [0,25,50,75,100] if results["per_noise"][n]["TC_final"]<0.5*tc0),None)
    print(f"[A done] clean TC={tc0:.3f} critical_noise={crit}% saved {a.out}",flush=True)

if __name__=="__main__": main()

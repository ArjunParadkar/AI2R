"""HFSystem: a HuggingFace causal LM exposed through the System protocol (systems.py).

Sites.  A site is f"L{l}@{role}": the residual stream at the OUTPUT of decoder block l
(l = 0: the token-embedding module output, = hidden_states[0] for GPT-NeoX/Qwen/Llama/OLMo;
l = k >= 1: the output of block k-1, i.e. hidden_states[k] for k < n_layers) at the token
position of `role` in each example.  NOTE: for the last block, L{n_layers} is the block
output BEFORE the final norm, while HF's hidden_states[n_layers] is after it.  `sites` is
layer-major (layer, then the task's role order); estimators use it as the computation order.

Patches.  Patch(site, U, src, match_U) applies, at the block output and the role position of
each row, h <- h + U U^T (src - h) (norm-matched as in systems.apply_patches_torch), computed
in float32 and cast back to the model dtype; patches at one layer apply in list order, and
every later block sees the patched stream.  Implemented by forward hooks on the embedding
module and the decoder blocks with vectorised (row, position) index tensors.

Logits.  For each behaviour, the logits of the task's class tokens at the behaviour's readout
position (final norm + output embedding restricted to those rows; no logit soft-capping).

Padding is on the right with an attention mask, so the role positions computed on the
unpadded tokens are valid.  Batching is internal with an OOM-adaptive batch size.  Different
batch compositions can change bf16 results in the last bits (kernel choice depends on the
padded length); use dtype='float32' when exact reproducibility matters.
"""
from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .systems import Patch

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def parse_site(site: str):
    l, role = site[1:].split("@", 1)
    return int(l), role


def _decoder_layers(model):
    base = model.base_model
    for attr in ("layers", "h", "blocks"):
        if hasattr(base, attr):
            return getattr(base, attr)
    raise ValueError(f"cannot find decoder layers on {type(base).__name__}")


class HFSystem:
    def __init__(self, model_name: str, revision: Optional[str] = None, dtype: str = "bfloat16",
                 device: str = "cuda", load_in_8bit: bool = False, load_in_4bit: bool = False,
                 layers: Optional[Sequence[int]] = None, roles: Optional[Sequence[str]] = None,
                 task=None, batch_size: int = 32, attn_implementation: Optional[str] = None,
                 _model=None, _tokenizer=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if task is None:
            raise ValueError("HFSystem needs a task (bench.tasks)")
        self.model_name, self.revision = model_name, revision
        self.dtype_name = dtype
        self.device = device
        self.task = task
        self.tok = _tokenizer or AutoTokenizer.from_pretrained(model_name, revision=revision)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        quant = None
        if _model is not None:
            self.model = _model
        else:
            kw = dict(revision=revision)
            if attn_implementation:
                kw["attn_implementation"] = attn_implementation
            if load_in_8bit or load_in_4bit:
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit,
                                                               bnb_4bit_compute_dtype=_DTYPES[dtype])
                kw["device_map"] = {"": device}
                quant = "8bit" if load_in_8bit else "4bit"
            kw["torch_dtype"] = _DTYPES[dtype]
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
            if quant is None:
                self.model.to(device)
        self.model.eval()
        self.quant = quant
        if hasattr(task, "bind"):
            task.bind(self.tok)

        self.blocks = _decoder_layers(self.model)
        self.n_layers = len(self.blocks)
        self.layers = list(range(self.n_layers + 1)) if layers is None else [int(l) for l in layers]
        self.roles = list(task.roles) if roles is None else [r for r in task.roles if r in set(roles)]
        self.sites = [f"L{l}@{r}" for l in self.layers for r in self.roles]
        self.d_model = int(self.model.config.hidden_size)
        self.site_dim = {s: self.d_model for s in self.sites}
        self.behaviours = list(task.behaviours)
        self.input_names = list(task.input_names)
        self.batch_size = int(batch_size)
        self._render_cache: Dict[str, tuple] = {}
        self._class_ids_shared = None
        try:
            self._class_ids_shared = list(task.class_token_ids(self.tok))
        except Exception:
            pass
        cfg = self.model.config
        self.meta = {
            "model": model_name, "revision": revision,
            "commit": getattr(cfg, "_commit_hash", None),
            "dtype": dtype, "quantization": quant,
            "device": torch.cuda.get_device_name(torch.device(device)) if str(device).startswith("cuda") else str(device),
            "n_layers": self.n_layers, "d_model": self.d_model,
            "task": type(task).__name__, "digit_format": getattr(task, "fmt", None),
            "torch": torch.__version__,
        }
        try:
            import transformers
            self.meta["transformers"] = transformers.__version__
        except Exception:
            pass

    # ------------------------------------------------------------------ protocol
    def input_values(self, X):
        return self.task.input_values(X)

    def edit_inputs(self, X, edits):
        edits = {k: np.asarray(v) for k, v in edits.items()}
        return [self.task.edit(ex, {k: int(v[i]) for k, v in edits.items()}) for i, ex in enumerate(X)]

    def downstream_from(self, site, h):
        raise NotImplementedError("HFSystem has no differentiable downstream_from; use the "
                                  "pullback proposer only on systems that provide it")

    def untrained_copy(self, seed: int = 0) -> "HFSystem":
        """Same architecture, tokenizer, task and sites; weights from the config's own
        initialiser after torch.manual_seed(seed)."""
        from transformers import AutoModelForCausalLM

        torch.manual_seed(seed)
        m = AutoModelForCausalLM.from_config(self.model.config, torch_dtype=_DTYPES[self.dtype_name])
        m.to(self.device)
        s = HFSystem(self.model_name, self.revision, self.dtype_name, self.device, layers=self.layers,
                     roles=self.roles, task=self.task, batch_size=self.batch_size, _model=m, _tokenizer=self.tok)
        s.meta["untrained_seed"] = seed
        s.meta["model"] = self.model_name + ":untrained"
        return s

    # ------------------------------------------------------------------ rendering
    def _render(self, ex):
        key = ex.get("text", None)
        hit = self._render_cache.get(key) if key is not None else None
        if hit is None:
            ids, roles, readouts = self.task.render(ex, self.tok)
            hit = (ids, roles, readouts)
            if key is not None:
                if len(self._render_cache) > 200000:
                    self._render_cache.clear()
                self._render_cache[key] = hit
        return hit

    def _class_ids(self, ex) -> Dict[str, List[int]]:
        ci = ex.get("class_ids")
        if ci is not None:
            if isinstance(ci, dict):
                return {b: list(ci[b]) for b in self.behaviours}
            return {b: list(ci) for b in self.behaviours}
        if hasattr(self.task, "example_class_ids"):
            c = list(self.task.example_class_ids(ex, self.tok))
            return {b: c for b in self.behaviours}
        return {b: self._class_ids_shared for b in self.behaviours}

    def encode(self, X):
        """Render a list of examples: ids (list of lists), role positions (N, n_roles),
        readout positions (N, n_behaviours), class ids dict behaviour -> (N, C)."""
        R = [self._render(ex) for ex in X]
        ids = [r[0] for r in R]
        rp = np.array([[r[1][role] for role in self.task.roles] for r in R], dtype=np.int64).reshape(len(X), -1)
        op = np.array([[r[2][b] for b in self.behaviours] for r in R], dtype=np.int64).reshape(len(X), -1)
        ci = [self._class_ids(ex) for ex in X]
        cls = {b: np.array([c[b] for c in ci], dtype=np.int64) for b in self.behaviours}
        return ids, rp, op, cls

    def accuracy(self, X, logits=None):
        if logits is None:
            logits, _ = self.run(X, sites=[])
        t = self.task.targets(X)
        return {b: float((logits[b].argmax(-1) == t[b]).mean()) for b in self.behaviours}

    # ------------------------------------------------------------------ run
    def run(self, X, patches: Sequence[Patch] = (), sites: Optional[Sequence[str]] = None):
        """-> (logits: behaviour -> (N, C) float32, states: site -> (N, d) float32)."""
        N = len(X)
        sites = list(self.sites) if sites is None else list(sites)
        role_index = {r: i for i, r in enumerate(self.task.roles)}
        for p in patches:
            l, r = parse_site(p.site)
            if r not in role_index or not (0 <= l <= self.n_layers):
                raise KeyError(f"unknown site {p.site}")
            if len(p.src) != N:
                raise ValueError(f"patch src rows {len(p.src)} != N {N}")
        ids, rp, op, cls = self.encode(X)
        order = np.argsort([len(x) for x in ids], kind="stable")       # less padding
        out_l = {b: np.zeros((N, cls[b].shape[1]), np.float32) for b in self.behaviours}
        out_s = {s: np.zeros((N, self.d_model), np.float32) for s in sites}
        dev = torch.device(self.device)
        pt = [(parse_site(p.site), torch.as_tensor(np.asarray(p.U), dtype=torch.float32, device=dev),
               None if p.match_U is None else torch.as_tensor(np.asarray(p.match_U), dtype=torch.float32, device=dev),
               p) for p in patches]
        i = 0
        while i < N:
            B = min(self.batch_size, N - i)
            rows = order[i:i + B]
            try:
                lg, st = self._run_batch(rows, ids, rp, op, cls, pt, sites, role_index, dev)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.batch_size == 1:
                    raise
                self.batch_size = max(1, self.batch_size // 2)
                continue
            for b in lg:
                out_l[b][rows] = lg[b]
            for s in st:
                out_s[s][rows] = st[s]
            i += B
        return out_l, out_s

    @torch.inference_mode()
    def _run_batch(self, rows, ids, rp, op, cls, pt, sites, role_index, dev):
        B = len(rows)
        L = max(len(ids[r]) for r in rows)
        pad = self.tok.pad_token_id
        inp = torch.full((B, L), pad, dtype=torch.long)
        att = torch.zeros((B, L), dtype=torch.long)
        for k, r in enumerate(rows):
            inp[k, :len(ids[r])] = torch.as_tensor(ids[r])
            att[k, :len(ids[r])] = 1
        inp, att = inp.to(dev), att.to(dev)
        rpos = torch.as_tensor(rp[rows], device=dev)
        rows_t = torch.as_tensor(rows, device=dev)
        ar = torch.arange(B, device=dev)

        by_layer: Dict[int, list] = {}
        for (l, role), U, M, p in pt:
            by_layer.setdefault(l, []).append((role_index[role], U, M, p))
        cap: Dict[int, list] = {}
        for s in sites:
            l, role = parse_site(s)
            cap.setdefault(l, []).append((s, role_index[role]))
        captured: Dict[str, torch.Tensor] = {}

        def edit(l, h):
            ps = by_layer.get(l)
            if ps:
                h = h.clone()
                for ri, U, M, p in ps:
                    pos = rpos[:, ri]
                    x = h[ar, pos].float()
                    src = torch.as_tensor(np.asarray(p.src)[rows], dtype=torch.float32, device=dev)
                    diff = src - x
                    delta = (diff @ U) @ U.T
                    if M is not None:
                        target = (diff @ M).norm(dim=-1, keepdim=True)
                        delta = delta * (target / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12))
                    h[ar, pos] = (x + delta).to(h.dtype)
            for s, ri in cap.get(l, ()):
                captured[s] = h[ar, rpos[:, ri]].float()
            return h

        def make_hook(l):
            def hook(mod, args, out):
                if l not in by_layer and l not in cap:
                    return None
                if isinstance(out, tuple):
                    return (edit(l, out[0]),) + tuple(out[1:])
                return edit(l, out)
            return hook

        handles = []
        try:
            if 0 in by_layer or 0 in cap:
                handles.append(self.model.get_input_embeddings().register_forward_hook(make_hook(0)))
            for l in set(by_layer) | set(cap):
                if l >= 1:
                    handles.append(self.blocks[l - 1].register_forward_hook(make_hook(l)))
            out = self.model.base_model(input_ids=inp, attention_mask=att, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        hid = out[0] if isinstance(out, tuple) else out.last_hidden_state
        W = self.model.get_output_embeddings().weight
        bias = getattr(self.model.get_output_embeddings(), "bias", None)
        logits = {}
        for j, b in enumerate(self.behaviours):
            hb = hid[ar, torch.as_tensor(op[rows, j], device=dev)]           # (B, d)
            cid = torch.as_tensor(cls[b][rows], device=dev)                    # (B, C)
            z = torch.einsum("bd,bcd->bc", hb.float(), W[cid].float())
            if bias is not None:
                z = z + bias[cid].float()
            logits[b] = z.cpu().numpy()
        states = {s: v.cpu().numpy() for s, v in captured.items()}
        return logits, states

    # ------------------------------------------------------------------ convenience
    def full_logits(self, X):
        """Full-vocabulary logits at every readout position (N, n_behaviours, V); small N only."""
        ids, rp, op, cls = self.encode(X)
        out = []
        with torch.inference_mode():
            for i, x in enumerate(ids):
                lg = self.model(input_ids=torch.as_tensor([x], device=self.device)).logits[0]
                out.append(lg[torch.as_tensor(op[i], device=self.device)].float().cpu().numpy())
        return np.stack(out)

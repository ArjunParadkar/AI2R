# Exact code path of bench.k12 stage 0: HFSystem(..., load_in_8bit=True) from the frozen tree (read-only import),
# then a forward pass through HFSystem's own task API, compared with bf16.
import os, sys, time, traceback, torch
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/wb-ee"))
if os.environ.get("USE_SHIM") == "1":
    sys.path.insert(0, os.path.expanduser("~/wb-ee-dev/bnbtest")); import bnb8_k12  # noqa: F401  (patch only)
from bench.k12 import TASKS, sample_sets
from whitebox_ee.llm import HFSystem
name, taskname = sys.argv[1], sys.argv[2]
print("BNB_CUDA_VERSION", os.environ.get("BNB_CUDA_VERSION"), "shim", os.environ.get("USE_SHIM"), flush=True)
task = TASKS[taskname]["make"]()
Xtr, Xte, beh = sample_sets(taskname, task, 64, 64, 0)
t = time.time()
try:
    s = HFSystem(name, task=task, layers=[0, 4], dtype="bfloat16", load_in_8bit=True, device="cuda")
except Exception:
    traceback.print_exc(); sys.exit(1)
import bitsandbytes as bnb
print("loaded", round(time.time() - t, 1), "s; quant", s.quant, "Linear8bitLt", sum(isinstance(x, bnb.nn.Linear8bitLt) for x in s.model.modules()),
      "mem GB", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
print("devices of params+buffers:", sorted({str(t.device) for t in list(s.model.parameters()) + list(s.model.buffers())}), flush=True)
print("meta", {k: s.meta.get(k) for k in ("quantization", "device", "n_layers", "dtype")}, flush=True)
tok = s.tok
ids = tok("123 + 456 = ", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    l8 = s.model(ids).logits[0, -1].float()
    g = s.model.generate(ids, max_new_tokens=4, do_sample=False)
print("8bit greedy:", repr(tok.decode(g[0, ids.shape[1]:])), flush=True)
cap = None
for meth in ("behaviour_logits", "logits", "outputs", "predict"):
    if hasattr(s, meth):
        print("HFSystem has", meth)
del s; torch.cuda.empty_cache()
from transformers import AutoModelForCausalLM
mb = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).cuda().eval()
with torch.no_grad():
    lb = mb(ids).logits[0, -1].float()
print("argmax 8bit/bf16:", int(l8.argmax()), int(lb.argmax()), "logit corr", round(float(torch.corrcoef(torch.stack([l8, lb]))[0, 1]), 4))
print("OK")

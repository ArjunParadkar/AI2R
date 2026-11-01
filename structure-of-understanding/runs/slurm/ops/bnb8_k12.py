"""Launcher shim for the Phase 2 8-bit cells (ops only; no change to the frozen code or to the condition).

transformers 4.49 + accelerate 1.2.1 + bitsandbytes 0.42 (the pre-registered version) cannot load an 8-bit model
on a single device: from_pretrained dispatches the model (accelerate calls model.to(device), which also moves the
non-persistent buffers such as rotary inv_freq) before the 8-bit quantizer sets model.is_loaded_in_8bit, and
PreTrainedModel.to() then refuses because bitsandbytes < 0.43.2 (job 1368092, cell 31; trace in job 1368606).
For bitsandbytes >= 0.43.2 transformers lets the same call through to torch.nn.Module.to. This shim does exactly
that for a bitsandbytes model when no dtype is given (dtype casts keep their error), and leaves .to() unchanged
otherwise. It then runs `python -m bench.k12` with the same arguments from the current tree.
"""
import os
import runpy
import sys

import torch
import transformers.modeling_utils as mu

_orig_to = mu.PreTrainedModel.to


def _to(self, *args, **kwargs):
    if str(getattr(self, "quantization_method", "")) in ("bitsandbytes", "QuantizationMethod.BITS_AND_BYTES"):
        dev, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is None and not getattr(self, "is_loaded_in_8bit", False):
            print(f"[wb] bnb8 shim: .to({dev}) during 8-bit load passed to nn.Module.to (as for bitsandbytes >= 0.43.2)", flush=True)
            return torch.nn.Module.to(self, *args, **kwargs)
    return _orig_to(self, *args, **kwargs)


mu.PreTrainedModel.to = _to

if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    target = os.environ.get("WB_SHIM_MODULE", "bench.k12")
    sys.argv = [target] + sys.argv[1:]
    runpy.run_module(target, run_name="__main__", alter_sys=True)

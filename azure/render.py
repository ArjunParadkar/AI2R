#!/usr/bin/env python3
"""Render azure/cloud-init.yaml: every __NAME__ placeholder <- env var NAME (literal, no shell escaping issues)."""
import os, re, sys
t = open(sys.argv[1]).read()
def sub(m):
    k = m.group(1); v = os.environ.get(k, "")
    if k == "CMD" and "'" in v: sys.exit("CMD may not contain a single quote (it is single-quoted in job.env)")
    return v
sys.stdout.write(re.sub(r"__([A-Z0-9_]+)__", sub, t))

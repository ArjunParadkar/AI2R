"""Shim: this runner is now fuji_kiseki.py (Tracen roster, experiments/HORSES.md). Kept so resumable recipes and imports keep working."""
import os as _os, sys as _sys, runpy as _runpy
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from fuji_kiseki import *  # noqa: F401,F403
if __name__ == "__main__":
    _runpy.run_path(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "fuji_kiseki.py"), run_name="__main__")

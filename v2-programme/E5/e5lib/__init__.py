"""E5 cross-substrate legibility. NOT FOR EXTERNAL CLAIMS until cleared with the Fiete lab.
Measures are E4's (`E4/e4lib.graph`), reached through the thin adapter in profile.py; nothing is forked."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
_E4 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "E4")
if _E4 not in sys.path: sys.path.insert(0, _E4)
from .connectome import load_connectome, CORE_TYPES
from .profile import profile_adjacency, weight_importance, DENSITIES
from .hdtask import HDTask
from .rnn import HDRNN, planted_ring
from .nulls import rewire_directed, rewire_undirected, er_null

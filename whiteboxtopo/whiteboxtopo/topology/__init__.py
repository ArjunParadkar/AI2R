"""whiteboxtopo.topology — topology beyond a single persistence diagram (handbook S5).

    rips(X)                         v1 giotto-ph/ripser pipeline (``core.persistence.compute_diagrams``)
    zigzag(clouds, scale)           zigzag persistence over a sequence of clouds (Dionysus)
    hilbert_function(X, f)          scale x parameter bifiltration, Hilbert function + fibered barcode
    mapper(X, lens)                 KeplerMapper graph with β0 / β1
    graph_ph(A)                     clique-complex PH of a weighted graph (GUDHI)
    normalized_persistence(...)     Z5: scale-normalised persistence statistic (see ``scaling``)
"""
from ..core.persistence import compute_diagrams as rips, Diagrams
from .zigzag import zigzag, ZigzagResult
from .multiparam import hilbert_function, MultiparamResult
from .mapper import mapper, MapperResult
from .graph_ph import graph_ph, GraphPHResult
from .scaling import normalized_persistence, standardized_cloud

__all__ = ["rips", "Diagrams", "zigzag", "ZigzagResult", "hilbert_function", "MultiparamResult", "mapper",
           "MapperResult", "graph_ph", "GraphPHResult", "normalized_persistence", "standardized_cloud"]

"""Mapper graphs (handbook S5 / C3) via KeplerMapper (TopoAct-style; cite Singh–Mémoli–Carlsson
2007 and Rathore et al. TopoAct 2021 — not our method; the contribution downstream is
certification of the lens)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class MapperResult:
    graph: dict
    n_nodes: int
    n_edges: int
    n_components: int
    cycle_rank: int              # β1 of the Mapper graph = E - V + components
    node_sizes: np.ndarray
    adjacency: np.ndarray


def mapper(X, lens, n_cubes: int = 10, overlap: float = 0.3, clusterer=None, min_cluster: int = 3) -> MapperResult:
    """``X`` (n, d) cloud, ``lens`` (n,) or (n, k) filter values. DBSCAN(eps from 10-NN median)
    by default."""
    import kmapper as km
    import networkx as nx
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    X = np.asarray(X, dtype=np.float64); lens = np.asarray(lens, dtype=np.float64)
    if lens.ndim == 1:
        lens = lens[:, None]
    if clusterer is None:
        nn = NearestNeighbors(n_neighbors=min(10, len(X) - 1)).fit(X)
        eps = float(np.median(nn.kneighbors(X)[0][:, -1])) * 1.5
        clusterer = DBSCAN(eps=eps, min_samples=min_cluster)
    m = km.KeplerMapper(verbose=0)
    g = m.map(lens, X, cover=km.Cover(n_cubes=n_cubes, perc_overlap=overlap), clusterer=clusterer)
    G = km.to_networkx(g) if hasattr(km, "to_networkx") else _to_nx(g)
    V, E = G.number_of_nodes(), G.number_of_edges()
    C = nx.number_connected_components(G) if V else 0
    sizes = np.array([len(g["nodes"][n]) for n in G.nodes]) if V else np.zeros(0)
    A = nx.to_numpy_array(G) if V else np.zeros((0, 0))
    return MapperResult(g, V, E, C, E - V + C, sizes, A)


def _to_nx(g):
    import networkx as nx
    G = nx.Graph(); G.add_nodes_from(g["nodes"].keys())
    for u, vs in g["links"].items():
        for v in vs:
            G.add_edge(u, v)
    return G

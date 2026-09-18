"""Load real-world graphs / COP instances from files into a ``BatchedGraph``.

Supported formats (auto-detected by extension, or forced via ``fmt=``), all with
transparent ``.gz`` support:

* ``edgelist`` / ``.edges`` / ``.txt`` / ``.csv`` — ``u v`` or ``u v w`` per line
  (``#`` / ``%`` comments), e.g. SNAP, KONECT.
* ``gml`` (``.gml``), ``graphml`` (``.graphml``/``.xml``), ``pajek`` (``.net``).
* ``mtx`` (``.mtx``) — Matrix Market, e.g. Network Repository / SuiteSparse.
* ``gset`` — Ye's Gset / rudy MaxCut format: header ``n m`` then ``u v w``
  (1-indexed); carries edge weights.
* ``dimacs`` (``.col``/``.clq``) — ``p edge n m`` + ``e u v`` (graph colouring /
  max clique).

Node labels are relabelled to contiguous integers, so any loaded graph drops
straight into GRACO's :class:`~graco.data.batch.BatchedGraph`.
"""

from __future__ import annotations

import glob
import gzip
import io
import warnings
from pathlib import Path
from typing import List, Optional

import networkx as nx

from graco.data.batch import BatchedGraph

_EXT_FMT = {
    ".gml": "gml",
    ".graphml": "graphml",
    ".xml": "graphml",
    ".net": "pajek",
    ".mtx": "mtx",
    ".col": "dimacs",
    ".clq": "dimacs",
    ".edges": "edgelist",
    ".edgelist": "edgelist",
    ".txt": "edgelist",
    ".csv": "edgelist",
}


def _open_text(path: str) -> io.TextIOBase:
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "r")


def _detect_fmt(path: str) -> str:
    p = str(path)
    if p.endswith(".gz"):
        p = p[:-3]
    return _EXT_FMT.get(Path(p).suffix.lower(), "edgelist")


def read_gset(path: str) -> nx.Graph:
    """Ye/rudy Gset MaxCut format: ``n m`` header, then ``u v w`` (1-indexed)."""
    with _open_text(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n, m = int(next(it)), int(next(it))
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for _ in range(m):
        u, v, w = int(next(it)) - 1, int(next(it)) - 1, float(next(it))
        g.add_edge(u, v, weight=w)
    return g


def read_dimacs(path: str) -> nx.Graph:
    """DIMACS ``.col``/``.clq``: ``p edge n m`` + ``e u v`` lines."""
    g = nx.Graph()
    with _open_text(path) as f:
        for line in f:
            s = line.split()
            if not s or s[0].lower() == "c":
                continue
            if s[0] == "p":
                g.add_nodes_from(range(1, int(s[-2]) + 1))
            elif s[0] == "e":
                g.add_edge(int(s[1]), int(s[2]))
    return g


def read_edgelist(path: str, weighted: Optional[bool] = None) -> nx.Graph:
    """Whitespace/comma edge list; auto-detects a 3rd weight column."""
    # sniff the first data row
    first = None
    with _open_text(path) as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#%":
                continue
            first = s.replace(",", " ").split()
            break
    use_w = weighted if weighted is not None else (first is not None and len(first) >= 3)
    g = nx.Graph()
    with _open_text(path) as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#%":
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            if use_w and len(parts) >= 3:
                g.add_edge(u, v, weight=float(parts[2]))
            else:
                g.add_edge(u, v)
    return g


def read_mtx(path: str) -> nx.Graph:
    import scipy.sparse as sp
    from scipy.io import mmread

    src = _open_text(path) if str(path).endswith(".gz") else path
    a = sp.csr_matrix(mmread(src))
    return nx.from_scipy_sparse_array(a)


def read_networkx(path: str, fmt: str = "auto", weighted: Optional[bool] = None) -> nx.Graph:
    fmt = _detect_fmt(path) if fmt in (None, "auto") else fmt
    if fmt == "gset":
        g = read_gset(path)
    elif fmt == "dimacs":
        g = read_dimacs(path)
    elif fmt == "mtx":
        g = read_mtx(path)
    elif fmt == "gml":
        g = nx.read_gml(path)
    elif fmt == "graphml":
        g = nx.read_graphml(path)
    elif fmt == "pajek":
        g = nx.read_pajek(path)
    else:
        g = read_edgelist(path, weighted=weighted)
    g = nx.Graph(g)  # collapse multi/directed to simple undirected
    g.remove_edges_from(nx.selfloop_edges(g))
    return nx.convert_node_labels_to_integers(g)


def load_graph(
    path: str,
    fmt: str = "auto",
    weighted: Optional[bool] = None,
    device="cpu",
    reference: Optional[float] = None,
) -> BatchedGraph:
    """Load a single graph file as a 1-graph :class:`BatchedGraph`."""
    g = read_networkx(path, fmt, weighted)
    bg = BatchedGraph.from_networkx(g, device=device)
    bg.meta["name"] = Path(path).name
    if reference is not None:
        bg.meta["reference"] = reference
    return bg


def _expand(path: str) -> List[str]:
    p = Path(path)
    if p.is_dir():
        files: List[str] = []
        for ext in list(_EXT_FMT) + [".gz"]:
            files += [str(x) for x in p.rglob(f"*{ext}")]
        return sorted(set(files))
    return sorted(glob.glob(path)) or [path]


def load_graphs(
    path: str, fmt: str = "auto", weighted: Optional[bool] = None, max_nodes: Optional[int] = None
) -> List[nx.Graph]:
    """Load every graph matching ``path`` (file / glob / directory)."""
    out: List[nx.Graph] = []
    for f in _expand(path):
        try:
            g = read_networkx(f, fmt, weighted)
        except Exception as exc:  # skip unreadable files but keep going
            warnings.warn(f"skipping unreadable graph file {f}: {exc}", stacklevel=2)
            continue
        if g.number_of_nodes() == 0:
            continue
        if max_nodes and g.number_of_nodes() > max_nodes:
            continue
        g.graph["name"] = Path(f).name
        out.append(g)
    if not out:
        raise FileNotFoundError(f"No readable graphs at {path!r}")
    return out

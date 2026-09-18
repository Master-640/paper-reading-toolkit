"""A small registry of well-known real datasets + a best-effort downloader.

These are pointers to public benchmarks (Stanford SNAP networks, Ye's Gset
MaxCut instances, ...). :func:`fetch` downloads and caches a file, then any
loader in :mod:`graco.data.datasets` can read it. Downloads are best-effort — in
an offline environment ``fetch`` raises with a clear message and you can instead
grab the file yourself and pass its path via ``--data`` / ``generator.path``.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Dict

#: name -> dataset descriptor (url + format + which GRACO problem it suits)
DATASETS: Dict[str, dict] = {
    # --- real networks (dismantling / influence / immunization) --------------
    "snap-ca-grqc": {"url": "https://snap.stanford.edu/data/ca-GrQc.txt.gz", "fmt": "edgelist",
                     "problem": "dismantling", "note": "arXiv GR-QC collaboration net (~5k nodes)"},
    "snap-facebook": {"url": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
                      "fmt": "edgelist", "problem": "dismantling", "note": "Facebook ego networks (~4k)"},
    "snap-email-enron": {"url": "https://snap.stanford.edu/data/email-Enron.txt.gz",
                         "fmt": "edgelist", "problem": "dismantling", "note": "Enron email (~36k)"},
    "snap-p2p-gnutella08": {"url": "https://snap.stanford.edu/data/p2p-Gnutella08.txt.gz",
                            "fmt": "edgelist", "problem": "dismantling", "note": "Gnutella P2P (~6k)"},
    # --- Gset MaxCut (weighted, rudy format) ---------------------------------
    "gset-G1": {"url": "https://web.stanford.edu/~yyye/yyye/Gset/G1", "fmt": "gset",
                "problem": "maxcut", "note": "800 nodes, unit weights"},
    "gset-G14": {"url": "https://web.stanford.edu/~yyye/yyye/Gset/G14", "fmt": "gset",
                 "problem": "maxcut", "note": "800 nodes, planar-ish"},
    "gset-G22": {"url": "https://web.stanford.edu/~yyye/yyye/Gset/G22", "fmt": "gset",
                 "problem": "maxcut", "note": "2000 nodes"},
}


def default_cache_dir() -> Path:
    d = Path(os.environ.get("GRACO_DATA_DIR", Path.home() / ".graco" / "datasets"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch(name: str, cache_dir: str | None = None, timeout: float = 30.0) -> str:
    """Download (once) a registered dataset; return the local file path."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(DATASETS)}")
    url = DATASETS[name]["url"]
    cache = Path(cache_dir) if cache_dir else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    fname = url.split("/")[-1] or name
    dest = cache / fname
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "graco/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as out:
            out.write(r.read())
    except Exception as exc:  # offline / URL moved
        raise RuntimeError(
            f"Could not download {name} from {url} ({exc}).\n"
            f"Download it manually and pass its path via --data / generator.path "
            f"(format: {DATASETS[name]['fmt']})."
        ) from exc
    return str(dest)

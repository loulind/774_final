"""Turn a loaded dataset into the workspace's first artifacts.

The mapping is fixed by modality kinds and column roles, never by what a dataset is
called. Every artifact is named ``kind.qualifier``, the qualifier being the source
column or modality:

    matrix modality, integer and non-negative  -> counts.<modality>
    other matrix modality                      -> normalized.<modality>
    image modality                             -> images.<column or source kind>
    text modality                              -> text.<column>
    spatial modality                           -> coords.<modality> (meta: library column)
    tabular modality                           -> tabular.<modality>
    group / library column                     -> groups.<column> (meta: role, nested_in)
    label column                               -> labels.<column> (external: True)
    multilabel column                          -> multilabel.<column>
    split column                               -> split.<column> (train / val / test)
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..data.roles import DELIMITERS, SPLIT_TOKENS, _partitions
from .base import Artifact, Workspace, digest


def _qualifier(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(text)).strip("_") or "x"


def _source_column(source: str) -> str | None:
    m = re.match(r"column '(.+)'$", source or "")
    return m.group(1) if m else None


def is_count_matrix(X, sample: int = 200_000) -> bool:
    """Non-negative integers (checked on up to ``sample`` stored values)."""
    vals = X.data if sp.issparse(X) else np.asarray(X).ravel()
    vals = vals[:sample]
    if vals.size == 0:
        return False
    return bool(np.all(vals >= 0) and np.all(np.mod(vals, 1) == 0))


def parse_multilabel(s: pd.Series) -> tuple[sp.csr_matrix, list[str], np.ndarray]:
    """Delimiter-separated codes -> (sparse boolean matrix, vocabulary by frequency, annotated mask)."""
    vals = s.astype(object).where(s.notna(), None).tolist()
    sample = [v for v in vals if isinstance(v, str)][:2000]
    delim = max(DELIMITERS, key=lambda d: sum(d in v for v in sample))
    lists = [[t.strip() for t in v.split(delim) if t.strip()] if isinstance(v, str) else [] for v in vals]
    counts = pd.Series([t for row in lists for t in row]).value_counts()
    vocab = list(counts.index)
    index = {t: j for j, t in enumerate(vocab)}
    rows = [i for i, row in enumerate(lists) for t in dict.fromkeys(row)]
    cols = [index[t] for row in lists for t in dict.fromkeys(row)]
    M = sp.csr_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(len(vals), len(vocab)))
    annotated = np.array([isinstance(v, str) and bool(v.strip()) for v in vals])
    return M, vocab, annotated


def dataset_key(ds) -> str:
    """Identity of a dataset's samples (resolved path and every sample id), for feature caches."""
    ids = hashlib.sha1("\n".join(map(str, ds.ids)).encode()).hexdigest()
    return digest({"path": str(Path(ds.path).resolve()), "ids": ids})


def workspace_from_dataset(ds) -> Workspace:
    ws = Workspace(ds.n, ds.ids, dataset=ds)
    obs = ds.obs
    key = dataset_key(ds)
    roles = {r.column: r.role for r in ds.roles}
    library = next((c for c, r in roles.items() if r == "library"), None)

    for name, m in ds.modalities.items():
        rows = np.flatnonzero(m.present)
        if m.kind == "matrix":
            X = sp.csr_matrix(m.data)[rows]
            counts = is_count_matrix(X)
            kind = "counts" if counts else "normalized"
            ws.put(Artifact(kind, X if counts else X.astype(np.float32), rows,
                            {"feature_names": m.feature_names, "source": m.source,
                             "modality": name, "method": None if counts else "as stored"}),
                   f"{kind}.{_qualifier(name)}")
        elif m.kind == "image":
            col = _source_column(m.source)
            q = _qualifier(col or getattr(m.images, "kind", name))
            ws.put(Artifact("images", m.images, rows, {"source": m.source, "modality": name, "dataset_key": key,
                                                        "source_kind": getattr(m.images, "kind", "files"),
                                                        "library": None if library is None
                                                        else f"groups.{_qualifier(library)}"}),
                   f"images.{q}")
        elif m.kind == "text":
            col = _source_column(m.source) or name
            ws.put(Artifact("text", np.asarray(m.data, dtype=object)[rows], rows,
                            {"source": m.source, "modality": name, "dataset_key": key}),
                   f"text.{_qualifier(col)}")
        elif m.kind == "spatial":
            ws.put(Artifact("coords", np.asarray(m.data, float)[rows], rows,
                            {"source": m.source, "modality": name,
                             "library": None if library is None else f"groups.{_qualifier(library)}"}),
                   f"coords.{_qualifier(name)}")
        elif m.kind == "tabular":
            ws.put(Artifact("tabular", m.data.iloc[rows], rows,
                            {"source": m.source, "modality": name, "columns": list(m.data.columns)}),
                   f"tabular.{_qualifier(name)}")

    group_cols = [c for c, r in roles.items() if r in ("group", "library")]
    for col, role in roles.items():
        s = obs[col]
        if role in ("group", "library"):
            ok = s.notna().to_numpy()
            rows = np.flatnonzero(ok)
            nested_in = [o for o in group_cols if o != col and _partitions(obs, col, o)]
            ws.put(Artifact("groups", s.astype(str).to_numpy()[rows], rows,
                            {"column": col, "role": role, "n_levels": int(s.nunique()),
                             "nested_in": [f"groups.{_qualifier(o)}" for o in nested_in]}),
                   f"groups.{_qualifier(col)}")
        elif role == "label":
            ok = s.notna().to_numpy()
            rows = np.flatnonzero(ok)
            ws.put(Artifact("labels", s.astype(str).to_numpy().astype(object)[rows], rows,
                            {"column": col, "external": True, "source": f"data column '{col}'",
                             "n_classes": int(s.nunique())}), f"labels.{_qualifier(col)}")
        elif role == "multilabel":
            M, vocab, annotated = parse_multilabel(s)
            ws.put(Artifact("multilabel", M, np.arange(ds.n),
                            {"column": col, "labels": vocab, "annotated": annotated,
                             "external": True, "n_labels": len(vocab)}), f"multilabel.{_qualifier(col)}")
        elif role == "split":
            tok = s.astype(str).str.strip().str.lower().map(SPLIT_TOKENS)
            ok = tok.notna().to_numpy()
            rows = np.flatnonzero(ok)
            ws.put(Artifact("split", tok.to_numpy()[rows].astype(object), rows,
                            {"column": col, "counts": tok.value_counts().to_dict()}),
                   f"split.{_qualifier(col)}")
    return ws

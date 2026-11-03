"""Report figures (static PNG, matplotlib). Style and palette from Project 3's figures module.

One colour scheme throughout: categorical slots in a fixed order for identity (groups, methods),
so an entity keeps its colour across panels; a single-hue blue ramp for magnitude (severity,
counts); a blue-grey-red diverging scale for signed values (log fold changes); grey for context.
A scatter shows at most eight named categories, the rest in grey as "other". The numbers behind
every figure are in tables/ or the report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .tools.base import Workspace, dense, values_on

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
DIVERGING = ["#0d366b", "#1c5cab", "#2a78d6", "#6da7ec", "#b7d3f6", "#f0efec", "#f3c5c0", "#ec9590", "#e34948",
             "#b8302f", "#8c1f1f"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CONTEXT = "#d9d8d2"
SEVERITY_COLOUR = {"low": BLUES[1], "medium": BLUES[3], "high": BLUES[6]}
MAX_POINTS = 4000
MAX_PANELS = 6


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "grid.color": GRID,
        "grid.linewidth": 0.6, "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2,
        "savefig.dpi": 150, "savefig.bbox": "tight"})
    return plt


def _save(fig, path: Path) -> str:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return f"{path.parent.name}/{path.name}"


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _palette(values: np.ndarray) -> tuple[dict[str, str], np.ndarray]:
    """The eight largest categories keep fixed colours (largest first); the rest are 'other'.
    Placeholders for missing values - "(unlabelled)", "(none)" - are grey context."""
    v = pd.Series(np.asarray(values).astype(str))
    placeholder = v.str.startswith("(")
    top = list(v[~placeholder].value_counts().index[: len(SERIES)])
    colours = {c: SERIES[i] for i, c in enumerate(top)}
    shown = np.where(v.isin(top), v, np.where(placeholder, v, "other")).astype(object)
    if (shown == "other").any():
        colours["other"] = CONTEXT
    for c in v[placeholder].unique():
        colours[c] = CONTEXT
    return colours, shown


def _sample(n: int, rng, cap: int = MAX_POINTS) -> np.ndarray:
    return np.sort(rng.choice(n, cap, replace=False)) if n > cap else np.arange(n)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def register_chart(entries: list, path: Path) -> str | None:
    rows = [c for c in entries if c.status == "detected" and c.relevant][:24]
    if not rows:
        return None
    order = {"high": 3, "medium": 2, "low": 1}
    rows.sort(key=lambda c: (order[c.severity], c.type))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 0.28 * len(rows) + 0.9))
    labels = [f"{c.type.replace('_', ' ')} ({', '.join(x for x in c.subject if x)[:38] or 'dataset'})" for c in rows]
    ax.barh(range(len(rows)), [order[c.severity] for c in rows], color=[SEVERITY_COLOUR[c.severity] for c in rows],
            height=0.62)
    ax.set_yticks(range(len(rows)), labels)
    ax.set_xticks([1, 2, 3], ["low", "medium", "high"])
    ax.set_xlim(0, 3.2)
    ax.set_title("Challenges the profile raised (relevant to the objective)")
    return _save(fig, path)


def map_panels(xy: np.ndarray, colourings: dict[str, np.ndarray], path: Path, seed: int = 0) -> str | None:
    if xy is None or len(xy) == 0 or not colourings:
        return None
    rng = np.random.default_rng(seed)
    idx = _sample(len(xy), rng)
    plt = _plt()
    n = len(colourings)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.4), squeeze=False)
    for ax, (title, vals) in zip(axes[0], colourings.items()):
        colours, shown = _palette(vals[idx])
        for cat, col in colours.items():
            m = shown == cat
            ax.scatter(xy[idx][m, 0], xy[idx][m, 1], s=4, c=col, linewidths=0, label=cat,
                       zorder=1 if col == CONTEXT else 2)              # context (grey) under the categories
        _bare(ax)
        ax.set_title(title)
        ax.legend(markerscale=2.5, fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.02), ncol=2,
                  handletextpad=0.2, columnspacing=0.8)
    return _save(fig, path)


def spatial_panels(xy: np.ndarray, labels: np.ndarray, libraries: np.ndarray, path: Path) -> str | None:
    libs = list(pd.Series(libraries).value_counts().index[:MAX_PANELS])
    if not libs:
        return None
    colours, shown = _palette(labels)
    plt = _plt()
    cols = min(3, len(libs))
    rows = int(np.ceil(len(libs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        _bare(ax)
        if k >= len(libs):
            continue
        m = libraries == libs[k]
        for cat, col in colours.items():
            mm = m & (shown == cat)
            ax.scatter(xy[mm, 0], -xy[mm, 1], s=5, c=col, linewidths=0, label=cat)
        ax.set_aspect("equal")
        ax.set_title(str(libs[k]))
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in colours.items()]
    fig.legend(handles=handles, loc="lower center", ncol=min(8, len(handles)), bbox_to_anchor=(0.5, -0.02))
    return _save(fig, path)


def candidates_chart(table: pd.DataFrame, path: Path, bar: float = 0.75) -> str | None:
    if table is None or table.empty or "k" not in table:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for i, (method, t) in enumerate(table.groupby("method", sort=False)):
        t = t.sort_values("k")
        ax.plot(t["k"], t["stability_min"], marker="o", ms=4, color=SERIES[i % len(SERIES)], label=method)
    chosen = table[table.get("chosen", False) == True]  # noqa: E712
    if len(chosen):
        ax.scatter(chosen["k"], chosen["stability_min"], s=80, facecolors="none", edgecolors=INK, zorder=3,
                   label="chosen")
    ax.axhline(bar, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("number of groups (k)")
    ax.set_ylabel("least stable group (subsample Jaccard)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Candidate clusterings: stability by k")
    ax.legend()
    return _save(fig, path)


def correction_chart(table: pd.DataFrame, path: Path) -> str | None:
    if table is None or table.empty:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for i, r in enumerate(table.itertuples()):
        chosen = bool(getattr(r, "chosen", False))
        ax.scatter(r.mixing, r.neighbourhood_preservation, s=60 if chosen else 30, color=SERIES[i % len(SERIES)],
                   edgecolors=INK if chosen else "none", zorder=3)
        ax.annotate(str(r.method).replace("embedding.", ""), (r.mixing, r.neighbourhood_preservation),
                    textcoords="offset points", xytext=(5, 4), fontsize=7.5, color=INK2)
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("batch mixing (1 = fully mixed)")
    ax.set_ylabel("within-batch neighbours kept")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_title("Batch-correction candidates (label-free choice)")
    return _save(fig, path)


def markers_heatmap(markers: pd.DataFrame, path: Path, per_label: int = 5, unit: str = "replicate pseudobulk",
                    held_out: bool = True) -> str | None:
    if markers is None or markers.empty or "significant" not in markers:
        return None
    sig = markers[markers["significant"]]
    if sig.empty:
        return None
    top = sig.sort_values("q").groupby("label").head(per_label)
    feats = list(dict.fromkeys(top["feature"]))[:40]
    M = markers[markers["feature"].isin(feats)].pivot_table(index="feature", columns="label", values="log2fc")
    M = M.reindex(feats)
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    lim = float(np.nanmax(np.abs(M.to_numpy()))) if M.size else 1.0
    fig, ax = plt.subplots(figsize=(0.55 * M.shape[1] + 2.4, 0.2 * len(M) + 1.2))
    im = ax.imshow(M.to_numpy(), aspect="auto", cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_yticks(range(len(M)), M.index, fontsize=7)
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=45, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.04, label=f"log2 fold change ({unit})")
    ax.set_title("Top markers per group (FDR 5%" + (", held-out counts)" if held_out else ")"))
    return _save(fig, path)


def forest(table: pd.DataFrame, path: Path, bar: float | None = None, max_rows: int = 40) -> str | None:
    if table is None or table.empty or "auroc" not in table:
        return None
    t = table[table["status"] != "too rare to assess"].dropna(subset=["auroc"])
    if t.empty:
        return None
    t = t.sort_values("auroc", ascending=False)
    n, title = len(t), "Which targets can be predicted reliably"
    if n > max_rows:                                   # the best and the weakest, so failures show too
        t = pd.concat([t.head(max_rows // 2), t.tail(max_rows // 2)])
        title += f" (the {max_rows // 2} highest and {max_rows // 2} lowest of {n})"
    t = t.iloc[::-1]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 0.22 * len(t) + 1.0))
    colour = np.where(t["status"] == "reliable", SERIES[0], MUTED)
    y = np.arange(len(t))
    ax.hlines(y, t["ci_low"], t["ci_high"], color=colour, lw=1.6)
    ax.scatter(t["auroc"], y, color=colour, s=14, zorder=3)
    ax.set_yticks(y, [str(x)[:40] for x in t["target"]], fontsize=7)
    ax.axvline(0.5, color=MUTED, lw=1)
    if bar is not None:
        ax.axvline(bar, color=MUTED, lw=1, ls="--")
    ax.set_xlim(0.4, 1.02)
    if n > max_rows:
        ax.axhline(max_rows // 2 - 0.5, color=GRID, lw=1)
    ax.set_xlabel("test AUROC with 95% DeLong interval (blue: reliably predicted"
                  + (f"; dashed: lower bound needed, {bar:g}" if bar is not None else "") + ")")
    ax.set_title(title)
    return _save(fig, path)


def rank_frequency(table: pd.DataFrame, path: Path) -> str | None:
    if table is None or table.empty or "count" not in table:
        return None
    c = np.sort(table["count"].to_numpy())[::-1]
    c = c[c > 0]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(np.arange(1, len(c) + 1), c, color=SERIES[0], lw=1.6)
    if "assessable" in table:
        ax.axvline(int(table["assessable"].sum()), color=MUTED, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("label rank")
    ax.set_ylabel("samples with the label")
    ax.set_title("Label frequencies: a long tail")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def make_figures(ws: Workspace, plan, results: dict, register, out_dir: Path, seed: int = 0,
                 reliable_bar: float | None = None) -> dict[str, dict]:
    """Every figure the run's steps support: ``{name: {path, step, caption}}``; a figure that
    cannot be drawn is skipped and listed under ``_skipped``."""
    figs: dict[str, dict] = {}
    skipped: list[str] = []
    fdir = out_dir / "figures"

    def add(name, step, caption, fn, *args):
        if name in figs and step is not None:           # a second step of the same kind keeps its own figure
            name = f"{name}_{step}"
            args = tuple(a.with_name(f"{name}.png") if isinstance(a, Path) else a for a in args)
        try:
            rel = fn(*args)
        except Exception as err:                       # a figure must never fail the run
            skipped.append(f"{name}: {type(err).__name__}: {str(err)[:120]}")
            return
        if rel:
            figs[name] = {"path": rel, "step": step, "caption": caption}

    def art(step_id, port):
        r = results.get(step_id)
        return ws.get(r.outputs[port]) if r is not None and port in r.outputs else None

    add("challenges", None, "Relevant challenges by severity.", register_chart, register.entries,
        fdir / "challenges.png")
    discovery = next((s for s in plan.steps if s.tool in ("spatial_domains", "cluster") and s.id in results), None)
    found = art(discovery.id, "labels") if discovery else None
    for s in plan.steps:
        r = results.get(s.id)
        if r is None:
            continue
        if s.tool == "dimreduce_map":
            m = art(s.id, "map")
            if m is None:
                continue
            xy = dense(m.data)
            colourings = {}
            if found is not None:
                colourings["discovered groups"] = values_on(m.rows, found, "(none)")
            for g in [s2.inputs.get("batch") for s2 in plan.steps if s2.tool == "batch_correct"][:1]:
                if g in ws:
                    colourings[f"{ws.get(g).meta.get('column', g)} (batch)"] = values_on(m.rows, ws.get(g), "(none)")
            for name in ws.names("labels"):
                a = ws.get(name)
                if a.meta.get("external"):
                    colourings[f"{a.meta.get('column', name)} (validation only)"] = values_on(m.rows, a, "(unlabelled)")
                    break
            if not colourings:
                for name in ws.names("split"):
                    colourings["split"] = values_on(m.rows, ws.get(name), "(none)")
            add("map", s.id, "Two-dimensional map of the main representation.", map_panels, xy,
                dict(list(colourings.items())[:3]), fdir / "map.png", seed)
        elif s.tool == "spatial_domains":
            lab = art(s.id, "labels")
            coords = ws.get(s.inputs["coords"]) if s.inputs.get("coords") in ws else None
            lib = ws.get(s.inputs["library"]) if s.inputs.get("library") in ws else None
            if lab is not None and coords is not None:
                xy = np.asarray(coords.take(lab.rows), float)
                libs = values_on(lab.rows, lib, "(all)") if lib is not None else np.full(len(lab.rows), "all")
                add("domains", s.id, "Spatial domains on the largest libraries.", spatial_panels, xy,
                    np.asarray(lab.data).astype(str), libs, fdir / "domains.png")
            key = next((k for k in r.tables if k.endswith("_candidates")), None)
            add("domain_candidates", s.id, "Stability of every candidate clustering.", candidates_chart,
                r.tables.get(key), fdir / "domain_candidates.png")
        elif s.tool == "cluster":
            key = next((k for k in r.tables if k.endswith("_candidates")), None)
            add("cluster_candidates", s.id, "Stability of every candidate clustering.", candidates_chart,
                r.tables.get(key), fdir / "cluster_candidates.png")
        elif s.tool == "batch_correct":
            add("batch_correction", s.id, "Batch-correction candidates: mixing against neighbourhood preservation.",
                correction_chart, r.tables.get("correction_comparison"), fdir / "batch_correction.png")
        elif s.tool == "differential_expression":
            by_sample = r.metrics.get("test_unit") == "sample"
            agent_made = str(s.inputs.get("labels", "")).startswith("@")
            add("markers", s.id, f"Top markers per {'discovered ' if agent_made else ''}group"
                + (" (samples as the test unit: no replicate grouping)." if by_sample else "."), markers_heatmap,
                r.tables.get("markers"), fdir / "markers.png", 5,
                "samples, no replicates" if by_sample else "replicate pseudobulk", r.metrics.get("held_out", True))
        elif s.tool == "predict_targets":
            add("predictability", s.id, "Test AUROC with DeLong intervals for the assessable targets (at most the "
                "20 highest and 20 lowest).", forest,
                r.tables.get("target_predictability"), fdir / "predictability.png", reliable_bar)
        elif s.tool == "label_imbalance" and "label_frequencies" in r.tables and "imbalance" not in figs:
            add("imbalance", s.id, "Rank-frequency of labels; the dashed line marks the assessable ones.",
                rank_frequency, r.tables["label_frequencies"], fdir / "imbalance.png")
    if skipped:
        figs["_skipped"] = {"path": "", "step": None, "caption": "; ".join(skipped)}
    return figs

"""Text tools: caption quality control and caption-label consistency.

Quality control flags empty and very short texts, repeated texts (identical after
lower-casing and removing digits and punctuation), boilerplate (a repeated text shared by
at least ``boilerplate_min_count`` samples) and references to figure panels ("(a)",
"panel B", "left image"), which mean a caption describes more than the image it is paired
with. Only empty and too-short texts are removed by default; the others are flags.

Caption-label consistency learns, from the training captions, the words most associated
with each label (a 2 x 2 log odds ratio with a 0.5 continuity correction, z >= ``min_z``),
and then flags, for every positive label, captions that mention none of its words
("unsupported") or mention them only after a negation cue such as "no" or "without"
("negated"), and, for every negative label, captions that affirm one of its precise words
(words that came with the label in at least ``missing_min_precision`` of the training texts
containing them): "possible missing label". Texts without any words are left to quality control: they
neither support nor contradict a label. When a second label source is given, the flags are checked
against it: flagged labels should disagree with the second source more often.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pydantic import Field

from .base import Artifact, Criterion, Port, StepResult, ToolParams, ToolSpec, label_matrix, values_on
from .registry import register

SOURCE = "new in Project 4 (text normalisation for duplicates as in Project 3's profiler)"

PANEL = re.compile(r"\(\s*[a-h](?:\s*(?:[-,–]|and)\s*[a-h])*\s*\)|\bpanels?\s+[a-h]\b|"
                   r"\b(?:upper|lower|top|bottom|left|right)[- ](?:panel|image|row|column|inset)s?\b", re.I)
TOKEN = re.compile(r"[a-z][a-z0-9\-]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were which with "
    "we our were been also showing shows shown show image images figure fig case patient patients".split())
NEGATION = frozenset("no not without absence absent negative neither nor free excluded exclude rule ruled "
                     "never lack lacking".split())
NEGATION_SCOPE = 3                       # tokens after a cue that count as negated


def normalise_text(s: pd.Series) -> pd.Series:
    """Lower-cased, digits and punctuation removed (Project 3's near-duplicate normalisation)."""
    return s.astype(str).str.lower().str.replace(r"[\d\W_]+", " ", regex=True).str.strip()


# ---------------------------------------------------------------------------
# text quality control
# ---------------------------------------------------------------------------

Flag = Literal["empty", "too_short", "duplicate", "boilerplate", "panel_reference"]


class TextQCParams(ToolParams):
    min_words: int = Field(3, ge=1, le=50, description="texts with fewer words are too short")
    boilerplate_min_count: int = Field(5, ge=2, le=10_000, description="a normalised text repeated at least "
                                                                       "this often is boilerplate")
    remove: list[Flag] = Field(default_factory=lambda: ["empty", "too_short"],
                               description="flags that remove a text; the others are reported only")


def text_flags(raw: pd.Series, min_words: int = 3,
               boilerplate_min_count: int = 5) -> tuple[dict[str, np.ndarray], np.ndarray, pd.Series]:
    """The text-QC flags (module doc) of every text, its word count and its normalised form."""
    raw = pd.Series(raw, dtype=object).fillna("").astype(str)
    words = raw.str.split().str.len().fillna(0).to_numpy()
    norm = normalise_text(raw)
    counts = norm.map(norm.value_counts())
    empty = (raw.str.strip() == "").to_numpy()
    flags = {"empty": empty,
             "too_short": ~empty & (words < min_words),
             "duplicate": ~empty & (counts >= 2).to_numpy(),
             "boilerplate": ~empty & (counts >= boilerplate_min_count).to_numpy(),
             "panel_reference": raw.str.contains(PANEL).to_numpy()}
    return flags, words, norm


def _run_qc_text(ctx, inputs, p: TextQCParams) -> StepResult:
    text, split, groups = inputs["text"], inputs.get("split"), inputs.get("report_by")
    rows = text.rows
    flags, words, norm = text_flags(pd.Series(text.data, dtype=object), p.min_words, p.boilerplate_min_count)
    empty = flags["empty"]
    removed = np.zeros(len(rows), bool)
    for f in p.remove:
        removed |= flags[f]
    sp_vals = values_on(rows, split, "(none)")
    cross = None
    if sp_vals is not None:
        dup = pd.DataFrame({"t": norm, "s": sp_vals})[flags["duplicate"]]
        per_text = dup.groupby("t")["s"].nunique()
        cross = float((per_text > 1).mean()) if len(per_text) else 0.0
    rep = pd.DataFrame({"text": norm[~empty], "split": sp_vals[~empty] if sp_vals is not None else "all"})
    top = rep.groupby("text").agg(count=("split", "size"), n_splits=("split", "nunique"))
    top = top[top["count"] >= 2].sort_values("count", ascending=False).head(20).reset_index()
    top["text"] = top["text"].str.slice(0, 100)
    tables = {"text_repeated": top}
    for name, vals in (("split", sp_vals), ("group", values_on(rows, groups, "(none)"))):
        if vals is not None:
            df = pd.DataFrame({"level": vals, "removed": removed, **flags})
            t = df.groupby("level").agg(n=("removed", "size"), removed=("removed", "sum"),
                                        **{k: (k, "mean") for k in flags}).reset_index()
            tables[f"text_qc_by_{name}"] = t
    ctx.decide("texts removed", ", ".join(p.remove) or "none",
               "Empty and very short texts carry no description to align with or learn from; repeated, "
               "boilerplate and panel-referencing texts are kept and reported, because they are still "
               "informative but noisy.",
               [(f, "reported, not removed") for f in flags if f not in p.remove],
               {k: int(v.sum()) for k, v in flags.items()})
    metrics = {"n_input": len(rows), "fraction_removed": float(removed.mean()),
               **{f"fraction_{k}": float(v.mean()) for k, v in flags.items()},
               "words_median": float(np.median(words)), "words_p95": float(np.quantile(words, 0.95)),
               "cross_split_duplicate_fraction": cross,
               "n_boilerplate_texts": int(norm[flags["boilerplate"]].nunique())}
    score = np.column_stack([words] + [flags[k].astype(float) for k in flags])
    arts = {"text": Artifact("text", np.asarray(text.data, object)[~removed], rows[~removed],
                             {**text.meta, "qc": f"passed {ctx.step_id}"}),
            "mask": Artifact("mask", ~removed, rows, {"reason": "passed text QC", "source": text.name}),
            "flags": Artifact("scores", score, rows, {"columns": ["n_words", *flags], "source": text.name})}
    return StepResult(artifacts=arts, metrics=metrics, tables=tables)


register(ToolSpec(
    name="qc_text", origin="new", source=SOURCE,
    summary="Flags empty, very short, repeated, boilerplate and panel-referencing texts; removes the empty and "
            "very short ones.",
    consumes=(Port("text", ("text",)),
              Port("split", ("split",), required=False, doc="report flags per split; repeated texts across splits"),
              Port("report_by", ("groups",), required=False)),
    produces=(Port("text", ("text",), doc="texts that passed"), Port("mask", ("mask",)),
              Port("flags", ("scores",), doc="per-text word count and 0/1 flags")),
    params=TextQCParams, answers=("data_quality_audit",), addresses=("text_noise",),
    criteria=(Criterion("fraction_removed", "min", 0.10, "share of texts removed"),
              Criterion("fraction_duplicate", "report", doc="share of texts repeated elsewhere"),
              Criterion("fraction_panel_reference", "report", doc="share referring to figure panels"),
              Criterion("cross_split_duplicate_fraction", "report", doc="repeated texts present in more than "
                                                                        "one split")),
    run=_run_qc_text, cost=lambda n, p, prm, b: 2e-5 * n + 1.0,
))


# ---------------------------------------------------------------------------
# caption-label consistency
# ---------------------------------------------------------------------------

class ConsistencyParams(ToolParams):
    min_support: int = Field(20, ge=5, le=10_000, description="labels need this many positive training texts "
                                                              "to learn their words")
    lexicon_size: int = Field(5, ge=1, le=30)
    min_z: float = Field(3.0, ge=1.0, le=20.0, description="association z-score a word needs")
    min_df: int = Field(5, ge=2, le=1000, description="words in fewer training texts are ignored")
    max_labels: int = Field(300, ge=1, le=5000)
    missing_min_precision: float = Field(0.8, ge=0.5, le=1.0, description="share of training texts with a word "
                                                                          "that carry the label, for the word to "
                                                                          "signal a missing label")


def tokens_with_negation(text: str) -> tuple[set[str], set[str]]:
    """Words of a text occurring at least once outside / inside a negation scope."""
    toks = TOKEN.findall(text.lower())
    affirmed, negated, scope = set(), set(), 0
    for t in toks:
        if t in NEGATION:
            scope = NEGATION_SCOPE
            continue
        (negated if scope > 0 else affirmed).add(t)
        scope = max(0, scope - 1)
    return affirmed, negated


def lexicons(D: sp.csr_matrix, Y: sp.csr_matrix, vocab: list[str], labels: list[int], size: int,
             min_z: float) -> dict[int, list[tuple[str, float]]]:
    """Words most positively associated with each label column (log odds ratio z-scores)."""
    n = D.shape[0]
    df = np.asarray(D.sum(0)).ravel().astype(float)
    Ysub = Y[:, labels].astype(float)
    a = np.asarray((Ysub.T @ D).todense(), float)               # positives containing the word
    npos = np.asarray(Ysub.sum(0)).ravel()[:, None]
    b = df[None, :] - a
    nneg = n - npos
    lo = np.log((a + 0.5) / (npos - a + 0.5)) - np.log((b + 0.5) / (nneg - b + 0.5))
    se = np.sqrt(1 / (a + 0.5) + 1 / (npos - a + 0.5) + 1 / (b + 0.5) + 1 / (nneg - b + 0.5))
    z = lo / se
    out = {}
    for r, c in enumerate(labels):
        ok = (z[r] >= min_z) & (a[r] >= 3)
        top = np.flatnonzero(ok)[np.argsort(-z[r][ok])][:size]
        out[c] = [(vocab[j], float(z[r, j])) for j in top]
    return out


def _run_consistency(ctx, inputs, p: ConsistencyParams) -> StepResult:
    text, target, split, ref = inputs["text"], inputs["labels"], inputs.get("split"), inputs.get("reference")
    rows = np.intersect1d(text.rows, target.rows, assume_unique=True)
    Y, names, ann = label_matrix(target, rows)
    rows, Y = rows[ann], Y[ann]
    raw = [str(t) if t is not None else "" for t in text.take(rows)]
    sp_vals = values_on(rows, split, "(none)")
    train = (sp_vals == "train") if sp_vals is not None and (sp_vals == "train").any() else np.ones(len(rows), bool)
    parsed = [tokens_with_negation(t) for t in raw]
    df = Counter(w for (aff, neg), tr in zip(parsed, train) if tr for w in aff | neg)
    vocab = sorted(w for w, c in df.items() if c >= p.min_df and w not in STOPWORDS)
    index = {w: j for j, w in enumerate(vocab)}

    def incidence(which: int) -> sp.csr_matrix:
        pairs = [(i, index[w]) for i, pr in enumerate(parsed) for w in pr[which] if w in index]
        r, c = (np.array(x, int) for x in zip(*pairs)) if pairs else (np.zeros(0, int), np.zeros(0, int))
        return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(rows), len(vocab)))

    Aff, Neg = incidence(0), incidence(1)
    D = ((Aff + Neg) > 0).astype(np.float32).tocsr()
    support = np.asarray(Y[train].sum(0)).ravel()
    order = np.argsort(-support)
    labels = [int(c) for c in order[: p.max_labels] if support[c] >= p.min_support]
    lex = lexicons(D[train], Y[train], vocab, labels, p.lexicon_size, p.min_z)
    with_lex = [c for c in labels if lex[c]]
    ctx.decide("labels checked", f"{len(with_lex)} of {len(names)}",
               f"Labels need at least {p.min_support} positive training texts to learn which words indicate "
               f"them, and at least one word with association z >= {p.min_z:g}.",
               [("check every label", "a lexicon learned from a handful of texts is noise")],
               {"with_support": len(labels), "with_lexicon": len(with_lex), "vocabulary": len(vocab)})
    if not with_lex:
        raise RuntimeError("no label has enough support and associated words to check captions against")
    wj = np.array([index[w] for c in with_lex for w, _ in lex[c]])
    lk = np.array([k for k, c in enumerate(with_lex) for _ in lex[c]])
    L = sp.csr_matrix((np.ones(len(wj)), (wj, lk)), shape=(len(vocab), len(with_lex)))
    # precision of each lexicon word: share of the training texts containing it that carry the label
    Dw = D[train][:, wj]
    both = np.asarray(Dw.multiply(Y[train][:, [with_lex[k] for k in lk]].astype(np.float32)).sum(0)).ravel()
    precision = both / np.maximum(np.asarray(Dw.sum(0)).ravel(), 1)
    precise = precision >= p.missing_min_precision
    Lm = sp.csr_matrix((np.ones(int(precise.sum())), (wj[precise], lk[precise])), shape=L.shape)
    aff_hits, neg_hits = (Aff @ L).toarray(), (Neg @ L).toarray()
    Yl = Y[:, with_lex].toarray()
    has_words = np.asarray(D.sum(1)).ravel() > 0          # an empty text neither supports nor contradicts
    unsupported = Yl & (aff_hits == 0) & (neg_hits == 0) & has_words[:, None]
    negated = Yl & (aff_hits == 0) & (neg_hits > 0)
    mentions = (Aff @ Lm).toarray() > 0
    missing = ~Yl & mentions
    checkable = np.bincount(lk[precise], minlength=len(with_lex)) > 0      # labels with a precise word
    heldout = ~train
    pos_n = Yl.sum(0)
    table = pd.DataFrame({"label": [names[c] for c in with_lex], "n_positive": pos_n,
                          "lexicon": [", ".join(w for w, _ in lex[c]) for c in with_lex],
                          "unsupported_rate": unsupported.sum(0) / np.maximum(pos_n, 1),
                          "negated_rate": negated.sum(0) / np.maximum(pos_n, 1),
                          "precise_words": [", ".join(vocab[j] for j, k2 in zip(wj[precise], lk[precise]) if k2 == k)
                                            for k in range(len(with_lex))],
                          "possible_missing_rate": np.where(checkable, missing.sum(0) / np.maximum(mentions.sum(0), 1),
                                                            np.nan)})
    flagged = unsupported | negated
    metrics = {"n_texts": len(rows), "n_texts_without_words": int((~has_words).sum()),
               "n_labels_checked": len(with_lex),
               "unsupported_rate": float(unsupported.sum() / max(1, Yl.sum())),
               "negated_rate": float(negated.sum() / max(1, Yl.sum())),
               "n_labels_checked_for_missing": int(checkable.sum()),
               "possible_missing_rate": float(missing.sum() / max(1, (~Yl[:, checkable]).sum())),
               "texts_with_flag_fraction": float((flagged.any(1) | missing.any(1)).mean()),
               "unsupported_rate_heldout": float(unsupported[heldout].sum() / max(1, Yl[heldout].sum()))
               if heldout.any() else None}
    if ref is not None:
        rrows = np.intersect1d(rows, ref.rows, assume_unique=True)
        R, rnames, rann = label_matrix(ref, rrows)
        shared = [(k, rnames.index(names[c])) for k, c in enumerate(with_lex) if names[c] in rnames]
        if shared:
            pos_r = np.searchsorted(rows, rrows[rann])
            Rm = R[rann][:, [j for _, j in shared]].toarray()
            ks = [k for k, _ in shared]
            sub_pos, sub_flag = Yl[pos_r][:, ks], flagged[pos_r][:, ks]
            absent = sub_pos & ~Rm
            metrics["reference_absent_given_flagged"] = float(absent[sub_flag].mean()) if sub_flag.any() else None
            metrics["reference_absent_given_supported"] = (float(absent[sub_pos & ~sub_flag].mean())
                                                           if (sub_pos & ~sub_flag).any() else None)
            metrics["n_reference_labels"] = len(shared)
    ex = []
    for i, k in zip(*np.nonzero(flagged)):
        if len(ex) >= 30:
            break
        ex.append({"row": int(rows[i]), "label": names[with_lex[k]],
                   "flag": "negated" if negated[i, k] else "unsupported", "text": raw[i][:160]})
    score = np.column_stack([unsupported.sum(1), negated.sum(1), missing.sum(1)]).astype(float)
    arts = {"flags": Artifact("scores", score, rows, {"columns": ["n_unsupported", "n_negated",
                                                                  "n_possible_missing"], "source": text.name})}
    return StepResult(artifacts=arts, metrics=metrics,
                      tables={"label_lexicons": table, "consistency_examples": pd.DataFrame(ex)})


register(ToolSpec(
    name="caption_label_consistency", origin="new", source=SOURCE,
    summary="Learns each label's indicative words from the training texts and flags labels their text does not "
            "support, labels the text negates, and texts that name a label the sample lacks.",
    consumes=(Port("text", ("text",)), Port("labels", ("multilabel", "labels")),
              Port("split", ("split",), required=False, doc="words are learned on the training split"),
              Port("reference", ("multilabel", "labels"), required=False, doc="second label source to check the "
                                                                              "flags against (report only)")),
    produces=(Port("flags", ("scores",), doc="per-text counts of unsupported, negated and possibly missing labels"),),
    params=ConsistencyParams, answers=("data_quality_audit",), addresses=("label_noise", "text_noise"),
    criteria=(Criterion("unsupported_rate", "report", doc="positive labels with none of their words in the text"),
              Criterion("negated_rate", "report", doc="positive labels whose words appear only negated"),
              Criterion("reference_absent_given_flagged", "report", selection=False,
                        doc="flagged labels the second source lacks"),
              Criterion("reference_absent_given_supported", "report", selection=False,
                        doc="supported labels the second source lacks")),
    run=_run_consistency, cost=lambda n, p, prm, b: 1e-4 * n + 2.0,
))

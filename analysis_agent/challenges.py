"""Step 3: the challenge register - the profile's measurements turned into analytical challenges.

Every challenge type of the taxonomy has a rule here: which profile measurements are its
evidence, and the named thresholds (``RULES``) that grade them none / low / medium / high.
Every type is checked for every artifact it concerns, and checks that found nothing stay in
the register ("not detected", "not applicable", "not measurable yet"), so the report can say
what was looked for as well as what was found.

A challenge is relevant when it threatens one of the objective's question types
(``taxonomy.CHALLENGE_AFFECTS``); without an objective every question counts. Its candidate
mitigations are the tools whose declared ``addresses`` include its type. The planner must
cover every relevant challenge of high severity with a step, or accept it with a reason.

Some challenges can only be measured once steps have run - pretraining overlap after
encoder screening, batch effects in learned image features. They enter as "not measurable"
and are updated from step results later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .schema import Challenge, ChallengeRegister, DataProfile, Evidence, Severity
from .taxonomy import CHALLENGE_AFFECTS, CHALLENGE_TYPES

SEVERITY_ORDER: dict[str, int] = {"none": 0, "low": 1, "medium": 2, "high": 3}


@dataclass(frozen=True)
class Rule:
    """Named thresholds for one measurement: the lowest level whose bound is reached."""
    metric: str
    low: float
    medium: float
    high: float
    higher_is_worse: bool = True

    def grade(self, value) -> Severity:
        if value is None or isinstance(value, bool) or not math.isfinite(float(value)):
            return "none"
        v = float(value)
        if self.higher_is_worse:
            return "high" if v >= self.high else "medium" if v >= self.medium else "low" if v >= self.low else "none"
        return "high" if v <= self.high else "medium" if v <= self.medium else "low" if v <= self.low else "none"

    def text(self) -> str:
        op = ">=" if self.higher_is_worse else "<="
        return f"low {op} {self.low:g}, medium {op} {self.medium:g}, high {op} {self.high:g}"


RULES: dict[str, Rule] = {
    "qc_flagged": Rule("share of samples failing the default count-QC rule", 0.005, 0.03, 0.10),
    "unusable_images": Rule("share of unreadable, tiny, strip-shaped or blank images", 0.001, 0.01, 0.05),
    "library_iqr": Rule("interquartile range of log2 library size", 0.5, 1.0, 2.0),
    "depth_pc1": Rule("|Spearman rho| of PC1 with log depth after library-size scaling", 0.1, 0.3, 0.5),
    "features_per_sample": Rule("features per sample (p / n)", 0.1, 1.0, 10.0),
    "sparsity": Rule("share of zero entries", 0.5, 0.9, 0.99),
    "group_silhouette": Rule("silhouette width by group", 0.02, 0.05, 0.15),
    "group_mixing": Rule("neighbour mixing of groups (1 = as mixed as overall)", 0.9, 0.8, 0.5, False),
    "colour_share": Rule("between-group share of image colour variance", 0.05, 0.2, 0.5),
    "cramers_v": Rule("Cramér's V between group and label", 0.1, 0.3, 0.5),
    "samples_per_unit": Rule("samples per independent unit (coarsest grouping)", 1.2, 2.0, 10.0),
    "autocorrelated": Rule("share of variable features with significant Moran's I", 0.05, 0.2, 0.5),
    "annotated": Rule("share of samples with a label", 0.99, 0.5, 0.1, False),
    "class_ratio": Rule("largest / smallest class size", 3.0, 10.0, 50.0),
    "too_rare": Rule("share of labels too rare to assess", 0.1, 0.5, 0.9),
    "kappa": Rule("Cohen's kappa between label sources", 0.95, 0.8, 0.6, False),
    "duplicates": Rule("share of images in a near-duplicate pair", 0.005, 0.02, 0.1),
    "group_overlap": Rule("share of a held-out split whose group also occurs in another split", 0.001, 0.05, 0.2),
    "text_flagged": Rule("share of texts empty, very short, repeated, boilerplate or referring to panels",
                         0.01, 0.05, 0.2),
    "missing": Rule("share of samples without the modality", 0.01, 0.1, 0.3),
    "size_spread": Rule("Q3 / Q1 of the shorter image side", 1.5, 2.0, 4.0),
    "colour_mix": Rule("share of the minority colour mode (greyscale vs colour)", 0.05, 0.2, 0.4),
    "hypotheses": Rule("hypotheses the questions imply", 2, 20, 1000),
    "compute_share": Rule("largest declared tool cost / step budget", 0.25, 0.5, 1.0),
}

# A label source must cover this share of the first source's assignments for its agreement to
# stand for the whole first source; below it, label noise is at least "low" (partly unmeasured).
AGREEMENT_COVERAGE = 0.9
STRUCTURE_QUESTIONS = ("structure_discovery", "spatial_structure")


def worst(*levels: Severity) -> Severity:
    return max(levels, key=lambda s: SEVERITY_ORDER[s]) if levels else "none"


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool) or isinstance(v, int):
        return f"{v:,}" if not isinstance(v, bool) else str(v)
    if abs(v) >= 100:
        return f"{v:,.0f}"
    return f"{v:.3g}"


def _ev(rule_key: str | None, metric: str, value, source: str) -> Evidence:
    rule = RULES[rule_key].text() if rule_key else ""
    return Evidence(metric=metric, value=value, rule=rule, source=source)


class _Builder:
    def __init__(self, questions: list[str] | None):
        from .tools import registry

        self.questions = questions
        self.entries: list[Challenge] = []
        self.tools = {t: sorted(s.name for s in registry.addressing(t)) for t in CHALLENGE_TYPES}

    def add(self, kind: str, subject: list[str], severity: Severity, summary: str, evidence: list[Evidence],
            status: str | None = None) -> Challenge:
        affects = list(CHALLENGE_AFFECTS[kind])
        relevant = self.questions is None or bool(set(affects) & set(self.questions))
        if status is None:
            status = "detected" if severity != "none" else "not_detected"
        cid = kind + (":" + ",".join(subject) if subject else "")
        c = Challenge(id=cid, type=kind, subject=subject, summary=summary, status=status,
                      severity=severity if status == "detected" else "none", evidence=evidence, affects=affects,
                      relevant=relevant, mitigations=self.tools[kind])
        self.entries.append(c)
        return c

    def absent(self, kind: str, why: str, status: str = "not_applicable") -> None:
        self.add(kind, [], "none", why, [], status)


def assess(profile: DataProfile, questions: list[str] | None = None) -> ChallengeRegister:
    """Build the register from a profile; ``questions`` are the objective's question types
    (None: judge relevance as if every question were asked)."""
    b = _Builder(questions)
    _low_quality(b, profile)
    _depth(b, profile)
    _dimensionality(b, profile)
    _batch(b, profile)
    _confounding(b, profile)
    _nesting(b, profile)
    _double_dipping(b, profile, questions)
    _spatial(b, profile)
    _labels(b, profile)
    _noise(b, profile)
    _duplicates(b, profile)
    _leakage(b, profile)
    _text(b, profile)
    _missing(b, profile)
    _images(b, profile)
    _testing(b, profile, questions)
    _compute(b, profile)
    _pretraining(b, profile)
    order = {t: i for i, t in enumerate(CHALLENGE_TYPES)}
    entries = sorted(b.entries, key=lambda c: (order[c.type], c.id))
    return ChallengeRegister(questions=questions, entries=entries,
                             thresholds={k: {"metric": r.metric, "low": r.low, "medium": r.medium, "high": r.high,
                                             "higher_is_worse": r.higher_is_worse} for k, r in RULES.items()})


# ---------------------------------------------------------------------------
# rules, one per challenge type
# ---------------------------------------------------------------------------

def _low_quality(b: _Builder, p: DataProfile) -> None:
    found = False
    for m in p.matrices:
        if m.qc_flagged_fraction is None:
            continue
        found = True
        sev = RULES["qc_flagged"].grade(m.qc_flagged_fraction)
        if m.n_empty:
            sev = worst(sev, "low")                 # an empty sample breaks normalisation
        where = "within each library" if m.qc_per_library else "over all samples"
        b.add("low_quality_samples", [m.artifact], sev,
              f"{m.qc_flagged_fraction:.1%} of samples in {m.artifact} fail the default quality rule (3 MADs and "
              f"2-fold below the median depth or detected features, {where}); {m.n_empty} are empty.",
              [_ev("qc_flagged", RULES["qc_flagged"].metric, m.qc_flagged_fraction, f"matrices[{m.artifact}]"),
               _ev(None, "empty samples", m.n_empty, f"matrices[{m.artifact}]")])
    for im in p.images:
        if im.unusable_fraction is None:
            continue
        found = True
        sev = RULES["unusable_images"].grade(im.unusable_fraction)
        b.add("low_quality_samples", [im.artifact], sev,
              f"{im.unusable_fraction:.1%} of the {im.n_measured:,} measured images in {im.artifact} are unreadable, "
              "tiny, strip-shaped or blank.",
              [_ev("unusable_images", RULES["unusable_images"].metric, im.unusable_fraction, f"images[{im.artifact}]"),
               *[_ev(None, f"{k} images", v, f"images[{im.artifact}]") for k, v in im.flag_counts.items()]])
    if not found:
        b.absent("low_quality_samples", "No count matrix or images to check sample quality on.")


def _depth(b: _Builder, p: DataProfile) -> None:
    counts = [m for m in p.matrices if m.is_counts]
    for m in counts:
        sev = worst(RULES["library_iqr"].grade(m.library_size_log2_iqr),
                    RULES["depth_pc1"].grade(m.pc1_depth_rho_scaled))
        rng = (f"; per-library medians from {_fmt(m.library_median_range[0])} to {_fmt(m.library_median_range[1])}"
               if m.library_median_range else "")
        b.add("depth_variation", [m.artifact], sev,
              f"Library sizes in {m.artifact} span an interquartile range of {_fmt(m.library_size_log2_iqr)} on the "
              f"log2 scale{rng}, and PC1 still tracks depth after library-size scaling "
              f"(|rho| {_fmt(m.pc1_depth_rho_scaled)}).",
              [_ev("library_iqr", RULES["library_iqr"].metric, m.library_size_log2_iqr, f"matrices[{m.artifact}]"),
               _ev("depth_pc1", RULES["depth_pc1"].metric, m.pc1_depth_rho_scaled, f"matrices[{m.artifact}]"),
               _ev(None, "|rho(PC1, log depth)| on log1p of raw counts", m.pc1_depth_rho_raw,
                   f"matrices[{m.artifact}]")])
    if not counts:
        b.absent("depth_variation", "No count matrix: sequencing depth does not apply.")


def _dimensionality(b: _Builder, p: DataProfile) -> None:
    for m in p.matrices:
        sev = worst(RULES["features_per_sample"].grade(m.features_per_sample), RULES["sparsity"].grade(m.sparsity))
        b.add("high_dimensionality", [m.artifact], sev,
              f"{m.artifact} has {m.n_features:,} features for {m.n_samples:,} samples ({_fmt(m.features_per_sample)} "
              f"per sample) and {m.sparsity:.0%} zeros; 50 components hold {_fmt(m.variance_top_components)} of the "
              "variance of the most variable features.",
              [_ev("features_per_sample", RULES["features_per_sample"].metric, m.features_per_sample,
                   f"matrices[{m.artifact}]"),
               _ev("sparsity", RULES["sparsity"].metric, m.sparsity, f"matrices[{m.artifact}]"),
               _ev(None, "share of variance in 50 principal components", m.variance_top_components,
                   f"matrices[{m.artifact}]")])
    if not p.matrices:
        b.absent("high_dimensionality", "No feature matrix; images and texts are reduced by encoders.")


def _batch(b: _Builder, p: DataProfile) -> None:
    for g in p.group_structure:
        nested = g.outer is not None
        sil = g.within_outer_silhouette if nested else g.silhouette
        mix = g.within_outer_mixing if nested else g.mixing
        sev = worst(RULES["group_silhouette"].grade(sil), RULES["group_mixing"].grade(mix))
        src = f"group_structure[{g.matrix}, {g.group}]"
        if nested:
            summary = (f"Within each {_col(p, g.outer)}, samples separate by {g.column} (silhouette {_fmt(sil)}, "
                       f"neighbour mixing {_fmt(mix)}); {g.column}s inside one {_col(p, g.outer)} explain "
                       f"{_fmt(g.share_between_inner_within_outer)} of the variance, {_col(p, g.outer)}s "
                       f"{_fmt(g.share_between_outer)}.")
            ev = [_ev("group_silhouette", f"silhouette by {g.column} within {_col(p, g.outer)}", sil, src),
                  _ev("group_mixing", f"neighbour mixing of {g.column}s within {_col(p, g.outer)}", mix, src),
                  _ev(None, f"variance share between {g.column}s of one {_col(p, g.outer)}",
                      g.share_between_inner_within_outer, src),
                  _ev(None, f"variance share between {_col(p, g.outer)}s", g.share_between_outer, src)]
        else:
            summary = (f"Samples separate by {g.column} (silhouette {_fmt(sil)}, neighbour mixing {_fmt(mix)}, "
                       f"{_fmt(g.variance_share)} of the variance between {g.column}s)"
                       + (f"; {g.column} differences mix biology and technical effects."
                          if _is_outer(p, g.group) else "."))
            ev = [_ev("group_silhouette", f"silhouette by {g.column}", sil, src),
                  _ev("group_mixing", f"neighbour mixing of {g.column}s", mix, src),
                  _ev(None, f"variance share between {g.column}s", g.variance_share, src)]
        b.add("batch_effect", [g.group, g.matrix], sev, summary, ev)
    for im in p.images:
        if im.colour_between_group_share is None:
            continue
        sev = RULES["colour_share"].grade(im.colour_between_group_share)
        b.add("batch_effect", [im.colour_group or "", im.artifact], sev,
              f"Mean image colour differs between {_col(p, im.colour_group)} levels (between-group share "
              f"{_fmt(im.colour_between_group_share)}), as staining or scanning batches do.",
              [_ev("colour_share", RULES["colour_share"].metric, im.colour_between_group_share,
                   f"images[{im.artifact}]")])
    if not p.group_structure and not any(i.colour_between_group_share is not None for i in p.images):
        why = ("No grouping with levels large enough to measure." if p.groups else "No grouping of samples.")
        status = "not_applicable" if not p.groups else "not_measurable"
        if p.groups and not p.matrices:
            why = ("Groups are too small to compare image colour; group effects in learned features are measured "
                   "after feature extraction.")
        b.absent("batch_effect", why, status)


def _confounding(b: _Builder, p: DataProfile) -> None:
    found = False
    for lab in p.labels:
        for grp, v in lab.cramers_v.items():
            found = True
            sev = RULES["cramers_v"].grade(v)
            b.add("batch_confounding", [grp, lab.artifact], sev,
                  f"Label '{lab.column}' is associated with {_col(p, grp)} (Cramér's V {_fmt(v)}): a correction or "
                  f"comparison across {_col(p, grp)}s also touches the labelled biology.",
                  [_ev("cramers_v", RULES["cramers_v"].metric, v, f"labels[{lab.artifact}]")])
    if not found:
        b.absent("batch_confounding", "No single-label column with groups large enough to cross-tabulate.",
                 "not_applicable" if not p.labels or not p.groups else "not_measurable")


def _nesting(b: _Builder, p: DataProfile) -> None:
    if not p.groups:
        b.absent("nested_sampling", "No grouping of samples: every sample is its own unit.")
        return
    unit = min(p.groups, key=lambda g: g.n_levels)          # the coarsest grouping is the independent unit
    sev = RULES["samples_per_unit"].grade(unit.samples_per_level)
    inner = [g.column for g in p.groups if unit.artifact in g.nested_in]
    b.add("nested_sampling", [unit.artifact], sev,
          f"The {p.n_samples:,} samples come from {unit.n_levels:,} {unit.column} levels "
          f"({_fmt(unit.samples_per_level)} per level" + (f", with {', '.join(inner)} nested inside" if inner else "")
          + f"); {unit.shared_fraction:.0%} of samples share their {unit.column} with others, so the independent "
          f"unit is the {unit.column}.",
          [_ev("samples_per_unit", RULES["samples_per_unit"].metric, unit.samples_per_level,
               f"groups[{unit.artifact}]"),
           _ev(None, f"{unit.column} levels", unit.n_levels, f"groups[{unit.artifact}]"),
           _ev(None, f"share of samples sharing their {unit.column}", unit.shared_fraction,
               f"groups[{unit.artifact}]")])


def _double_dipping(b: _Builder, p: DataProfile, questions) -> None:
    counts = [m.artifact for m in p.matrices if m.is_counts]
    if not counts:
        b.absent("double_dipping", "No count matrix to find groups in and then test them on.")
        return
    if questions is not None and not ("differential_analysis" in questions
                                      and any(q in questions for q in STRUCTURE_QUESTIONS)):
        b.absent("double_dipping", "The objective does not ask to test features between groups it discovers.")
        return
    b.add("double_dipping", counts, "high",
          f"Groups found in {counts[0]} and then tested for marker features on the same counts would be tested on "
          "the data that defined them, inflating significance.",
          [_ev(None, "questions implying discovery then testing",
               ", ".join(q for q in ("differential_analysis", *STRUCTURE_QUESTIONS) if questions is None
                         or q in questions), "objective")])


def _spatial(b: _Builder, p: DataProfile) -> None:
    for s in p.spatial:
        if s.fraction_autocorrelated is None:
            b.add("spatial_autocorrelation", [s.artifact], "none", f"No feature matrix to test {s.artifact} on.", [],
                  "not_measurable")
            continue
        sev = RULES["autocorrelated"].grade(s.fraction_autocorrelated)
        b.add("spatial_autocorrelation", [s.artifact, s.matrix or ""], sev,
              f"{s.fraction_autocorrelated:.0%} of the {s.features_tested} most variable features of {s.matrix} are "
              f"spatially autocorrelated within libraries (median Moran's I {_fmt(s.median_morans_i)}): neighbouring "
              "samples are not independent.",
              [_ev("autocorrelated", RULES["autocorrelated"].metric, s.fraction_autocorrelated,
                   f"spatial[{s.artifact}]"),
               _ev(None, "median Moran's I", s.median_morans_i, f"spatial[{s.artifact}]")])
    if not p.spatial:
        b.absent("spatial_autocorrelation", "No spatial coordinates.")


def _labels(b: _Builder, p: DataProfile) -> None:
    for lab in p.labels:
        src = f"labels[{lab.artifact}]"
        sev = RULES["annotated"].grade(lab.annotated_fraction)
        cover = "; ".join(f"{_col(p, g)}: {v[0]} of {v[1]} levels" for g, v in lab.annotated_groups.items()
                          if v[0] < v[1])
        b.add("partial_labels", [lab.artifact], sev,
              f"{lab.annotated_fraction:.0%} of samples carry '{lab.column}'"
              + (f" ({cover} labelled)" if cover else "") + ".",
              [_ev("annotated", RULES["annotated"].metric, lab.annotated_fraction, src),
               *[_ev(None, f"{_col(p, g)} levels labelled", f"{v[0]} of {v[1]}", src)
                 for g, v in lab.annotated_groups.items()]])
        sev = worst(RULES["class_ratio"].grade(lab.imbalance_ratio), RULES["too_rare"].grade(lab.too_rare_fraction))
        b.add("class_imbalance", [lab.artifact], sev,
              f"'{lab.column}' has {lab.n_classes:,} {'labels' if lab.kind == 'multilabel' else 'classes'}, the "
              f"largest {_fmt(lab.imbalance_ratio)} times the smallest; {lab.n_assessable} have the support to be "
              f"modelled and judged ({lab.too_rare_fraction:.0%} are too rare to assess).",
              [_ev("class_ratio", RULES["class_ratio"].metric, lab.imbalance_ratio, src),
               _ev("too_rare", RULES["too_rare"].metric, lab.too_rare_fraction, src),
               _ev(None, "labels with support to model and judge", lab.n_assessable, src),
               *([_ev(None, "Gini of label frequencies", lab.gini, src)] if lab.gini is not None else [])])
    if not p.labels:
        b.absent("partial_labels", "No label column.")
        b.absent("class_imbalance", "No label column.")


def _noise(b: _Builder, p: DataProfile) -> None:
    for a in p.label_agreement:
        sev = RULES["kappa"].grade(a.kappa_micro)
        partial = a.first_source_coverage is not None and a.first_source_coverage < AGREEMENT_COVERAGE
        if partial:
            sev = worst(sev, "low")
        b.add("label_noise", [a.first, a.second], sev,
              f"{a.first} and {a.second} agree with kappa {_fmt(a.kappa_micro)} on {a.n_shared_labels} shared labels "
              f"(precision {_fmt(a.precision_micro)}, recall {_fmt(a.recall_micro)})"
              + (f"; those labels hold {a.first_source_coverage:.0%} of {a.first}'s assignments, so the rest is "
                 "unmeasured" if partial else "") + ".",
              [_ev("kappa", RULES["kappa"].metric, a.kappa_micro, f"label_agreement[{a.first}]"),
               _ev(None, "share of the first source's assignments on shared labels", a.first_source_coverage,
                   f"label_agreement[{a.first}]")])
    measured = {x for a in p.label_agreement for x in (a.first, a.second)}
    for lab in p.labels:
        if lab.artifact in measured:
            continue
        how = ("caption-label consistency can estimate it after the run" if p.texts
               else "no second label source or text to check it against")
        b.add("label_noise", [lab.artifact], "none", f"Noise in '{lab.column}' is not measurable from the profile: "
              f"{how}.", [], "not_measurable")
    if not p.labels:
        b.absent("label_noise", "No label column.")


def _duplicates(b: _Builder, p: DataProfile) -> None:
    hashed = [i for i in p.images if i.duplicate_fraction is not None]
    for im in hashed:
        sev = RULES["duplicates"].grade(im.duplicate_fraction)
        if im.cross_split_duplicate_pairs:
            sev = worst(sev, "medium")              # a copy on both sides of a split leaks
        b.add("near_duplicates", [im.artifact], sev,
              f"{im.duplicate_fraction:.1%} of the {im.n_hashed:,} hashed images in {im.artifact} have a near-copy "
              f"({im.n_duplicate_pairs:,} pairs: 64-bit difference-hash matches confirmed at 256 bits, of "
              f"{im.n_hash_candidates or 0:,} 64-bit matches)"
              + (f"; {im.cross_split_duplicate_pairs} pairs cross the predefined split"
                 if im.cross_split_duplicate_pairs is not None else "") + ".",
              [_ev("duplicates", RULES["duplicates"].metric, im.duplicate_fraction, f"images[{im.artifact}]"),
               _ev(None, "near-duplicate pairs crossing the split", im.cross_split_duplicate_pairs,
                   f"images[{im.artifact}]"),
               _ev(None, "64-bit hash matches before confirmation", im.n_hash_candidates, f"images[{im.artifact}]")])
    for im in p.images:
        if im.duplicate_fraction is None:
            b.add("near_duplicates", [im.artifact], "none", f"{im.artifact} are cut from larger images, so copies "
                  "are not expected and not hashed.", [], "not_applicable")
    if not p.images:
        b.absent("near_duplicates", "No images.")


def _leakage(b: _Builder, p: DataProfile) -> None:
    if p.split is None:
        b.absent("split_leakage", "No predefined split: the agent makes grouped splits itself.")
        return
    held = [s for s in p.split.counts if s != "train"]
    found = False
    for g in p.groups:
        if not g.split_overlap:
            continue
        found = True
        shares = {s: g.split_overlap.get(s, 0.0) for s in held}
        top = max(shares, key=shares.get) if shares else None
        sev = RULES["group_overlap"].grade(shares[top]) if top else "none"
        b.add("split_leakage", [g.artifact, p.split.artifact], sev,
              "; ".join(f"{v:.0%} of the {s} split shares its {g.column} with another split" for s, v in shares.items())
              + ".",
              [_ev("group_overlap", f"share of the {s} split whose {g.column} occurs in another split", v,
                   f"groups[{g.artifact}]") for s, v in shares.items()])
    for im in p.images:
        if im.cross_split_duplicate_pairs:
            found = True
            b.add("split_leakage", [im.artifact, p.split.artifact], "medium",
                  f"{im.cross_split_duplicate_pairs} near-duplicate image pairs lie on different sides of the split.",
                  [_ev(None, "near-duplicate pairs crossing the split", im.cross_split_duplicate_pairs,
                       f"images[{im.artifact}]")])
    if not found:
        b.add("split_leakage", [p.split.artifact], "none", "No grouping or near-copy links samples across the split.",
              [], "not_detected")


def _text(b: _Builder, p: DataProfile) -> None:
    for t in p.texts:
        sev = RULES["text_flagged"].grade(t.flagged_fraction)
        parts = ", ".join(f"{k.replace('_', ' ')} {v:.1%}" for k, v in t.flag_fractions.items() if v > 0)
        b.add("text_noise", [t.artifact], sev,
              f"{t.flagged_fraction:.1%} of {t.artifact} are flagged ({parts or 'none'}); median "
              f"{_fmt(t.words_median)} words.",
              [_ev("text_flagged", RULES["text_flagged"].metric, t.flagged_fraction, f"texts[{t.artifact}]"),
               *[_ev(None, f"share {k.replace('_', ' ')}", v, f"texts[{t.artifact}]")
                 for k, v in t.flag_fractions.items()]])
    if not p.texts:
        b.absent("text_noise", "No text.")


def _missing(b: _Builder, p: DataProfile) -> None:
    found = False
    for m in p.modalities:
        n_missing = round(m.missing_fraction * p.n_samples)
        if n_missing == 0:
            continue
        found = True
        b.add("missing_data", [m.name], RULES["missing"].grade(m.missing_fraction),
              f"{n_missing:,} of {p.n_samples:,} samples ({m.missing_fraction:.1%}) have no {m.name}.",
              [_ev("missing", RULES["missing"].metric, m.missing_fraction, f"modalities[{m.name}]")])
    for t in p.tabular:
        if t.columns_with_missing == 0:
            continue
        found = True
        b.add("missing_data", [t.artifact], RULES["missing"].grade(t.missing_cell_fraction),
              f"{t.missing_cell_fraction:.1%} of the cells of {t.artifact} are empty, in {t.columns_with_missing} of "
              f"its {t.n_columns} columns.",
              [_ev("missing", "share of empty cells", t.missing_cell_fraction, f"tabular[{t.artifact}]"),
               _ev(None, "columns with empty cells", t.columns_with_missing, f"tabular[{t.artifact}]")])
    if not found:
        b.add("missing_data", [], "none", "Every sample has every modality and no table cell is empty.", [],
              "not_detected")


def _images(b: _Builder, p: DataProfile) -> None:
    for im in p.images:
        mix = None if im.greyscale_fraction is None else min(im.greyscale_fraction, 1 - im.greyscale_fraction)
        sev = worst(RULES["size_spread"].grade(im.size_iqr_ratio), RULES["colour_mix"].grade(mix))
        size = ("sizes are uniform" if (im.size_iqr_ratio or 1.0) < 1.05 else
                f"the shorter side differs {_fmt(im.size_iqr_ratio)}-fold between its quartiles")
        grey = "n/a" if im.greyscale_fraction is None else f"{im.greyscale_fraction:.0%}"
        b.add("image_heterogeneity", [im.artifact], sev,
              f"In {im.artifact} {size} (median {_fmt(im.width_median)} x {_fmt(im.height_median)} px) and {grey} of "
              f"{im.n_colour_sampled} sampled images are greyscale, the rest colour.",
              [_ev("size_spread", RULES["size_spread"].metric, im.size_iqr_ratio, f"images[{im.artifact}]"),
               _ev("colour_mix", RULES["colour_mix"].metric, mix, f"images[{im.artifact}]")])
    if not p.images:
        b.absent("image_heterogeneity", "No images.")


def _testing(b: _Builder, p: DataProfile, questions) -> None:
    asked = set(questions) if questions is not None else None
    counts: dict[str, int] = {}
    feats = max((m.n_features_detected_5pct or 0 for m in p.matrices), default=0)
    if feats and (asked is None or "differential_analysis" in asked):
        counts["features compared between groups"] = feats
    if p.spatial and feats and (asked is None or "spatial_structure" in asked):
        counts["features tested for spatial patterns"] = feats
    modelable = max((lab.n_modelable or 0 for lab in p.labels), default=0)
    if modelable and (asked is None or "predictive_modelling" in asked):
        counts["targets modelled"] = modelable
    if not counts:
        b.absent("multiple_testing", "The questions imply no family of tests.")
        return
    top = max(counts, key=counts.get)
    b.add("multiple_testing", [], RULES["hypotheses"].grade(counts[top]),
          "; ".join(f"{_fmt(v)} {k}" for k, v in counts.items()) + ": without error control, false positives are "
          "expected among them.",
          [_ev("hypotheses", f"{RULES['hypotheses'].metric}: {k}", v, "profile") for k, v in counts.items()])


def _compute(b: _Builder, p: DataProfile) -> None:
    if p.compute is None or not p.compute.estimates:
        b.absent("compute", "No cost estimates.", "not_measurable")
        return
    top = p.compute.estimates[0]
    share = top.seconds / max(1.0, p.compute.step_seconds)
    b.add("compute", [], RULES["compute_share"].grade(share),
          f"The most expensive tool on these data, {top.tool}, is estimated at {_fmt(top.seconds)} s, "
          f"{share:.0%} of the {_fmt(p.compute.step_seconds)} s step budget on {p.compute.device}.",
          [_ev("compute_share", RULES["compute_share"].metric, share, "compute"),
           *[_ev(None, f"estimated seconds: {e.tool}", round(e.seconds, 1), "compute")
             for e in p.compute.estimates[:4]]])


def _pretraining(b: _Builder, p: DataProfile) -> None:
    if p.images or p.texts:
        b.add("pretraining_overlap", [i.artifact for i in p.images] + [t.artifact for t in p.texts], "none",
              "Pretrained encoders may have seen these images or texts; this is judged after encoder screening, from "
              "zero-shot retrieval far above what the training split supports.", [], "not_measurable")
    else:
        b.absent("pretraining_overlap", "No images or texts for pretrained encoders.")


def _col(p: DataProfile, artifact: str | None) -> str:
    g = next((g for g in p.groups if g.artifact == artifact), None)
    return g.column if g is not None else (artifact or "group")


def _is_outer(p: DataProfile, artifact: str) -> bool:
    return any(artifact in g.nested_in for g in p.groups)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(reg: ChallengeRegister) -> str:
    """The register as Markdown: detected challenges by severity, then the checks that found nothing."""
    q = ", ".join(reg.questions) if reg.questions else "all question types (no objective given)"
    lines = ["# Challenge register", "", f"Relevance judged against: {q}.", ""]
    det = sorted(reg.detected("low", relevant_only=False),
                 key=lambda c: (-int(c.relevant), -SEVERITY_ORDER[c.severity], c.id))
    lines += ["| challenge | severity | relevant | evidence | candidate mitigations |", "|---|---|---|---|---|"]
    for c in det:
        ev = "; ".join(f"{e.metric} = {_fmt(e.value) if not isinstance(e.value, str) else e.value}"
                       + (f" ({e.rule})" if e.rule else "") for e in c.evidence[:3])
        subject = ", ".join(s for s in c.subject if s) or "dataset"
        lines.append(f"| **{c.type.replace('_', ' ')}** ({subject}): {_cell(c.summary)} | {c.severity} | "
                     f"{'yes' if c.relevant else 'no'} | {_cell(ev)} | "
                     f"{', '.join(c.mitigations) or 'planning (subsample, cap)'} |")
    rest = [c for c in reg.entries if c.status != "detected"]
    if rest:
        lines += ["", "## Checked, not found or not measurable yet", ""]
        for c in rest:
            lines.append(f"- {c.type.replace('_', ' ')} ({c.status.replace('_', ' ')}): {c.summary}")
    lines += ["", "## Thresholds", "", "| rule | measurement | low | medium | high |", "|---|---|---|---|---|"]
    for k, r in RULES.items():
        op = ">=" if r.higher_is_worse else "<="
        lines.append(f"| {k} | {_cell(r.metric)} | {op} {r.low:g} | {op} {r.medium:g} | {op} {r.high:g} |")
    return "\n".join(lines) + "\n"

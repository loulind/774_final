"""Step 2: the objective, read as question types with targets (rules; the LLM may refine them).

Cue phrases (``CUES``) mark each question type of the taxonomy in the objective's wording,
and the challenge types it names (``CHALLENGE_CUES``: "class imbalance", "near-duplicates",
"noisy captions") are attached to its data-quality question. Its nouns are then matched
against the data, never against a list of datasets: column names (through the artifacts
the columns became), modality words (image, caption, gene, spatial) and the image domain
the profile measured. Each question ends up naming the artifacts it is about - what to
predict and from what, which grouping is the batch, which label source is the reference.
"Reliably" fixes the configured reliability rule as a constraint before any result is seen.
Visualisation is always added as an implied question, because the report needs maps.
"""

from __future__ import annotations

import re

from .config import Criteria
from .schema import ArtifactRecord, DataProfile, Decision, Objective, Question

CUES: dict[str, tuple[str, ...]] = {
    "structure_discovery": (r"\bdomains?\b", r"\bclusters?\b", r"\bclustering\b", r"\bsub-?types?\b",
                            r"\bsub-?groups?\b", r"\bsub-?populations?\b", r"\bpopulations?\b", r"\bcell types?\b",
                            r"\bcell states?\b", r"\bheterogeneity\b", r"\bdiscover", r"\bsegment"),
    "spatial_structure": (r"\bspatial(ly)?\b", r"\bniches?\b", r"\bneighbou?rhoods?\b", r"\barchitecture\b",
                          r"\bdomains?\b"),
    "cross_modal_association": (r"\brelate\b", r"\brelat(ion|ionship)s?\b", r"\bassociat", r"\bcorrespond",
                                r"\balign", r"\bagreement between\b", r"\bcross-modal\b", r"\bmodalit(y|ies)\b",
                                r"\bintegrat(e|es|ed|ing|ion)\b", r"\bjoint(ly)?\b", r"\bconcordan"),
    "differential_analysis": (r"\bmarkers?\b", r"\bdifferential(ly)?\b", r"\bsignatures?\b",
                              r"\bdiffer(s|ences)? between\b", r"\benriched (genes|features|proteins)\b",
                              r"\b(change|changes|alter|alters|affect|affects)\b(?:\W+\w+){0,2}\W+expression\b",
                              r"\b(up|down)-?regulat"),
    "batch_assessment": (r"\bbatch(es)?\b", r"\btechnical (variation|effects?)\b", r"\bsite effects?\b"),
    "predictive_modelling": (r"\bpredict(ed|ion|ions|able|ability|s|ing)?\b", r"\bclassif(y|ied|ication|ier)\b",
                             r"\bprognos", r"\bdiagnos(e|is|tic)\b"),
    "data_quality_audit": (r"\bdata[- ]quality\b", r"\bquality (issues|problems|control)\b", r"\bimbalance",
                           r"\bduplicat", r"\bnois(y|e)\b", r"\bmissing\b", r"\bleakage\b", r"\boutliers?\b",
                           r"\bquality\b(?:\W+\w+){0,3}\W+(issues|problems)\b", r"\b(low|poor)[- ]quality\b"),
    "representation_learning": (r"\bembeddings?\b", r"\brepresentations?\b", r"\blatent\b"),
    "generative_modelling": (r"\bgenerative\b", r"\bdensity\b", r"\banomal", r"\bimput", r"\bsynthes"),
    "visualization": (r"\bvisuali[sz]", r"\bmaps?\b", r"\bplots?\b", r"\bumap\b", r"\bt-?sne\b"),
}

CHALLENGE_CUES: dict[str, tuple[str, ...]] = {
    "class_imbalance": (r"\bimbalance", r"\blong[- ]tail", r"\brare (classes|labels|concepts|categories)\b"),
    "near_duplicates": (r"\bduplicat", r"\bcopies\b", r"\bredundan"),
    "text_noise": (r"\bnoisy (captions?|texts?|reports?)\b", r"\b(caption|text) (noise|quality)\b"),
    "label_noise": (r"\bnoisy (labels?|annotations?|captions?|concepts?)\b", r"\blabel (noise|errors?)\b",
                    r"\bmislabel", r"\binter-?(rater|reader|observer|annotator)\b",
                    r"\b(readers?|raters?|annotators?|observers?)\b(?:\W+\w+){0,3}\W+(dis)?agree",
                    r"\b(dis)?agree\w*\b(?:\W+\w+){0,3}\W+(readers?|raters?|annotators?|observers?)\b",
                    r"\b(reader|rater|annotator|observer) (dis)?agree"),
    "missing_data": (r"\bmissing\b",),
    "split_leakage": (r"\bleak",),
    "low_quality_samples": (r"\blow[- ]quality\b", r"\bunusable\b", r"\bcorrupt",
                            r"\b(cell|sample|image)[- ]quality\b"),
    "depth_variation": (r"\b(sequencing|read|library)[- ]depth\b", r"\blibrary sizes?\b"),
    "batch_effect": (r"\bbatch(es)?\b", r"\bsite effects?\b", r"\btechnical (variation|effects?)\b"),
}

# Words that point at a kind of data, matched against the objective's (singular) tokens.
MODALITY_WORDS: dict[str, frozenset[str]] = {
    "images": frozenset("image imaging picture photo scan radiograph x ray histology histological histopathology "
                        "morphology microscopy stain stained figure slide-image patch ct mri tomography ultrasound "
                        "mammogram radiology radiological".split()),
    "text": frozenset("text caption report description note".split()),
    "matrix": frozenset("gene expression transcript transcriptome rna count protein peak abundance chromatin "
                        "accessibility atac methylation".split()),
    "coords": frozenset("spatial spatially location coordinate neighbourhood neighborhood".split()),
}
STOPWORDS = frozenset("a an and are as at be by can for from how in into is it its of on or such that the them "
                      "these this to which with what whether level data issue issues identify find assess "
                      "determine characterize characterise relate".split())


def _singular(w: str) -> str:
    if len(w) > 4 and w.endswith(("ches", "shes", "xes", "sses")):
        return w[:-2]
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _tokens(text: str) -> set[str]:
    return {_singular(w) for w in re.findall(r"[a-z0-9]+", text.lower())} - STOPWORDS


VERBS = ("identify|find|assess|determine|relate|characteri[sz]e|predict|estimate|compare|test|describe|map|"
         "quantify|evaluate|measure|detect|discover|cluster|classify|visuali[sz]e|explain|examine|investigate")
CLAUSE_BREAK = re.compile(rf"[.;:]\s*|,?\s+and\s+(?=(?:{VERBS})\b)|,\s*(?=(?:{VERBS})\b)", re.I)


def _clause(text: str, pos: int) -> str:
    """The clause of the objective around a match: sentences and colon / semicolon breaks, and
    commas or "and" before a new verb ("identify ..., relate ..., and assess ...")."""
    breaks = list(CLAUSE_BREAK.finditer(text))
    starts = [0] + [m.end() for m in breaks]
    ends = [m.start() for m in breaks] + [len(text)]
    for a, b in zip(starts, ends):
        if a <= pos < max(b, a + 1):
            return re.sub(r"^(and|or|then)\s+", "", text[a:b].strip(" .;:,"), flags=re.I)
    return text.strip()


def _column_tokens(a: ArtifactRecord) -> set[str]:
    col = str(a.detail.get("column") or a.name.split(".", 1)[-1])
    return {_singular(w) for w in re.findall(r"[a-z0-9]+", col.lower())}


def _named_first(names: list[str], phrase: str) -> list[str]:
    """Artifacts whose own words (``counts.gene_expression`` -> gene, expression) occur in the
    phrase first; the profile's order otherwise."""
    words = _tokens(phrase)
    return sorted(names, key=lambda a: not {_singular(w) for w in re.findall(r"[a-z0-9]+", a.split(".", 1)[-1].lower())}
                  & words)


def _modality_artifacts(profile: DataProfile) -> dict[str, list[str]]:
    kinds = {"images": ("images",), "text": ("text",), "matrix": ("counts", "normalized"), "coords": ("coords",)}
    return {m: [a.name for a in profile.artifacts if a.kind in ks] for m, ks in kinds.items()}


def interpret(text: str, profile: DataProfile, criteria: Criteria | None = None) -> Objective:
    """Question types, targets and constraints of an objective, for these data."""
    criteria = criteria or Criteria()
    low = text.lower()
    tokens = _tokens(text)
    mods = _modality_artifacts(profile)
    mentioned = {m for m, words in MODALITY_WORDS.items() if tokens & words and mods[m]}
    labels = [a for a in profile.artifacts if a.kind in ("labels", "multilabel")]
    groups = [a for a in profile.artifacts if a.kind == "groups"]
    named_labels = sorted((a for a in labels if _column_tokens(a) & tokens),
                          key=lambda a: -int(a.detail.get("n_labels") or a.detail.get("n_classes") or 0))
    named_groups = [a for a in groups if _column_tokens(a) & tokens]
    notes: list[str] = []
    obj = Objective(text=text)

    for kind, pats in CUES.items():
        hit = next((m for p in pats if (m := re.search(p, low))), None)
        if hit is None:
            continue
        obj.questions.append(Question(type=kind, phrase=_clause(text, hit.start()), evidence=hit.group(0)))
    asked = {q.type for q in obj.questions}

    if "spatial_structure" in asked and not mods["coords"]:
        notes.append("the objective mentions spatial structure but the data have no coordinates; read as "
                     "structure discovery")
        obj.questions = [q for q in obj.questions if q.type != "spatial_structure"]
        if "structure_discovery" not in asked:
            obj.questions.append(Question(type="structure_discovery", phrase="(from the spatial wording)",
                                          evidence="spatial wording without coordinates"))
    asked = {q.type for q in obj.questions}

    features = (mods["matrix"] or [a.name for a in profile.artifacts if a.kind == "embedding"]
                or mods["images"] or mods["text"])
    for q in obj.questions:
        if q.type in ("structure_discovery", "representation_learning", "generative_modelling"):
            q.targets["features"] = features[:1]
        if q.type == "spatial_structure":
            q.targets["coords"] = mods["coords"]
            q.targets["features"] = features[:1]
        if q.type == "differential_analysis":
            q.targets["features"] = (_named_first(mods["matrix"], q.phrase) or features)[:1]
            q.targets["groups"] = (["discovered"] if "structure_discovery" in asked
                                   else [a.name for a in named_labels] or [a.name for a in labels][:1])
        if q.type == "batch_assessment":
            batch = named_groups or [g for g in groups if g.detail.get("role") == "library"] or groups
            q.targets["batch"] = [g.name for g in batch][:1]
            if batch:
                q.targets["outer"] = list(batch[0].detail.get("nested_in", []))[:1]
        if q.type == "cross_modal_association":
            here = {m for m, words in MODALITY_WORDS.items() if _tokens(q.phrase) & words and mods[m]} or mentioned
            others = [m for m in ("images", "text", "matrix") if m in here and mods[m]]
            q.targets["modalities"] = [a for m in (others or ["images", "text"]) for a in mods[m]]
            if "structure_discovery" in asked:
                q.targets["structure"] = ["discovered"]
        if q.type == "predictive_modelling":
            src = re.search(r"\bfrom (?:the )?([a-z\-]+)", q.phrase.lower())
            src_mod = next((m for m, w in MODALITY_WORDS.items() if src and _singular(src.group(1)) in w), None)
            q.targets["features"] = (mods[src_mod] if src_mod and mods[src_mod] else features)[:1]
            target = named_labels[:1] or labels[:1]
            q.targets["labels"] = [a.name for a in target]
            pairs = {(x.first, x.second) for x in profile.label_agreement}
            q.targets["reference"] = [b for a in q.targets["labels"] for (f, b) in pairs if f == a] + \
                                     [f for a in q.targets["labels"] for (f, b) in pairs if b == a]
            if re.search(r"\breliab", low):
                obj.constraints["reliability_rule"] = (
                    f">= {criteria.reliable_min_test_positives} positive evaluation samples, lower "
                    f"{criteria.ci_level:.0%} DeLong bound of AUROC >= {criteria.reliable_min_auroc_lower:g}, "
                    f"BH-adjusted p < {criteria.fdr:g} against AUROC 0.5; fewer positives: too rare to assess")
        if q.type == "data_quality_audit":
            q.challenges = [c for c, pats in CHALLENGE_CUES.items() if any(re.search(p, low) for p in pats)]
            q.targets["images"] = mods["images"]
            q.targets["text"] = mods["text"]
            q.targets["labels"] = [a.name for a in (named_labels or labels)]
    if "visualization" not in asked and features:
        obj.questions.append(Question(type="visualization", phrase="(maps for the report)", implied=True,
                                      targets={"features": features[:1]}, evidence="every report shows maps"))
    if not [q for q in obj.questions if not q.implied]:
        notes.append("no question type recognised in the objective; the plan covers data quality and structure")
        obj.questions.insert(0, Question(type="data_quality_audit", phrase="(default)", implied=True,
                                         targets={"images": mods["images"], "text": mods["text"]}))
        if features:
            obj.questions.insert(1, Question(type="structure_discovery", phrase="(default)", implied=True,
                                             targets={"features": features[:1]}))
    obj.notes = notes
    return obj


def objective_decisions(obj: Objective) -> list[Decision]:
    """One decision-log entry per question read from the objective (and per constraint)."""
    out = []
    for q in obj.questions:
        targets = "; ".join(f"{role}: {', '.join(v)}" for role, v in q.targets.items() if v)
        out.append(Decision(step="objective", decision=f"question: {q.type}",
                            chosen=q.phrase if not q.implied else "implied",
                            rationale=(f"cue '{q.evidence}'" if not q.implied else q.evidence)
                            + (f"; targets {targets}" if targets else "")
                            + (f"; named challenges {', '.join(q.challenges)}" if q.challenges else ""),
                            evidence={"targets": q.targets, "challenges": q.challenges}, source="heuristic"))
    for name, value in obj.constraints.items():
        out.append(Decision(step="objective", decision=f"constraint: {name}", chosen=str(value),
                            rationale="fixed before any result is seen", source="heuristic"))
    return out

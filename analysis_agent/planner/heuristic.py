"""The rule-based planner: recipes per question type, chained through artifact kinds and
adjusted by the challenge register.

The plan is built in a fixed order - prepare each kind of data the questions use, handle
batch structure, answer each question, audit what the objective asks to audit, map the
result - and every rule reads only the objective's question types and targets, the
profile's artifacts and groupings, and the register's challenges. Challenges insert steps
(QC when samples fail the quality rule, count splitting when groups found in counts are
then tested on them, batch correction when a grouping separates the data) and set
parameters (the correction scope, the replicate unit, the cross-validation groups). A last
pass checks that every relevant high-severity challenge is handled by a step whose tool
addresses it and whose inputs derive from the challenge's data, adds a safe mitigation
when one is missing, and otherwise records it as an accepted limitation with a reason.
Every choice is logged with its rationale and the alternatives rejected.
"""

from __future__ import annotations

from ..challenges import SEVERITY_ORDER
from ..schema import (AcceptedChallenge, Alternative, AnalysisPlan, Challenge, ChallengeRegister, DataProfile,
                      Decision, Objective, PlanStep)
from ..tools import registry
from .refs import Trace

# Mitigations the last pass may add on its own: audits whose inputs are unambiguous by kind.
SAFE_MITIGATIONS = ("qc_counts", "qc_images", "qc_text", "label_imbalance", "near_duplicates")


class _Planner:
    def __init__(self, profile: DataProfile, register: ChallengeRegister, objective: Objective):
        self.p, self.reg, self.obj = profile, register, objective
        self.plan = AnalysisPlan(objective=objective.text, questions=objective.types())
        self.trace = Trace(self.plan, profile)
        self.decisions: list[Decision] = []
        self.asked = set(objective.types())
        self.arts = {a.name: a for a in profile.artifacts}
        self.groups = {g.artifact: g for g in profile.groups}
        self.unit = min(profile.groups, key=lambda g: g.n_levels).artifact if profile.groups else None
        self.library = next((g.artifact for g in profile.groups if g.role == "library"), None)
        self.split = profile.split.artifact if profile.split else None
        self.single_labels = [a.name for a in profile.artifacts if a.kind == "labels" and a.detail.get("external")]

    # --- helpers ---------------------------------------------------------------
    def of(self, kind: str) -> list[str]:
        return [a.name for a in self.p.artifacts if a.kind == kind]

    def col(self, artifact: str | None) -> str:
        if artifact in self.groups:
            return self.groups[artifact].column
        a = self.arts.get(artifact or "")
        return str(a.detail.get("column")) if a is not None and a.detail.get("column") else str(artifact)

    def target(self, qtype: str, role: str) -> list[str]:
        q = self.obj.get(qtype)
        return list(q.targets.get(role, [])) if q is not None else []

    def challenges(self, kind: str, subject: str | None = None) -> list[Challenge]:
        return [c for c in self.reg.of_type(kind)
                if c.status == "detected" and (subject is None or subject in c.subject)]

    def worst(self, kind: str, subject: str | None = None) -> str:
        found = self.challenges(kind, subject)
        return max((c.severity for c in found), key=SEVERITY_ORDER.get) if found else "none"

    def add(self, sid: str, tool: str, inputs: dict, params: dict | None = None, purpose: str = "",
            answers=(), rejected=(), validation_only: bool = False) -> dict[str, str]:
        spec = registry.get(tool)
        base, k = sid, 2
        while self.plan.step(sid) is not None:
            sid, k = f"{base}{k}", k + 1
        step = PlanStep(id=sid, tool=tool, inputs={p: v for p, v in inputs.items() if v is not None},
                        params=params or {}, purpose=purpose, answers=[a for a in answers if a],
                        alternatives_rejected=[Alternative(option=o, reason=r) for o, r in rejected],
                        criteria=[c.metric for c in spec.criteria], validation_only=validation_only)
        self.plan.steps.append(step)
        self.trace.add_step(step)
        return {port.name: f"@{sid}.{port.name}" for port in spec.produces}

    def decide(self, decision: str, chosen: str, rationale: str, rejected=(), evidence: dict | None = None):
        self.decisions.append(Decision(step="plan", decision=decision, chosen=chosen, rationale=rationale,
                                       alternatives_rejected=[Alternative(option=o, reason=r) for o, r in rejected],
                                       evidence=evidence or {}, source="heuristic"))

    def uses(self, kind: str) -> bool:
        """Whether any question targets data of this kind (or, for matrices, needs them)."""
        names = set(self.of(kind))
        if not names:
            return False
        for q in self.obj.questions:
            for refs in q.targets.values():
                if names & set(refs):
                    return True
        return False

    # --- preparation ----------------------------------------------------------------
    def prepare_counts(self, counts: str) -> dict[str, str]:
        mat: dict[str, str] = {"source": counts}
        lib, unit = self.library, self.unit
        quality = self.worst("low_quality_samples", counts)
        empty = next((m.n_empty for m in self.p.matrices if m.artifact == counts), 0)
        cur = counts
        if quality != "none" or empty or "data_quality_audit" in self.asked:
            out = self.add("qc", "qc_counts", {"counts": counts, "library": lib, "report_by": unit},
                           purpose="Remove empty and failing samples (MAD rule"
                                   + (f", per {self.col(lib)}" if lib else "") + ") and near-empty features before "
                                   "normalising.",
                           answers=["data_quality_audit"] if "data_quality_audit" in self.asked else [],
                           rejected=[("keep every sample", "failing samples distort normalisation and clustering")])
            cur = out["counts"]
        else:
            self.decide("count QC", "skipped", "No sample fails the default quality rule and none is empty.")
        mat["counts"] = mat["fit"] = mat["test"] = cur
        discover = bool(self.asked & {"structure_discovery", "spatial_structure"})
        if discover and "differential_analysis" in self.asked and self.target("differential_analysis", "groups") \
                == ["discovered"]:
            out = self.add("split", "count_split", {"counts": cur},
                           purpose="Split every count into a half that finds groups and a half that tests them, so "
                                   "markers are not tested on the data that defined the groups.",
                           answers=["differential_analysis"],
                           rejected=[("test markers on the same counts", "double dipping: groups found in the data "
                                                                         "always look different in the same data"),
                                     ("hold out samples instead", "the groups would have to be predicted on new "
                                                                  "samples first, and neighbours leak in space")])
            mat["fit"], mat["test"] = out["fit"], out["test"]
            self.decide("count splitting", "yes", "The objective asks to discover groups and to test markers between "
                        "them; count splitting (data thinning) makes the two independent.",
                        [("no splitting", "markers would be tested on the data that defined the groups")])
        batch = self.library if self.worst("batch_effect", self.library) != "none" else None
        out = self.add("norm", "normalize_counts", {"counts": mat["fit"], "batch": batch},
                       purpose="Normalise depth (log or Pearson residuals, chosen by how much depth signal remains) "
                               "and keep variable features"
                               + (f" chosen within each {self.col(batch)}" if batch else "") + ".",
                       answers=["representation_learning"] if "representation_learning" in self.asked else [],
                       rejected=[("raw counts", "library sizes differ, so depth would dominate every comparison")]
                       + ([("pooled feature selection", f"features differing only between {self.col(batch)}s would "
                                                         "be picked")] if batch else []))
        mat["normalized"] = out["normalized"]
        out = self.add("pca", "dimreduce_pca", {"matrix": mat["normalized"]},
                       purpose="Reduce the normalised features to the principal components that exceed their "
                               "permutation null (Project 1 sub-agent).",
                       rejected=[("cluster the full feature matrix",
                                  "noise in thousands of features swamps structure")])
        mat["embedding"] = mat["representation"] = out["embedding"]
        return mat

    def handle_batches(self, mat: dict[str, str]) -> None:
        asked = "batch_assessment" in self.asked
        batch = (self.target("batch_assessment", "batch") or ([self.library] if self.library else [])
                 or [g for g in self.groups if self.worst("batch_effect", g) != "none"][:1])
        batch = batch[0] if batch else None
        if batch is None:
            return
        outer = (self.groups[batch].nested_in or [None])[0]
        label = self.single_labels[0] if self.single_labels else None
        levels = [self.worst("batch_effect", batch)] + ([self.worst("batch_effect", outer)] if outer else [])
        strongest = max(levels, key=SEVERITY_ORDER.get)
        if asked or strongest != "none":
            self.add("batch_assess", "batch_assess",
                     {"embedding": mat["embedding"], "batch": batch, "outer": outer, "labels": label},
                     purpose=f"Measure how strongly {self.col(batch)}s separate the samples"
                             + (f", within each {self.col(outer)}" if outer else "")
                             + (f", and how {self.col(batch)}s coincide with '{self.col(label)}' (report only)"
                                if label else "") + ".",
                     answers=["batch_assessment"] if asked else [],
                     rejected=[("judge batches across " + self.col(outer) + "s",
                                f"{self.col(outer)} differences are biology as well as technique")] if outer else [])
        if SEVERITY_ORDER[strongest] < SEVERITY_ORDER["medium"]:
            self.decide("batch correction", "none", f"{self.col(batch)} structure is at most "
                        f"{strongest}; correcting would only risk removing biology.")
            return
        candidate = None
        if self.of("counts") and mat.get("fit"):
            out = self.add("batch_vae", "generative_batch_vae", {"counts": mat["fit"], "batch": batch,
                                                                 "features": mat["normalized"]},
                           purpose=f"A count VAE conditioned on {self.col(batch)} (Project 2 sub-agent) as a third "
                                   "correction candidate next to Harmony and ComBat.",
                           rejected=[("only linear corrections", "a conditioned count model can remove non-linear "
                                                                 "batch effects")])
            candidate = out["embedding"]
        params: dict = {}
        unit, scope_rej = self.unit, []
        replicate_level = (outer is not None and outer == unit and "differential_analysis" in self.asked
                           and self.target("differential_analysis", "groups") == ["discovered"]
                           and self.worst("batch_effect", outer) == "high")
        if replicate_level:
            params["within_outer"] = "no"
            ev = {e.metric: e.value for c in self.challenges("batch_effect", outer) for e in c.evidence}
            self.decide("batch correction scope", f"across all {self.col(batch)}s (also removing "
                        f"{self.col(outer)} differences)",
                        f"Markers are tested with the {self.col(outer)} as the replicate, so discovered groups must "
                        f"recur across {self.col(outer)}s; {self.col(outer)} differences dominate the data "
                        f"(high in the register) and a correction inside each {self.col(outer)} would keep them, "
                        f"leaving groups confined to single {self.col(outer)}s. {self.col(batch).capitalize()} effects "
                        f"within {self.col(outer)}s are still measured by the batch assessment.",
                        [(f"correct within each {self.col(outer)}", f"keeps {self.col(outer)} differences; groups "
                          f"could be confined to single {self.col(outer)}s and markers untestable")], ev)
            scope_rej = [(f"correct within each {self.col(outer)}", "groups must recur across the replicate unit")]
        out = self.add("batch_correct", "batch_correct",
                       {"embedding": mat["embedding"], "batch": batch, "outer": outer, "candidate": candidate,
                        "labels": label}, params=params,
                       purpose=f"Compare Harmony, ComBat{', the count VAE' if candidate else ''} and no correction "
                               f"for {self.col(batch)} effects; keep the best mixing that preserves within-"
                               f"{self.col(batch)} neighbourhoods (label-free).",
                       answers=["batch_assessment"] if asked else [],
                       rejected=[("correct without checking neighbourhoods", "over-correction removes biology")]
                       + scope_rej)
        mat["representation"] = out["embedding"]

    def prepare_images(self, images: str, text: str | None, mat: dict | None) -> dict[str, str]:
        out: dict[str, str] = {"images": images, "text": text}
        audit = "data_quality_audit" in self.asked
        if audit or self.worst("low_quality_samples", images) != "none":
            o = self.add("image_qc", "qc_images", {"images": images, "split": self.split},
                         purpose="Flag and remove unreadable, tiny, strip-shaped and blank images; the pass also "
                                 "hashes every image for the duplicate audit.",
                         answers=["data_quality_audit"] if audit else [])
            out["images"] = o["images"]
        if text and (audit or self.worst("text_noise", text) != "none"):
            o = self.add("text_qc", "qc_text", {"text": text, "split": self.split},
                         purpose="Remove empty and very short texts; flag repeated, boilerplate and panel-referencing "
                                 "ones.", answers=["data_quality_audit"] if audit else [])
            out["text"] = o["text"]
        img_lib = self.arts[images].detail.get("library")
        partner = mat["embedding"] if mat else None
        o = self.add("features", "multimodal_features",
                     {"images": out["images"], "text": out["text"], "partner": partner if not out["text"] else None,
                      "library": img_lib if img_lib in self.groups else None,
                      "holdout": self.unit if img_lib in self.groups else None, "split": self.split},
                     purpose="Screen pretrained encoders on held-out pairs and extract image"
                             + (" and text" if out["text"] else "") + " features (Project 3 sub-agent; colour "
                             "normalisation decided from the measured colour differences).",
                     answers=["representation_learning"] if "representation_learning" in self.asked else [],
                     rejected=[("one fixed encoder", "the best encoder depends on the image domain")])
        out["image_features"] = o["image_features"]
        out["text_features"] = o["text_features"] if out["text"] else None
        return out

    # --- questions ---------------------------------------------------------------------
    def answer(self, mat, media) -> None:
        coords = (self.target("spatial_structure", "coords") or self.of("coords"))[:1]
        coords = coords[0] if coords else None
        structure = None
        rep = mat["representation"] if mat else (media or {}).get("image_features")
        if self.asked & {"structure_discovery", "spatial_structure"} and rep:
            if coords:
                out = self.add("domains", "spatial_domains",
                               {"embedding": rep, "coords": coords, "library": self.library, "groups": self.unit},
                               purpose="Find spatial domains: features joined with their neighbourhood mean "
                                       "(BANKSY-style), number of domains chosen by subsample stability and "
                                       "separation.",
                               answers=["structure_discovery", "spatial_structure"],
                               rejected=[("cluster own features only", "ignores that neighbouring samples share a "
                                                                       "domain"),
                                         ("a fixed number of domains", "the data, not a preset, decide how fine")])
            else:
                out = self.add("clusters", "cluster", {"embedding": rep, "groups": self.unit},
                               purpose="Find groups of samples, the number chosen by subsample stability and "
                                       "separation.", answers=["structure_discovery"],
                               rejected=[("a fixed number of clusters", "the data decide how fine")])
            structure = out["labels"]
            for lab in self.single_labels:
                self.add("validate_" + self.col(lab), "label_association",
                         {"found": structure, "reference": lab, "groups": self.unit},
                         purpose=f"Compare the discovered groups with '{self.col(lab)}' (validation only: never "
                                 "used to choose).",
                         answers=["cross_modal_association"] if "cross_modal_association" in self.asked else
                         ["structure_discovery"], validation_only=True)
        if "spatial_structure" in self.asked and coords and mat:
            self.add("svg", "spatial_svg", {"matrix": mat["counts"], "coords": coords, "library": self.library},
                     purpose="Find spatially variable features (Moran's I within libraries, BH FDR) on all counts: "
                             "the test does not use the discovered groups, so it needs no held-out half.",
                     answers=["spatial_structure"])
            if structure:
                self.add("enrichment", "spatial_enrichment",
                         {"labels": structure, "coords": coords, "library": self.library},
                         purpose="Test which domains border each other more or less often than chance "
                                 "(label permutations within libraries).", answers=["spatial_structure"])
        if "cross_modal_association" in self.asked:
            mods = self.target("cross_modal_association", "modalities")
            if structure and media and media.get("image_features") and set(mods) & set(self.of("images")):
                self.add("histology", "multimodal_predictability",
                         {"target": structure, "groups": self.unit, "image": media["image_features"],
                          "other": mat["representation"] if mat else None},
                         purpose=f"How well images predict the discovered groups, with {self.col(self.unit)}-"
                                 "grouped cross-validation, against the other modality and both (Project 3 "
                                 "sub-agent).", answers=["cross_modal_association"])
            if media and media.get("text_features") and set(mods) & set(self.of("text")):
                self.add("image_text", "multimodal_agreement",
                         {"image": media["image_features"], "text": media["text_features"], "split": self.split},
                         purpose="How well images and texts match each other, and which pairs do not.",
                         answers=["cross_modal_association"])
        if "differential_analysis" in self.asked and mat and self.of("counts"):
            groups = structure if self.target("differential_analysis", "groups") == ["discovered"] else \
                (self.target("differential_analysis", "groups") or self.single_labels or [None])[0]
            if groups and self.unit:
                self.add("markers", "differential_expression",
                         {"counts": mat["test"], "labels": groups, "replicate": self.unit},
                         purpose=f"Marker features per group: {self.col(self.unit)} pseudobulk, paired moderated "
                                 "t-test, BH FDR"
                                 + (", on the held-out half of the counts" if mat["test"] != mat["fit"] else "")
                                 + "; the naive per-sample test is reported for comparison.",
                         answers=["differential_analysis"],
                         rejected=[("per-sample tests", f"samples of one {self.col(self.unit)} are not independent")])
                self.decide("test unit for markers", self.col(self.unit),
                            f"Samples are nested in {self.col(self.unit)}s ({self.groups[self.unit].n_levels} "
                            "levels); the independent unit is the coarsest grouping.",
                            [("the sample", "pseudoreplication")])
        if "predictive_modelling" in self.asked:
            self.predict(mat, media)

    def predict(self, mat, media) -> None:
        feat_target = self.target("predictive_modelling", "features")
        if media and feat_target and set(feat_target) & set(self.of("images")):
            features = media.get("image_features")
        else:
            features = mat["representation"] if mat else (media or {}).get("image_features")
        labels = self.target("predictive_modelling", "labels")
        if not features or not labels:
            return
        leakage = (media or {}).get("duplicates")
        groups = self.unit
        ref = self.target("predictive_modelling", "reference")
        self.decide("prediction target and features", f"'{self.col(labels[0])}' from "
                    f"{'images' if media and features == media.get('image_features') else 'the main representation'}",
                    "Named by the objective" + (f"; '{self.col(ref[0])}' defines some of the same labels and serves as "
                                                "the reference for label noise" if ref else "") + ".",
                    [(f"predict '{self.col(r)}'", "a reference source, used to estimate noise") for r in ref])
        if groups:
            self.decide("cross-validation groups", self.col(groups) + (" + near-duplicate clusters" if leakage else ""),
                        f"Samples share {self.col(groups)}s ({self.groups[groups].shared_fraction:.0%} of them)"
                        + ("; near-copies link samples too" if leakage else "") + ", so folds and the test "
                        "separation keep linked samples together.", [("random folds", "related samples leak")])
        source = "image features" if media and features == media.get("image_features") else "the representation"
        self.add("predict", "predict_targets",
                 {"features": features, "target": labels[0], "split": self.split, "groups": groups,
                  "leakage": leakage},
                 purpose=f"Predict '{self.col(labels[0])}' from {source}"
                         + (f"; folds and the test separation keep {self.col(groups)}s" if groups else "")
                         + (" and near-duplicate clusters" if leakage else "") + " together; the reliability "
                         "rule fixed in the configuration decides which targets are reliably predicted.",
                 answers=["predictive_modelling"],
                 rejected=[("random folds", "related samples on both sides of a split inflate the scores")]
                 + ([("fit on every non-test sample", "group-mates of test samples would leak")]
                    if self.worst("split_leakage") != "none" else []))
        if self.obj.constraints.get("reliability_rule"):
            self.decide("reliably predicted", "configured rule", str(self.obj.constraints["reliability_rule"]),
                        [("rank targets by AUROC", "a ranking does not say which are reliable")])

    def audit(self, mat, media) -> None:
        if "data_quality_audit" not in self.asked:
            return
        q = self.obj.get("data_quality_audit")
        named = set(q.challenges) if q else set()
        relevant = {c.type for c in self.reg.detected("low") if "data_quality_audit" in c.affects} - {"compute"}
        want = named | relevant
        labels = self.target("data_quality_audit", "labels") or self.target("predictive_modelling", "labels")
        if "class_imbalance" in want:
            for lab in labels:
                self.add("imbalance_" + self.col(lab), "label_imbalance", {"target": lab, "split": self.split},
                         purpose=f"Frequency audit of '{self.col(lab)}': imbalance, long tail and how many labels "
                                 "have the support to be judged.", answers=["data_quality_audit"])
        if media and media.get("images") and "near_duplicates" in want:
            out = self.add("duplicates", "near_duplicates",
                           {"images": media["images"], "embedding": media.get("image_features"),
                            "split": self.split, "groups": self.unit},
                           purpose="Find near-duplicate images (confirmed difference hashes and mutual nearest "
                                   "neighbours in feature space); clusters crossing splits are leakage.",
                           answers=["data_quality_audit"])
            media["duplicates"] = out["clusters"]
        if "label_noise" in want or "text_noise" in want:
            pairs = [(a.first, a.second) for a in self.p.label_agreement]
            for first, second in pairs:
                self.add("label_sources", "label_agreement", {"labels": first, "reference": second},
                         purpose=f"Agreement of '{self.col(first)}' with '{self.col(second)}': an estimate of label "
                                 "noise where both define a label.", answers=["data_quality_audit"])
            if media and media.get("text") and labels:
                ref = next((b for a, b in pairs if a == labels[0]), None)
                self.add("caption_labels", "caption_label_consistency",
                         {"text": media["text"], "labels": labels[0], "split": self.split, "reference": ref},
                         purpose="Flag labels their text does not support, negates, or names without the label "
                                 "(lexicons learned on training texts).", answers=["data_quality_audit"])
            if media and media.get("text_features") and media.get("image_features") \
                    and not any(s.tool == "multimodal_agreement" for s in self.plan.steps):
                self.add("image_text", "multimodal_agreement",
                         {"image": media["image_features"], "text": media["text_features"], "split": self.split},
                         purpose="Flag image-text pairs that match each other worse than random partners do "
                                 "(noisy or mismatched texts).", answers=["data_quality_audit"])
        self.decide("data-quality audits", ", ".join(sorted(want)) or "none",
                    "The challenges the objective names" + (f" ({', '.join(sorted(named))})" if named else "")
                    + " and every relevant challenge the profile detected.", [])
        if media and media.get("image_features"):
            self.add("atypical", "generative_density", {"features": media["image_features"], "split": self.split},
                     purpose="Score how atypical every image is under a density model of the features "
                             "(Project 2 sub-agent), as a quality signal next to the QC flags.",
                     answers=["data_quality_audit"])

    def visualise(self, mat, media) -> None:
        rep = (mat or {}).get("representation") or (media or {}).get("image_features")
        if rep:
            self.add("map", "dimreduce_map", {"embedding": rep},
                     purpose="A two-dimensional map of the main representation for the report (Project 1 "
                             "sub-agent: method chosen by trustworthiness and stability).",
                     answers=["visualization"])

    # --- challenges ----------------------------------------------------------------------
    def annotate(self) -> None:
        detected = [c for c in self.reg.entries if c.status == "detected"]
        for step in self.plan.steps:
            spec = registry.get(step.tool)
            step.addresses = [c.id for c in detected if c.type in spec.addresses and self.trace.handles(step, c)]

    def cover(self) -> None:
        for c in self.reg.detected("high", relevant_only=True):
            if any(c.id in s.addresses for s in self.plan.steps):
                continue
            added = False
            for tool in c.mitigations:
                if tool not in SAFE_MITIGATIONS:
                    continue
                spec = registry.get(tool)
                inputs = {}
                for port in spec.consumes:
                    pick = next((s for s in c.subject if self.trace.kind.get(s) in port.kinds), None)
                    if pick is None and not port.required:
                        continue
                    pick = pick or next((n for n in self.trace.kind if self.trace.kind[n] in port.kinds), None)
                    if pick is None:
                        break
                    inputs[port.name] = pick
                else:
                    self.add(f"mitigate_{c.type}", tool, inputs, purpose=f"Handle {c.type.replace('_', ' ')}: "
                             f"{c.summary}")
                    added = True
                    break
            self.annotate()
            if not added and not any(c.id in s.addresses for s in self.plan.steps):
                self.plan.accepted.append(AcceptedChallenge(
                    challenge=c.id, reason=f"No step of this plan uses {', '.join(c.subject) or 'these data'} in a "
                                           f"way a tool for {c.type.replace('_', ' ')} applies to; reported as a "
                                           "limitation."))

    # --- whole plan ---------------------------------------------------------------------------
    def summarise(self) -> None:
        handled = {c.id: [s.id for s in self.plan.steps if c.id in s.addresses]
                   for c in self.reg.detected("low", relevant_only=True)}
        high = [c.id for c in self.reg.detected("high", relevant_only=True)]
        accepted = {a.challenge for a in self.plan.accepted}
        for c in self.reg.detected("low", relevant_only=True):
            if handled.get(c.id) or c.id in accepted:
                continue
            how = ("checked against the budget by the plan validator's cost estimate" if c.type == "compute"
                   else "no step of this plan uses its data in a way a tool for it applies to")
            self.plan.notes.append(f"{c.id} ({c.severity}): {how}")
        self.decide("challenge handling", f"{sum(bool(v) for v in handled.values())} of {len(handled)} relevant "
                    f"challenges handled by steps; {len(self.plan.accepted)} accepted",
                    "Each relevant challenge is matched to the steps whose tool addresses its type and whose inputs "
                    f"derive from its data; every high-severity one ({len(high)}) must be handled or accepted.",
                    [], {"handled_by": {k: v for k, v in handled.items()}})

    def build(self) -> AnalysisPlan:
        counts = self.of("counts")
        mat = None
        if counts and (self.uses("counts") or self.asked & {"batch_assessment", "differential_analysis"}):
            mat = self.prepare_counts(counts[0])
            self.handle_batches(mat)
        images, texts = self.of("images"), self.of("text")
        need_images = images and (self.uses("images") or (self.asked & {"data_quality_audit"} and images))
        need_text = texts and (self.uses("text") or (self.asked & {"data_quality_audit"} and texts))
        media = self.prepare_images(images[0], texts[0] if need_text else None, mat) if need_images else None
        if not need_images and images:
            self.decide("images", "not used", "No question of the objective uses the images.")
        if media:                                     # audits first: duplicate clusters group the predictions
            self.audit(mat, media)
        self.answer(mat, media)
        if not media:
            self.audit(mat, media)
        if "visualization" in self.asked:
            self.visualise(mat, media)
        self.annotate()
        self.cover()
        self.summarise()
        for q in self.obj.questions:
            if not q.implied and not any(q.type in s.answers for s in self.plan.steps):
                self.plan.notes.append(f"no step answers '{q.type}' on these data")
        return self.plan


def heuristic_plan(profile: DataProfile, register: ChallengeRegister,
                   objective: Objective) -> tuple[AnalysisPlan, list[Decision]]:
    """The rule-based plan for an objective on profiled data, with the decisions behind it."""
    planner = _Planner(profile, register, objective)
    plan = planner.build()
    return plan, planner.decisions

"""Objective interpretation, the rule-based planner and the plan guardrails, on the synthetic
planted / clean datasets of conftest.py: the challenges, not the dataset, decide the steps."""

import pytest

from analysis_agent.challenges import assess
from analysis_agent.config import load_config
from analysis_agent.objective import interpret, objective_decisions
from analysis_agent.planner import plan_analysis, render_plan_markdown, validate
from analysis_agent.schema import AnalysisPlan, PlanStep

TISSUE = ("Characterize spatial tissue heterogeneity: identify spatial domains, relate them to histology, find "
          "marker genes, and assess slide-level batch effects.")
TABLE = ("Determine which clinical concepts can be reliably predicted from images, and identify data-quality "
         "issues such as class imbalance, near-duplicates, and noisy captions.")
BUDGET = load_config(quick=True).budget("cpu")


def _plan(registers, name, planted, text):
    prof = registers[name, planted][0]
    obj = interpret(text, prof)
    reg = assess(prof, obj.types())
    plan, report, decisions = plan_analysis(prof, reg, obj, BUDGET)
    return prof, obj, reg, plan, report, decisions


def _tools(plan):
    return [s.tool for s in plan.steps]


def test_two_matrices_of_one_source(tmp_path):
    from conftest import _cells, _profile

    prof = _profile(_cells(tmp_path), tmp_path / "cache")
    counts = [a.name for a in prof.artifacts if a.kind == "counts"]
    assert counts == ["counts.peaks", "counts.gene_expression"] and not prof.groups
    obj = interpret("Discover populations of cells using both count matrices, find marker genes for each "
                    "population, and assess how well the two modalities agree.", prof)
    assert obj.types(include_implied=False) == ["structure_discovery", "cross_modal_association",
                                                "differential_analysis"]
    assert obj.get("differential_analysis").targets["features"] == ["counts.gene_expression"]   # named in its clause
    reg = assess(prof, obj.types())
    nested = [c for c in reg.of_type("nested_sampling") if c.status == "detected"]
    assert len(nested) == 1 and nested[0].severity == "medium" and not nested[0].subject
    plan, report, decisions = plan_analysis(prof, reg, obj, BUDGET)
    assert report.valid, [i.message for i in report.errors()]
    steps = {s.id: s for s in plan.steps}
    src = steps["split"].inputs["counts"]                                            # QC only if samples fail
    assert "counts.gene_expression" in (src, steps.get(src[1:].split(".")[0], steps["split"]).inputs.get("counts"))
    assert steps["pca_peaks"].inputs["matrix"] == "@norm_peaks.normalized"           # the partner, prepared alone
    assert steps["modality_groups"].inputs == {"found": "@clusters.labels", "reference": "@clusters_peaks.labels"}
    assert steps["modality_pairs"].tool == "multimodal_agreement"
    assert steps["markers"].inputs == {"counts": "@split.test", "labels": "@clusters.labels"}   # no replicate
    assert any(d.decision == "test unit for markers" and "no replicate" in d.chosen for d in decisions)


def test_image_site_effects_without_a_count_matrix(tmp_path):
    from conftest import _profile, _sites

    prof = _profile(_sites(tmp_path), tmp_path / "cache")
    obj = interpret("Determine whether the grade can be predicted from the images, and identify data-quality issues "
                    "such as class imbalance and site effects.", prof)
    assert {"predictive_modelling", "data_quality_audit", "batch_assessment"} <= set(obj.types())
    reg = assess(prof, obj.types())
    assert any(c.status == "detected" and c.severity in ("medium", "high") for c in reg.of_type("batch_effect"))
    plan, report, decisions = plan_analysis(prof, reg, obj, BUDGET)
    assert report.valid, [i.message for i in report.errors()]
    steps = {s.id: s for s in plan.steps}
    assert steps["features"].inputs["library"] == "groups.site"            # colour measured per site
    assert steps["batch_assess"].inputs == {"embedding": "@features.image_features", "batch": "groups.site",
                                            "labels": "labels.grade"}
    assert "batch_assessment" in steps["batch_assess"].answers and steps["predict"].inputs["groups"] == "groups.site"


# ---------------------------------------------------------------------------
# objective
# ---------------------------------------------------------------------------

def test_objective_reads_question_types_and_targets(registers):
    obj = interpret(TISSUE, registers["tissue", True][0])
    assert obj.types(include_implied=False) == ["structure_discovery", "spatial_structure", "cross_modal_association",
                                                "differential_analysis", "batch_assessment"]
    assert obj.get("batch_assessment").targets == {"batch": ["groups.slide"], "outer": ["groups.patient"]}
    assert obj.get("differential_analysis").targets["groups"] == ["discovered"]
    assert obj.get("visualization").implied and obj.get("differential_analysis").phrase == "find marker genes"
    t = interpret(TABLE, registers["table", True][0])
    assert t.types(include_implied=False) == ["predictive_modelling", "data_quality_audit"]
    pred = t.get("predictive_modelling")
    assert pred.targets["features"] == ["images.image_path"] and pred.targets["labels"] == ["multilabel.concepts"]
    assert pred.targets["reference"] == ["labels.concepts_manual"]                   # single-label here
    assert set(t.get("data_quality_audit").challenges) == {"class_imbalance", "near_duplicates", "text_noise",
                                                          "label_noise"}
    assert "reliability_rule" in t.constraints and len(objective_decisions(t)) == len(t.questions) + 1


def test_objective_without_cues_or_coordinates(registers):
    prof = registers["table", False][0]
    vague = interpret("Tell me something about this collection.", prof)
    assert [q.type for q in vague.questions][:2] == ["data_quality_audit", "structure_discovery"]
    assert all(q.implied for q in vague.questions) and vague.notes
    spatial = interpret("Find spatial niches in these images.", prof)
    assert "spatial_structure" not in spatial.types() and "structure_discovery" in spatial.types()
    assert any("no coordinates" in n for n in spatial.notes)


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------

def test_tissue_plan_follows_the_challenges(registers):
    prof, obj, reg, plan, report, decisions = _plan(registers, "tissue", True, TISSUE)
    assert report.valid, [i.message for i in report.errors()]
    tools = _tools(plan)
    for tool in ("qc_counts", "count_split", "normalize_counts", "dimreduce_pca", "batch_assess", "batch_correct",
                 "spatial_domains", "label_association", "spatial_svg", "differential_expression", "dimreduce_map"):
        assert tool in tools, tool
    assert tools.index("count_split") < tools.index("spatial_domains") < tools.index("differential_expression")
    markers = next(s for s in plan.steps if s.tool == "differential_expression")
    assert markers.inputs["counts"] == "@split.test" and markers.inputs["replicate"] == "groups.patient"
    assert next(s for s in plan.steps if s.tool == "label_association").validation_only
    for c in reg.detected("high"):
        assert any(c.id in s.addresses for s in plan.steps), c.id
    assert {d.decision for d in decisions} >= {"count splitting", "test unit for markers", "challenge handling"}
    assert "| 1 | qc | qc_counts |" in render_plan_markdown(plan, report, decisions)


def test_clean_data_get_a_lighter_plan(registers):
    _, _, _, planted, _, _ = _plan(registers, "tissue", True, TISSUE)
    _, _, _, clean, report, decisions = _plan(registers, "tissue", False, TISSUE)
    assert report.valid
    assert "batch_correct" in _tools(planted) and "batch_correct" not in _tools(clean)   # no batch effect
    assert "qc_counts" not in _tools(clean)                                            # no failing samples
    assert any(d.decision == "batch correction" and d.chosen == "none" for d in decisions)
    found = interpret("Find clusters of spots.", registers["tissue", False][0])
    _, _, _, plain, _, _ = _plan(registers, "tissue", False, found.text)
    assert "count_split" not in _tools(plain) and "differential_expression" not in _tools(plain)


def test_table_plan_keeps_related_samples_together(registers):
    _, obj, reg, plan, report, decisions = _plan(registers, "table", True, TABLE)
    assert report.valid, [i.message for i in report.errors()]
    tools = _tools(plan)
    for tool in ("qc_images", "qc_text", "multimodal_features", "label_imbalance", "near_duplicates",
                 "label_agreement", "caption_label_consistency", "multimodal_agreement", "predict_targets"):
        assert tool in tools, tool
    pred = next(s for s in plan.steps if s.tool == "predict_targets")
    assert pred.inputs["features"] == "@features.image_features" and pred.inputs["groups"] == "groups.article"
    assert pred.inputs["leakage"] == "@duplicates.clusters" and tools.index("near_duplicates") < tools.index(
        "predict_targets")
    assert any(c.startswith("split_leakage") for c in pred.addresses)


# ---------------------------------------------------------------------------
# guardrails
# ---------------------------------------------------------------------------

def _broken(registers, edit):
    prof, obj, reg, plan, _, _ = _plan(registers, "tissue", True, TISSUE)
    bad = plan.model_copy(deep=True)
    edit(bad)
    return validate(bad, prof, reg, obj, BUDGET)


def _rules(report, severity="error"):
    return {i.rule for i in report.issues if i.severity == severity and not i.repaired}


def test_guardrails_catch_structural_errors(registers):
    def unknown(p):
        p.steps[0].tool = "magic_tool"

    def unbound(p):
        p.step("pca").inputs.pop("matrix")

    def forward(p):
        p.step("norm").inputs["counts"] = "@markers.table"

    def wrong_kind(p):
        p.step("pca").inputs["matrix"] = "groups.slide"

    for edit, rule in ((unknown, "tools"), (unbound, "inputs"), (forward, "inputs"), (wrong_kind, "inputs")):
        _, report = _broken(registers, edit)
        assert not report.valid and rule in _rules(report), (edit.__name__, report.issues)
    _, report = validate({"objective": "x", "steps": "not a list"}, *_plan(registers, "tissue", True, TISSUE)[:1],
                         _plan(registers, "tissue", True, TISSUE)[2], _plan(registers, "tissue", True, TISSUE)[1],
                         BUDGET)
    assert not report.valid and _rules(report) == {"schema"}


def test_guardrails_repair_parameters_and_grouping(registers):
    def edit(p):
        p.step("domains").params = {"k_max": 500, "not_a_param": 1}
        p.step("markers").inputs["replicate"] = "groups.slide"

    plan, report = _broken(registers, edit)
    assert report.valid
    assert plan.step("domains").params == {"k_max": 60} and plan.step("markers").inputs["replicate"] == "groups.patient"
    assert {i.rule for i in report.issues if i.repaired} >= {"params", "grouping"}


def test_guardrails_keep_reserved_labels_out_of_the_analysis(registers):
    def edit(p):
        p.steps.append(PlanStep(id="peek", tool="multimodal_predictability",
                                inputs={"target": "labels.label", "groups": "groups.patient",
                                        "image": "@pca.embedding"}))
        p.steps.append(p.step("enrichment").model_copy(update={"id": "after", "inputs": {
            **p.step("enrichment").inputs, "labels": "@peek.predicted"}}))

    plan, report = _broken(registers, edit)
    assert plan.step("peek").validation_only                          # repaired: labels reserved for validation
    assert not report.valid and "labels" in _rules(report)            # but nothing may consume its output


def test_guardrails_catch_double_dipping_and_unhandled_challenges(registers):
    def no_split(p):
        p.step("markers").inputs["counts"] = "@qc.counts"

    _, report = _broken(registers, no_split)
    assert not report.valid and "double_dipping" in _rules(report)

    def drop_split(p):
        p.steps = [s for s in p.steps if s.tool not in ("count_split", "differential_expression")]
        for s in p.steps:
            for k, v in list(s.inputs.items()):
                if v.startswith("@split."):
                    s.inputs[k] = "@qc.counts"

    _, report = _broken(registers, drop_split)
    assert not report.valid and "challenges" in _rules(report)

    def accept(p):
        drop_split(p)
        p.accepted.append({"challenge": "double_dipping:counts.expression", "reason": "markers dropped from this plan"})

    _, report = _broken(registers, accept)
    assert "challenges" not in _rules(report)


def test_guardrails_check_the_budget(registers):
    prof, obj, reg, plan, _, _ = _plan(registers, "tissue", True, TISSUE)
    tight = BUDGET.model_copy(update={"step_seconds": 1.0, "total_seconds": 2.0})
    _, report = validate(plan, prof, reg, obj, tight)
    assert not report.valid and "cost" in _rules(report) and report.estimated_seconds > 2.0


def test_plan_schema_round_trips(registers):
    plan = _plan(registers, "table", True, TABLE)[3]
    again = AnalysisPlan.model_validate_json(plan.model_dump_json())
    assert again == plan
    with pytest.raises(Exception):
        AnalysisPlan.model_validate({"objective": "x", "steps": [{"tool": "qc_counts"}]})   # a step needs an id

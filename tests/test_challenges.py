"""The profile and the challenge register on synthetic data: planted challenges are found at the
right severity, and the clean versions of the same designs raise no false alarms (the planted and
clean datasets are built in conftest.py)."""

import json

import pytest

from analysis_agent.challenges import RULES, SEVERITY_ORDER, assess, render_markdown
from analysis_agent.config import REPO_ROOT
from analysis_agent.schema import json_schemas


def _sev(reg, kind, subject=None):
    hits = [c for c in reg.of_type(kind) if subject is None or subject in c.subject]
    assert hits, f"no {kind} entry"
    return max((c.severity for c in hits), key=SEVERITY_ORDER.get)


def _at_least(reg, kind, level, subject=None):
    got = _sev(reg, kind, subject)
    assert SEVERITY_ORDER[got] >= SEVERITY_ORDER[level], f"{kind}: {got} < {level}"


def test_planted_tissue_challenges_are_found(registers):
    prof, reg, _ = registers["tissue", True]
    _at_least(reg, "depth_variation", "high")
    _at_least(reg, "low_quality_samples", "medium", "counts.expression")
    _at_least(reg, "batch_effect", "medium", "groups.slide")
    _at_least(reg, "spatial_autocorrelation", "high")
    _at_least(reg, "nested_sampling", "high")
    assert _sev(reg, "partial_labels") == "medium"                     # half the samples labelled
    assert _sev(reg, "class_imbalance") in ("medium", "high")
    g = next(x for x in prof.group_structure if x.group == "groups.slide")
    assert g.outer == "groups.patient" and g.within_outer_silhouette > RULES["group_silhouette"].medium
    lab = prof.labels[0]
    assert lab.annotated_groups["groups.slide"] == [3, 6] and lab.annotated_fraction == pytest.approx(0.5)


def test_clean_tissue_raises_no_false_alarms(registers):
    prof, reg, _ = registers["tissue", False]
    for kind in ("depth_variation", "low_quality_samples", "batch_effect", "spatial_autocorrelation",
                 "partial_labels", "class_imbalance", "batch_confounding", "high_dimensionality", "missing_data"):
        assert _sev(reg, kind) == "none", (kind, [c.summary for c in reg.of_type(kind)])
    _at_least(reg, "nested_sampling", "high")                          # the design is still nested


def test_planted_table_challenges_are_found(registers):
    prof, reg, _ = registers["table", True]
    _at_least(reg, "near_duplicates", "high")
    _at_least(reg, "split_leakage", "high", "groups.article")
    _at_least(reg, "text_noise", "medium")
    _at_least(reg, "label_noise", "medium")
    _at_least(reg, "class_imbalance", "high", "multilabel.concepts")
    _at_least(reg, "image_heterogeneity", "medium")
    _at_least(reg, "missing_data", "low")
    im = prof.images[0]
    assert im.n_duplicate_pairs >= 40 and im.cross_split_duplicate_pairs >= 10
    assert prof.groups[0].split_overlap["test"] == pytest.approx(0.5)


def test_clean_table_raises_no_false_alarms(registers):
    _, reg, _ = registers["table", False]
    for kind in ("near_duplicates", "split_leakage", "text_noise", "label_noise", "class_imbalance",
                 "image_heterogeneity", "missing_data", "low_quality_samples"):
        assert _sev(reg, kind) == "none", (kind, [c.summary for c in reg.of_type(kind)])


def test_relevance_follows_the_questions(registers):
    prof, _, _ = registers["tissue", True]
    reg = assess(prof, ["predictive_modelling"])
    assert reg.get("double_dipping").status == "not_applicable"      # nothing is discovered then tested
    assert reg.of_type("spatial_autocorrelation")[0].relevant         # it threatens random splits
    assert not reg.of_type("depth_variation")[0].relevant
    both = assess(prof, ["structure_discovery", "differential_analysis"])
    assert both.of_type("double_dipping")[0].severity == "high"
    assert "count_split" in both.of_type("double_dipping")[0].mitigations


def test_every_challenge_type_is_checked_and_the_outputs_are_clean(registers):
    from analysis_agent.taxonomy import CHALLENGE_TYPES

    for prof, reg, d in registers.values():
        assert {c.type for c in reg.entries} == set(CHALLENGE_TYPES)
        assert len({c.id for c in reg.entries}) == len(reg.entries)
        text = prof.model_dump_json() + reg.model_dump_json()
        assert str(d) not in text and "/" + d.name not in text           # no locations reach the profile
        md = render_markdown(reg)
        assert md.count("\n| **") == len(reg.detected("low", relevant_only=False))


def test_schema_files_match_the_models():
    for name, schema in json_schemas().items():
        path = REPO_ROOT / "schemas" / f"{name}.schema.json"
        assert json.loads(path.read_text()) == schema, f"run `bioagent schema --write` ({name})"

"""The vocabulary the agent reasons in: scientific question types and analytical challenges.

Both lists are generic: nothing in them names a dataset, a disease or a technology.
The objective is translated into question types, the data profile into challenges,
and every tool declares which question types it answers and which challenges it
addresses. Planning is then a matter of covering the objective's questions while
handling, or explicitly accepting, every challenge the data pose.
"""

from __future__ import annotations

QUESTION_TYPES: dict[str, str] = {
    "structure_discovery": "find groups of samples (clusters, domains) without using labels",
    "spatial_structure": "how features and groups are arranged in space: domains, spatially "
                         "variable features, neighbourhood relations",
    "cross_modal_association": "how one modality relates to another (e.g. image to expression, "
                               "image to text)",
    "differential_analysis": "which features differ between groups, with error control",
    "batch_assessment": "how strongly technical groups (slides, sites, runs) shape the data, "
                        "and whether and how to correct for them",
    "predictive_modelling": "how well an outcome can be predicted, estimated without leakage",
    "data_quality_audit": "unusable samples, imbalance, duplicates, label and text noise",
    "representation_learning": "a compact numeric representation of the samples for later steps",
    "generative_modelling": "a probability model of the data: densities, imputation, "
                            "conditional (batch-aware) latent spaces",
    "visualization": "two-dimensional maps and figures for inspection",
}

CHALLENGE_TYPES: dict[str, str] = {
    "low_quality_samples": "samples too sparse, empty, unreadable or degenerate to analyse",
    "depth_variation": "technical variation in library size or sequencing depth",
    "high_dimensionality": "many more features than can be modelled directly; sparsity",
    "batch_effect": "technical groups (slides, sites) that separate the data, possibly nested",
    "batch_confounding": "technical groups that coincide with the biology of interest",
    "nested_sampling": "samples nested in groups (spots in slides in patients), so the "
                       "independent unit is the group, not the sample",
    "double_dipping": "testing hypotheses on the same data that generated them",
    "spatial_autocorrelation": "neighbouring samples are not independent",
    "partial_labels": "labels exist for only some samples or groups",
    "class_imbalance": "classes or labels with very different frequencies; a long tail",
    "label_noise": "labels that disagree between sources or with the data",
    "near_duplicates": "repeated or near-identical samples, which can leak across splits",
    "split_leakage": "related samples (one group, or near-copies) on both sides of a predefined split",
    "text_noise": "empty, very short, duplicated or boilerplate text",
    "missing_data": "missing values or missing modalities",
    "image_heterogeneity": "images that differ in size, colour, channels or staining",
    "multiple_testing": "many hypotheses tested at once",
    "compute": "estimated cost close to or above the compute budget",
    "pretraining_overlap": "a pretrained model may have seen the evaluation data",
}

# The questions each challenge threatens: a challenge is relevant to an objective when it
# affects at least one of the objective's question types.
_ALL_ANALYSES = ("structure_discovery", "spatial_structure", "cross_modal_association", "differential_analysis",
                 "batch_assessment", "predictive_modelling", "representation_learning", "generative_modelling")
CHALLENGE_AFFECTS: dict[str, tuple[str, ...]] = {
    "low_quality_samples": _ALL_ANALYSES + ("data_quality_audit", "visualization"),
    "depth_variation": ("structure_discovery", "spatial_structure", "differential_analysis", "batch_assessment",
                        "representation_learning", "visualization"),
    "high_dimensionality": ("structure_discovery", "spatial_structure", "predictive_modelling",
                            "representation_learning", "generative_modelling", "visualization"),
    "batch_effect": ("structure_discovery", "spatial_structure", "cross_modal_association", "differential_analysis",
                     "batch_assessment", "predictive_modelling", "representation_learning"),
    "batch_confounding": ("structure_discovery", "differential_analysis", "batch_assessment", "predictive_modelling"),
    "nested_sampling": ("differential_analysis", "predictive_modelling", "cross_modal_association",
                        "batch_assessment"),
    "double_dipping": ("differential_analysis",),
    "spatial_autocorrelation": ("spatial_structure", "structure_discovery", "differential_analysis",
                                "predictive_modelling"),
    "partial_labels": ("predictive_modelling", "structure_discovery", "cross_modal_association"),
    "class_imbalance": ("predictive_modelling", "data_quality_audit"),
    "label_noise": ("predictive_modelling", "data_quality_audit"),
    "near_duplicates": ("predictive_modelling", "cross_modal_association", "data_quality_audit"),
    "split_leakage": ("predictive_modelling", "cross_modal_association", "data_quality_audit"),
    "text_noise": ("cross_modal_association", "predictive_modelling", "data_quality_audit"),
    "missing_data": _ALL_ANALYSES + ("data_quality_audit",),
    "image_heterogeneity": ("cross_modal_association", "predictive_modelling", "representation_learning",
                            "structure_discovery", "data_quality_audit"),
    "multiple_testing": ("differential_analysis", "spatial_structure", "predictive_modelling"),
    "compute": _ALL_ANALYSES + ("data_quality_audit", "visualization"),
    "pretraining_overlap": ("cross_modal_association", "predictive_modelling"),
}

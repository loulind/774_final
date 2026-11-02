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
    "text_noise": "empty, very short, duplicated or boilerplate text",
    "missing_data": "missing values or missing modalities",
    "image_heterogeneity": "images that differ in size, colour, channels or staining",
    "multiple_testing": "many hypotheses tested at once",
    "compute": "estimated cost close to or above the compute budget",
    "pretraining_overlap": "a pretrained model may have seen the evaluation data",
}

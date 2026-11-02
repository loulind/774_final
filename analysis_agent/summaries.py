"""The few metrics that summarise each tool's step, for the register, the report and the templates."""

from __future__ import annotations

HEADLINE: dict[str, tuple[str, ...]] = {
    "qc_counts": ("n_removed", "fraction_removed", "n_features_dropped", "per_library_thresholds"),
    "qc_images": ("n_removed", "fraction_removed"),
    "qc_text": ("fraction_removed", "fraction_duplicate", "fraction_panel_reference"),
    "count_split": ("method", "median_dependence_binomial", "fit_count_share"),
    "normalize_counts": ("method", "depth_corr_pc1_before", "depth_corr_pc1_after", "n_features"),
    "dimreduce_pca": ("n_components", "variance_explained"),
    "dimreduce_map": ("method", "trustworthiness@10", "continuity@10", "stability"),
    "batch_assess": ("batch_effect", "within_outer_batch_silhouette", "within_outer_mixing", "batch_silhouette",
                     "mixing", "share_between_outer", "share_between_inner_within_outer"),
    "batch_correct": ("method", "within_outer", "mixing", "mixing_uncorrected", "neighbourhood_preservation",
                      "outer_mixing"),
    "generative_batch_vae": ("model", "val_neg_elbo_per_gene", "epochs_run"),
    "generative_density": ("model", "val_loglik_gain", "flagged_fraction"),
    "multimodal_features": ("image_encoder", "text_encoder", "screen_score", "screen_chance", "colour_normalised",
                            "zero_shot_recall_at_10"),
    "multimodal_predictability": ("balanced_accuracy_image", "balanced_accuracy_other", "balanced_accuracy_both",
                                  "chance_balanced_accuracy"),
    "multimodal_agreement": ("aligner", "val_recall_at_10", "flagged_fraction"),
    "spatial_domains": ("method", "k", "stability", "spatial_coherence", "spatial_coherence_own_features",
                        "cramers_v_groups"),
    "cluster": ("method", "k", "stability", "cramers_v_groups"),
    "label_association": ("ari", "nmi", "purity", "ari_group_median"),
    "spatial_svg": ("n_significant", "n_tested", "permuted_p05_fraction"),
    "spatial_enrichment": ("n_enriched", "n_depleted", "self_enrichment_median_z"),
    "differential_expression": ("testable", "n_markers", "n_markers_naive", "naive_inflation", "loo_stable_fraction",
                                "n_replicates_min"),
    "predict_targets": ("model", "n_reliable", "n_assessed", "n_too_rare_to_assess", "median_auroc",
                        "cv_auroc_median", "n_set_aside", "leakage_gap_median"),
    "near_duplicates": ("n_clusters", "fraction_in_clusters", "n_hash_pairs", "n_embedding_pairs",
                        "cross_split_clusters"),
    "label_imbalance": ("n_labels_used", "imbalance_ratio", "gini", "n_assessable"),
    "label_agreement": ("kappa_micro", "n_shared_labels", "precision_micro", "recall_micro"),
    "caption_label_consistency": ("n_labels_checked", "unsupported_rate", "negated_rate", "possible_missing_rate"),
    "generative_impute": ("missing_fraction",),
}


def fmt(v) -> str:
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3g}"
    return str(v)


def headline(tool: str, metrics: dict, n: int | None = None) -> str:
    """'metric value' pairs of a step's key metrics (the ones it reported)."""
    keys = [k for k in HEADLINE.get(tool, ()) if metrics.get(k) is not None]
    if not keys:
        keys = [k for k, v in metrics.items() if isinstance(v, (int, float, str, bool))][:4]
    return ", ".join(f"{k.replace('_', ' ')} {fmt(metrics[k])}" for k in keys[:n])

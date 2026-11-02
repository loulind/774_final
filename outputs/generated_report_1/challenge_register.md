# Challenge register

Relevance judged against: structure_discovery, spatial_structure, cross_modal_association, differential_analysis, batch_assessment, visualization.

| challenge | severity | relevant | evidence | candidate mitigations |
|---|---|---|---|---|
| **batch effect** (groups.patient, counts.expression): Samples separate by patient (silhouette 0.375, neighbour mixing 0.00412, 0.664 of the variance between patients); patient differences mix biology and technical effects. | high | yes | silhouette by patient = 0.375 (low >= 0.02, medium >= 0.05, high >= 0.15); neighbour mixing of patients = 0.00412 (low <= 0.9, medium <= 0.8, high <= 0.5); variance share between patients = 0.664 | batch_assess, batch_correct, generative_batch_vae, multimodal_features |
| **depth variation** (counts.expression): Library sizes in counts.expression span an interquartile range of 1.86 on the log2 scale; per-library medians from 636 to 12,194, and PC1 still tracks depth after library-size scaling (\|rho\| 0.766). | high | yes | interquartile range of log2 library size = 1.86 (low >= 0.5, medium >= 1, high >= 2); \|Spearman rho\| of PC1 with log depth after library-size scaling = 0.766 (low >= 0.1, medium >= 0.3, high >= 0.5); \|rho(PC1, log depth)\| on log1p of raw counts = 0.994 | generative_batch_vae, normalize_counts |
| **double dipping** (counts.expression): Groups found in counts.expression and then tested for marker features on the same counts would be tested on the data that defined them, inflating significance. | high | yes | questions implying discovery then testing = differential_analysis, structure_discovery, spatial_structure | count_split |
| **multiple testing** (dataset): 9,514 features compared between groups; 9,514 features tested for spatial patterns: without error control, false positives are expected among them. | high | yes | hypotheses the questions imply: features compared between groups = 9,514 (low >= 2, medium >= 20, high >= 1000); hypotheses the questions imply: features tested for spatial patterns = 9,514 (low >= 2, medium >= 20, high >= 1000) | differential_expression, predict_targets, spatial_enrichment, spatial_svg |
| **nested sampling** (groups.patient): The 13,620 samples come from 8 patient levels (1,702 per level, with slide nested inside); 100% of samples share their patient with others, so the independent unit is the patient. | high | yes | samples per independent unit (coarsest grouping) = 1,702 (low >= 1.2, medium >= 2, high >= 10); patient levels = 8; share of samples sharing their patient = 1 | batch_assess, differential_expression, multimodal_predictability, predict_targets |
| **spatial autocorrelation** (coords.spatial, counts.expression): 100% of the 100 most variable features of counts.expression are spatially autocorrelated within libraries (median Moran's I 0.134): neighbouring samples are not independent. | high | yes | share of variable features with significant Moran's I = 1 (low >= 0.05, medium >= 0.2, high >= 0.5); median Moran's I = 0.134 | spatial_domains, spatial_svg |
| **batch confounding** (groups.patient, labels.label): Label 'label' is associated with patient (Cramér's V 0.369): a correction or comparison across patients also touches the labelled biology. | medium | yes | Cramér's V between group and label = 0.369 (low >= 0.1, medium >= 0.3, high >= 0.5) | batch_assess |
| **batch effect** (groups.slide, counts.expression): Within each patient, samples separate by slide (silhouette 0.0324, neighbour mixing 0.667); slides inside one patient explain 0.0526 of the variance, patients 0.664. | medium | yes | silhouette by slide within patient = 0.0324 (low >= 0.02, medium >= 0.05, high >= 0.15); neighbour mixing of slides within patient = 0.667 (low <= 0.9, medium <= 0.8, high <= 0.5); variance share between slides of one patient = 0.0526 | batch_assess, batch_correct, generative_batch_vae, multimodal_features |
| **partial labels** (labels.label): 23% of samples carry 'label' (slide: 8 of 36 levels labelled). | medium | yes | share of samples with a label = 0.232 (low <= 0.99, medium <= 0.5, high <= 0.1); slide levels labelled = 8 of 36; patient levels labelled = 8 of 8 | label_association |
| **batch effect** (groups.slide, images.spot_patches): Mean image colour differs between slide levels (between-group share 0.126), as staining or scanning batches do. | low | yes | between-group share of image colour variance = 0.126 (low >= 0.05, medium >= 0.2, high >= 0.5) | batch_assess, batch_correct, generative_batch_vae, multimodal_features |
| **high dimensionality** (counts.expression): counts.expression has 11,871 features for 13,619 samples (0.872 per sample) and 83% zeros; 50 components hold 0.236 of the variance of the most variable features. | low | yes | features per sample (p / n) = 0.872 (low >= 0.1, medium >= 1, high >= 10); share of zero entries = 0.827 (low >= 0.5, medium >= 0.9, high >= 0.99); share of variance in 50 principal components = 0.236 | dimreduce_map, dimreduce_pca, generative_batch_vae, multimodal_features, normalize_counts |
| **low quality samples** (counts.expression): 1.5% of samples in counts.expression fail the default quality rule (3 MADs and 2-fold below the median depth or detected features, within each library); 0 are empty. | low | yes | share of samples failing the default count-QC rule = 0.0148 (low >= 0.005, medium >= 0.03, high >= 0.1); empty samples = 0 | generative_density, qc_counts, qc_images |
| **class imbalance** (labels.label): 'label' has 6 classes, the largest 18.1 times the smallest; 6 have the support to be modelled and judged (0% are too rare to assess). | medium | no | largest / smallest class size = 18.1 (low >= 3, medium >= 10, high >= 50); share of labels too rare to assess = 0 (low >= 0.1, medium >= 0.5, high >= 0.9); labels with support to model and judge = 6 | label_imbalance, multimodal_predictability, predict_targets |

## Checked, not found or not measurable yet

- low quality samples (not detected): 0.0% of the 240 measured images in images.spot_patches are unreadable, tiny, strip-shaped or blank.
- label noise (not measurable): Noise in 'label' is not measurable from the profile: no second label source or text to check it against.
- near duplicates (not applicable): images.spot_patches are cut from larger images, so copies are not expected and not hashed.
- split leakage (not applicable): No predefined split: the agent makes grouped splits itself.
- text noise (not applicable): No text.
- missing data (not detected): 1 of 13,620 samples (0.0%) have no expression.
- image heterogeneity (not detected): In images.spot_patches sizes are uniform (median 145 x 145 px) and 0% of 240 sampled images are greyscale, the rest colour.
- compute (not detected): The most expensive tool on these data, multimodal_features, is estimated at 875 s, 24% of the 3,600 s step budget on mps.
- pretraining overlap (not measurable): Pretrained encoders may have seen these images or texts; this is judged after encoder screening, from zero-shot retrieval far above what the training split supports.

## Thresholds

| rule | measurement | low | medium | high |
|---|---|---|---|---|
| qc_flagged | share of samples failing the default count-QC rule | >= 0.005 | >= 0.03 | >= 0.1 |
| unusable_images | share of unreadable, tiny, strip-shaped or blank images | >= 0.001 | >= 0.01 | >= 0.05 |
| library_iqr | interquartile range of log2 library size | >= 0.5 | >= 1 | >= 2 |
| depth_pc1 | \|Spearman rho\| of PC1 with log depth after library-size scaling | >= 0.1 | >= 0.3 | >= 0.5 |
| features_per_sample | features per sample (p / n) | >= 0.1 | >= 1 | >= 10 |
| sparsity | share of zero entries | >= 0.5 | >= 0.9 | >= 0.99 |
| group_silhouette | silhouette width by group | >= 0.02 | >= 0.05 | >= 0.15 |
| group_mixing | neighbour mixing of groups (1 = as mixed as overall) | <= 0.9 | <= 0.8 | <= 0.5 |
| colour_share | between-group share of image colour variance | >= 0.05 | >= 0.2 | >= 0.5 |
| cramers_v | Cramér's V between group and label | >= 0.1 | >= 0.3 | >= 0.5 |
| samples_per_unit | samples per independent unit (coarsest grouping) | >= 1.2 | >= 2 | >= 10 |
| autocorrelated | share of variable features with significant Moran's I | >= 0.05 | >= 0.2 | >= 0.5 |
| annotated | share of samples with a label | <= 0.99 | <= 0.5 | <= 0.1 |
| class_ratio | largest / smallest class size | >= 3 | >= 10 | >= 50 |
| too_rare | share of labels too rare to assess | >= 0.1 | >= 0.5 | >= 0.9 |
| kappa | Cohen's kappa between label sources | <= 0.95 | <= 0.8 | <= 0.6 |
| duplicates | share of images in a near-duplicate pair | >= 0.005 | >= 0.02 | >= 0.1 |
| group_overlap | share of a held-out split whose group also occurs in another split | >= 0.001 | >= 0.05 | >= 0.2 |
| text_flagged | share of texts empty, very short, repeated, boilerplate or referring to panels | >= 0.01 | >= 0.05 | >= 0.2 |
| missing | share of samples without the modality | >= 0.01 | >= 0.1 | >= 0.3 |
| size_spread | Q3 / Q1 of the shorter image side | >= 1.5 | >= 2 | >= 4 |
| colour_mix | share of the minority colour mode (greyscale vs colour) | >= 0.05 | >= 0.2 | >= 0.4 |
| hypotheses | hypotheses the questions imply | >= 2 | >= 20 | >= 1000 |
| compute_share | largest declared tool cost / step budget | >= 0.25 | >= 0.5 | >= 1 |

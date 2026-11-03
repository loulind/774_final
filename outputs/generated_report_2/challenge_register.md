# Challenge register

Relevance judged against: predictive_modelling, data_quality_audit, visualization.

| challenge | severity | relevant | evidence | candidate mitigations |
|---|---|---|---|---|
| **class imbalance** (multilabel.concepts): 'concepts' has 1,947 labels, the largest 27,798 times the smallest; 221 have the support to be modelled and judged (89% are too rare to assess). | high | yes | largest / smallest class size = 27,798 (low >= 3, medium >= 10, high >= 50); share of labels too rare to assess = 0.886 (low >= 0.1, medium >= 0.5, high >= 0.9); labels with support to model and judge = 221 | label_imbalance, multimodal_predictability, predict_targets |
| **class imbalance** (multilabel.concepts_manual): 'concepts_manual' has 18 labels, the largest 544 times the smallest; 17 have the support to be modelled and judged (6% are too rare to assess). | high | yes | largest / smallest class size = 544 (low >= 3, medium >= 10, high >= 50); share of labels too rare to assess = 0.0556 (low >= 0.1, medium >= 0.5, high >= 0.9); labels with support to model and judge = 17 | label_imbalance, multimodal_predictability, predict_targets |
| **multiple testing** (dataset): 1,721 targets modelled: without error control, false positives are expected among them. | high | yes | hypotheses the questions imply: targets modelled = 1,721 (low >= 2, medium >= 20, high >= 1000) | differential_expression, predict_targets, spatial_enrichment, spatial_svg |
| **split leakage** (groups.article, split.split): 64% of the test split shares its article with another split; 64% of the val split shares its article with another split. | high | yes | share of the test split whose article occurs in another split = 0.639 (low >= 0.001, medium >= 0.05, high >= 0.2); share of the val split whose article occurs in another split = 0.645 (low >= 0.001, medium >= 0.05, high >= 0.2) | near_duplicates, predict_targets |
| **compute** (dataset): The most expensive tool on these data, multimodal_features, is estimated at 3,522 s, 98% of the 3,600 s step budget on mps. | medium | yes | largest declared tool cost / step budget = 0.978 (low >= 0.25, medium >= 0.5, high >= 1); estimated seconds: multimodal_features = 3,522; estimated seconds: generative_density = 742 | planning (subsample, cap) |
| **image heterogeneity** (images.image_path): In images.image_path sizes are uniform (median 294 x 256 px) and 74% of 240 sampled images are greyscale, the rest colour. | medium | yes | Q3 / Q1 of the shorter image side = 1 (low >= 1.5, medium >= 2, high >= 4); share of the minority colour mode (greyscale vs colour) = 0.258 (low >= 0.05, medium >= 0.2, high >= 0.4) | generative_density, multimodal_features, qc_images |
| **near duplicates** (images.image_path): 0.5% of the 79,785 hashed images in images.image_path have a near-copy (217 pairs: 64-bit difference-hash matches confirmed at 256 bits, of 57,627 64-bit matches); 57 pairs cross the predefined split. | medium | yes | share of images in a near-duplicate pair = 0.00452 (low >= 0.005, medium >= 0.02, high >= 0.1); near-duplicate pairs crossing the split = 57; 64-bit hash matches before confirmation = 57,627 | near_duplicates, predict_targets |
| **pretraining overlap** (@image_qc.images, @text_qc.text): Zero-shot retrieval reaches recall@10 0.95 on held-out pairs: the encoder may have been pretrained on these or very similar pairs. | medium | yes | zero-shot recall@10 on held-out pairs = 0.955 (flag >= 0.5) | planning (subsample, cap) |
| **split leakage** (images.image_path, split.split): 57 near-duplicate image pairs lie on different sides of the split. | medium | yes | near-duplicate pairs crossing the split = 57 | near_duplicates, predict_targets |
| **label noise** (multilabel.concepts, multilabel.concepts_manual): multilabel.concepts and multilabel.concepts_manual agree with kappa 0.964 on 18 shared labels (precision 0.937, recall 1); those labels hold 47% of multilabel.concepts's assignments, so the rest is unmeasured. | low | yes | Cohen's kappa between label sources = 0.964 (low <= 0.95, medium <= 0.8, high <= 0.6); share of the first source's assignments on shared labels = 0.468 | caption_label_consistency, label_agreement, multimodal_agreement |
| **nested sampling** (groups.article): The 79,789 samples come from 44,975 article levels (1.77 per level); 68% of samples share their article with others, so the independent unit is the article. | low | yes | samples per independent unit (coarsest grouping) = 1.77 (low >= 1.2, medium >= 2, high >= 10); article levels = 44,975; share of samples sharing their article = 0.676 | batch_assess, differential_expression, multimodal_predictability, predict_targets |
| **text noise** (text.caption): 3.6% of text.caption are flagged (too short 0.9%, duplicate 1.8%, boilerplate 0.5%, panel reference 1.2%); median 17 words. | low | yes | share of texts empty, very short, repeated, boilerplate or referring to panels = 0.0358 (low >= 0.01, medium >= 0.05, high >= 0.2); share empty = 0; share too short = 0.00929 | caption_label_consistency, multimodal_agreement, qc_text |

## Checked, not found or not measurable yet

- low quality samples (not detected): 0.0% of the 79,789 measured images in images.image_path are unreadable, tiny, strip-shaped or blank.
- depth variation (not applicable): No count matrix: sequencing depth does not apply.
- high dimensionality (not applicable): No feature matrix; images and texts are reduced by encoders.
- batch effect (not measurable): Groups are too small to compare image colour; group effects in learned features are measured after feature extraction.
- batch confounding (not measurable): No single-label column with groups large enough to cross-tabulate.
- double dipping (not applicable): No count matrix to find groups in and then test them on.
- spatial autocorrelation (not applicable): No spatial coordinates.
- partial labels (not detected): 100% of samples carry 'concepts'.
- partial labels (not detected): 99% of samples carry 'concepts_manual' (article: 44633 of 44975 levels labelled).
- missing data (not detected): 1 of 79,789 samples (0.0%) have no text.
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

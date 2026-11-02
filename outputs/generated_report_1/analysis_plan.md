# Analysis plan

Objective: Characterize spatial tissue heterogeneity: identify spatial domains, relate them to histology, find marker genes, and assess slide-level batch effects.

Questions: structure_discovery, spatial_structure, cross_modal_association, differential_analysis, batch_assessment, visualization. Source: llm.

| # | step | tool | inputs | parameters | purpose | handles |
|---|---|---|---|---|---|---|
| 1 | qc | qc_counts | counts=counts.expression, library=groups.slide, report_by=groups.patient | defaults | Remove empty and failing samples (MAD rule, per slide) and near-empty features before normalising. | low_quality_samples:counts.expression |
| 2 | split | count_split | counts=@qc.counts | defaults | Split every count into a half that finds groups and a half that tests them, so markers are not tested on the data that defined the groups. | double_dipping:counts.expression |
| 3 | norm | normalize_counts | counts=@split.fit, batch=groups.slide | defaults | Normalise depth on the fit half (log or Pearson residuals, chosen by remaining depth signal) and keep variable features chosen within each slide. | depth_variation:counts.expression, high_dimensionality:counts.expression |
| 4 | pca | dimreduce_pca | matrix=@norm.normalized | defaults | Reduce the normalised features to the principal components that exceed their permutation null. | high_dimensionality:counts.expression |
| 5 | batch_assess | batch_assess | embedding=@pca.embedding, batch=groups.slide, outer=groups.patient, labels=labels.label | defaults | Measure how strongly slides separate the samples, within each patient, and how batches coincide with 'label' (report only). | batch_effect:groups.patient,counts.expression, batch_effect:groups.slide,counts.expression, batch_confounding:groups.patient,labels.label, nested_sampling:groups.patient |
| 6 | batch_vae | generative_batch_vae | counts=@split.fit, batch=groups.slide, features=@norm.normalized | defaults | A count VAE conditioned on slide as a non-linear correction candidate next to Harmony and ComBat. | depth_variation:counts.expression, high_dimensionality:counts.expression, batch_effect:groups.slide,counts.expression |
| 7 | batch_correct | batch_correct | embedding=@pca.embedding, batch=groups.slide, outer=groups.patient, candidate=@batch_vae.embedding, labels=labels.label | within_outer=no | Compare Harmony, ComBat, the count VAE and no correction for slide effects; keep the best mixing that preserves within-slide neighbourhoods (label-free; label silhouette reported only). | batch_effect:groups.patient,counts.expression, batch_effect:groups.slide,counts.expression |
| 8 | features | multimodal_features | images=images.spot_patches, partner=@pca.embedding, library=groups.slide, holdout=groups.patient | defaults | Screen pretrained encoders on patient-held-out pairs and extract histology image features, with stain normalisation decided from the measured colour differences. | batch_effect:groups.slide,images.spot_patches |
| 9 | domains | spatial_domains | embedding=@batch_correct.embedding, coords=coords.spatial, library=groups.slide, groups=groups.patient | defaults | Find spatial domains: features joined with their neighbourhood mean (BANKSY-style), number of domains chosen by subsample stability and spatial coherence. | spatial_autocorrelation:coords.spatial,counts.expression |
| 10 | validate_label | label_association | found=@domains.labels, reference=labels.label, groups=groups.patient | defaults | Compare the discovered domains with the histological annotation 'label' (validation only: never used to choose). | partial_labels:labels.label |
| 11 | norm_all | normalize_counts | counts=@qc.counts, batch=groups.slide | n_features=10000 | Depth-normalise all QC-passed counts, keeping essentially every detected feature, so spatial-variability tests are not driven by spatially varying library size. | depth_variation:counts.expression, high_dimensionality:counts.expression |
| 12 | svg | spatial_svg | matrix=@norm_all.normalized, coords=coords.spatial, library=groups.slide | defaults | Find spatially variable features (Moran's I within slides, BH FDR, permutation calibration) on depth-normalised counts; the test does not use the discovered groups. | spatial_autocorrelation:coords.spatial,counts.expression, multiple_testing |
| 13 | enrichment | spatial_enrichment | labels=@domains.labels, coords=coords.spatial, library=groups.slide | defaults | Test which domains border each other more or less often than chance (label permutations within slides). | multiple_testing |
| 14 | histology | multimodal_predictability | target=@domains.labels, groups=groups.patient, image=@features.image_features, other=@batch_correct.embedding | defaults | How well histology image features alone predict the discovered domains under patient-grouped cross-validation; the expression arm is a ceiling reference only. | nested_sampling:groups.patient, class_imbalance:labels.label |
| 15 | markers | differential_expression | counts=@split.test, labels=@domains.labels, replicate=groups.patient | defaults | Marker genes per domain: patient pseudobulk, paired moderated t-test, BH FDR, on the held-out half of the counts; the naive per-sample test is reported for comparison. | nested_sampling:groups.patient, multiple_testing |
| 16 | map | dimreduce_map | embedding=@batch_correct.embedding | defaults | A two-dimensional map of the corrected representation for the report, method chosen by trustworthiness and stability. | high_dimensionality:counts.expression |

## Planning decisions

- **count splitting**: yes. The objective asks to discover groups and to test markers between them; count splitting (data thinning) makes the two independent. Rejected: no splitting: markers would be tested on the data that defined the groups.
- **batch correction scope**: across all slides (also removing patient differences). Markers are tested with the patient as the replicate, so discovered groups must recur across patients; patient differences dominate the data (high in the register) and a correction inside each patient would keep them, leaving groups confined to single patients. Slide effects within patients are still measured by the batch assessment. Rejected: correct within each patient: keeps patient differences; groups could be confined to single patients and markers untestable.
- **test unit for markers**: patient. Samples are nested in patients (8 levels); the independent unit is the coarsest grouping. Rejected: the sample: pseudoreplication.
- **challenge handling**: 12 of 12 relevant challenges handled by steps; 0 accepted. Each relevant challenge is matched to the steps whose tool addresses its type and whose inputs derive from its data; every high-severity one (6) must be handled or accepted.
- **input to the spatially-variable-gene test**: add norm_all (normalize_counts on @qc.counts, n_features 10000, batch-aware) and run spatial_svg on its output instead of raw counts. depth_variation is high: log2 library-size IQR 1.86, per-slide medians 636 to 12,194, PC1 tracks depth with |rho| 0.994 on raw counts and still 0.766 after scaling. Library size is spatially structured within tissue, so Moran's I on raw counts would mark depth patterns as spatially variable genes across all ~9,514 detected features. n_features 10000 keeps essentially every feature detected in at least 5% of spots, so the multiple-testing family stays the full set. SVG does not use discovered groups, so using all counts creates no double dipping. Rejected: keep spatial_svg on @qc.counts: depth confounding of the spatial statistic; use @norm.normalized: it has only 2,000 features, from half the counts.
- **planner used**: llm. LLM revised the rule-based plan: added step norm_all (normalize_counts); svg.inputs: {'matrix': '@qc.counts', 'coords': 'coords.spatial', 'library': 'groups.slide'} -> {'matrix': '@norm_all.normalized', 'coords': 'coords.spatial', 'library': 'groups.slide'}

## Validation

Valid; estimated 1,825 s in all (longest step 875 s).

## Notes

- In 'histology' the expression arm ('other') is circular because domains were found from that embedding; only the image-alone balanced accuracy against chance is interpreted as the histology association.
- Domains found in only one or two patients cannot be tested with min_replicates 3; such domains are reported as patient-specific rather than forced into marker tests.
- Declared costs (largest: multimodal_features ~875 s, generative_batch_vae ~270 s, dimreduce_map ~200 s) sum well below the 10,800 s total budget.

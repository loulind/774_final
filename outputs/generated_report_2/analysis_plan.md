# Analysis plan

Objective: Determine which clinical concepts can be reliably predicted from images, and identify data-quality issues such as class imbalance, near-duplicates, and noisy captions.

Questions: predictive_modelling, data_quality_audit, visualization. Source: llm.

| # | step | tool | inputs | parameters | purpose | handles |
|---|---|---|---|---|---|---|
| 1 | image_qc | qc_images | images=images.image_path, split=split.split | defaults | Flag and remove unreadable, tiny, strip-shaped and blank images; the pass also hashes every image for the duplicate audit. | image_heterogeneity:images.image_path |
| 2 | text_qc | qc_text | text=text.caption, split=split.split | defaults | Remove empty and very short captions; flag repeated, boilerplate and panel-referencing ones. | text_noise:text.caption |
| 3 | features | multimodal_features | images=@image_qc.images, text=@text_qc.text, split=split.split | defaults | Screen pretrained encoders on held-out pairs (label-free) and extract image and caption features. | image_heterogeneity:images.image_path |
| 4 | imbalance_concepts | label_imbalance | target=multilabel.concepts, split=split.split | defaults | Frequency audit of 'concepts': imbalance, long tail and how many labels have the support to be judged. | class_imbalance:multilabel.concepts |
| 5 | imbalance_concepts_manual | label_imbalance | target=multilabel.concepts_manual, split=split.split | defaults | Frequency audit of 'concepts_manual': imbalance, long tail and how many labels have the support to be judged. | class_imbalance:multilabel.concepts_manual |
| 6 | duplicates | near_duplicates | images=@image_qc.images, embedding=@features.image_features, split=split.split, groups=groups.article | defaults | Find near-duplicate images (confirmed hashes, calibrated embedding cosine); clusters crossing splits or articles are leakage. | near_duplicates:images.image_path, split_leakage:groups.article,split.split, split_leakage:images.image_path,split.split |
| 7 | label_sources | label_agreement | labels=multilabel.concepts, reference=multilabel.concepts_manual | defaults | Agreement of 'concepts' with 'concepts_manual' per shared label: an estimate of label noise. | label_noise:multilabel.concepts,multilabel.concepts_manual |
| 8 | caption_labels | caption_label_consistency | text=@text_qc.text, labels=multilabel.concepts, split=split.split, reference=multilabel.concepts_manual | defaults | Flag labels their caption does not support or negates, and captions naming a label the sample lacks. | label_noise:multilabel.concepts,multilabel.concepts_manual, text_noise:text.caption |
| 9 | image_text | multimodal_agreement | image=@features.image_features, text=@features.text_features, split=split.split, groups=groups.article | defaults | Flag image-caption pairs that match worse than random partners (noisy or mismatched captions), with articles kept together in cross-fitting. | text_noise:text.caption |
| 10 | atypical | generative_density | features=@features.image_features, split=split.split | defaults | Score how atypical every image is under a cross-fitted density model, as a quality signal next to the QC flags. | image_heterogeneity:images.image_path |
| 11 | predict | predict_targets | features=@features.image_features, target=multilabel.concepts, split=split.split, groups=groups.article, leakage=@duplicates.clusters | defaults | Predict 'concepts' from image features with article- and duplicate-grouped folds, a single test evaluation, BH control and the configured reliability rule. | nested_sampling:groups.article, class_imbalance:multilabel.concepts, near_duplicates:images.image_path, split_leakage:groups.article,split.split, split_leakage:images.image_path,split.split, multiple_testing |
| 12 | predict_manual | predict_targets | features=@features.image_features, target=multilabel.concepts_manual, split=split.split, groups=groups.article, leakage=@duplicates.clusters | defaults | Predict the 18 manually curated concepts from image features under the same leakage-safe design, to check reliability verdicts against cleaner labels. | nested_sampling:groups.article, class_imbalance:multilabel.concepts_manual, near_duplicates:images.image_path, split_leakage:groups.article,split.split, split_leakage:images.image_path,split.split, multiple_testing |
| 13 | map | dimreduce_map | embedding=@features.image_features | defaults | A two-dimensional map of the image features for the report. | - |

## Planning decisions

- **data-quality audits**: class_imbalance, image_heterogeneity, label_noise, near_duplicates, split_leakage, text_noise. The challenges the objective names (class_imbalance, label_noise, near_duplicates, text_noise) and every relevant challenge the profile detected.
- **prediction target and features**: 'concepts' from images. Named by the objective; 'concepts_manual' defines some of the same labels and serves as the reference for label noise. Rejected: predict 'concepts_manual': a reference source, used to estimate noise.
- **cross-validation groups**: article + near-duplicate clusters. Samples share articles (68% of them); near-copies link samples too, so folds and the test separation keep linked samples together. Rejected: random folds: related samples leak.
- **reliably predicted**: configured rule. >= 20 positive evaluation samples, lower 95% DeLong bound of AUROC >= 0.8, BH-adjusted p < 0.05 against AUROC 0.5; fewer positives: too rare to assess Rejected: rank targets by AUROC: a ranking does not say which are reliable.
- **challenge handling**: 10 of 11 relevant challenges handled by steps; 0 accepted. Each relevant challenge is matched to the steps whose tool addresses its type and whose inputs derive from its data; every high-severity one (4) must be handled or accepted.
- **planner used**: llm. LLM revised the rule-based plan: added step predict_manual (predict_targets); image_text.inputs: {'image': '@features.image_features', 'text': '@features.text_features', 'split': 'split.split'} -> {'image': '@features.image_features', 'text': '@features.text_features', 'split': 'split.split', 'groups': 'groups.article'}

## Validation

Valid; estimated 7,230 s in all (longest step 3,522 s).

## Notes

- compute (medium): declared costs ≈ 3522+742+738+321+321+213+2×67 ≈ 5990 s plus small audit steps, within 10,800 s total; multimodal_features at 98% of the step budget is kept at defaults.

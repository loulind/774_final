# Methods summary

Each step, the tool it ran (and where the tool comes from), its non-default settings and what it decided. Full rationale is in decision_log.json.

**1. image_qc** - `qc_images` (new; new in Project 4 (the hash helper is Project 3's, in agents/multimodal/imaging.py)). Flags unreadable, tiny, strip-shaped and blank images; one cached pass also measures sizes and difference hashes for later steps.
  - image QC rules: unreadable, shorter side < 32px, aspect > 5, thumbnail grey s.d. < 2

**2. text_qc** - `qc_text` (new; new in Project 4 (text normalisation for duplicates as in Project 3's profiler)). Flags empty, very short, repeated, boilerplate and panel-referencing texts; removes the empty and very short ones.
  - texts removed: empty, too_short

**3. features** - `multimodal_features` (project3; Project 3 (multimodal_agent): encoders/registry.py, encoders/pretrained.py, preprocess/image.py and text.py, models/linear.py (CCA), the contrastive head and loss, retrieval and probe metrics and the image profiler copied; screening, colour and alignment rules adapted from encoders/screening.py and planner/heuristic.py; tool wiring in agents/multimodal/ is new). Frozen image (and caption) features from pretrained encoders. Candidates for the measured image domain are screened on held-out pairs by how well their features align with the paired modality (CCA or zero-shot recall@10, label-free); stain colour is normalised per group when it differs between groups.
  - image domain: greyscale_imaging
  - colour normalisation: none
  - image encoder: biomedclip_image
  - text encoder: biomedclip_text

**4. imbalance_concepts** - `label_imbalance` (new; new in Project 4 (difference hash from Project 3's profiler)). Label frequency audit: counts per split, imbalance ratio, Gini, effective number of labels and how many labels have the support to be assessed.
  - labels assessable: 221 of 1,947

**5. imbalance_concepts_manual** - `label_imbalance` (new; new in Project 4 (difference hash from Project 3's profiler)). Label frequency audit: counts per split, imbalance ratio, Gini, effective number of labels and how many labels have the support to be assessed.
  - labels assessable: 17 of 18

**6. duplicates** - `near_duplicates` (new; new in Project 4 (difference hash from Project 3's profiler)). Near-duplicate images by difference hash and, calibrated on those, by embedding cosine; clusters (usable as CV groups), cross-split leakage and cross-group reuse.
  - embedding duplicate threshold: cosine >= 0.9473
  - near-duplicate clusters: 3,373 clusters, 6,782 images

**7. label_sources** - `label_agreement` (new; new in Project 4 (difference hash from Project 3's profiler)). Agreement between two label sources on co-annotated samples (Cohen's kappa, precision and recall per shared label): an estimate of label noise.
  - label noise estimate: kappa 0.96 on 18 shared labels

**8. caption_labels** - `caption_label_consistency` (new; new in Project 4 (text normalisation for duplicates as in Project 3's profiler)). Learns each label's indicative words from the training texts and flags labels their text does not support, labels the text negates, and texts that name a label the sample lacks.
  - labels checked: 300 of 1947

**9. image_text** - `multimodal_agreement` (project3; Project 3 (multimodal_agent): encoders/registry.py, encoders/pretrained.py, preprocess/image.py and text.py, models/linear.py (CCA), the contrastive head and loss, retrieval and probe metrics and the image profiler copied; screening, colour and alignment rules adapted from encoders/screening.py and planner/heuristic.py; tool wiring in agents/multimodal/ is new). Image-text agreement per pair: zero-shot, CCA and contrastive alignments are compared by validation recall@10 (label-free); the best scores each pair against random partners (cross-fitted), and pairs that agree less than an unrelated partner would are flagged.
  - alignment training / validation pairs: the predefined split (train fits, val selects; test unused)
  - contrastive head and training: embed_dim=256, hidden=512, n_layers=2, dropout=0.1, residual=True, lr=0.001, weight_decay=0.0001, batch_size=512, max_epochs=300, patience=15, seconds=240.0
  - image-text alignment: zero_shot

**10. atypical** - `generative_density` (project2; Project 2 (genmodel_agent): models/base.py, train.py, vae.py, flows.py, registry.py and the NB-vs-ZINB fit copied; warrant, likelihood and hyperparameter rules adapted from planner/heuristic.py; tool wiring in agents/generative/ is new). Density model (masked autoregressive flow, kept only if it beats a Gaussian on held-out data) on the leading principal components of a representation; flags samples whose density is unusually low, each scored by a cross-fitted model that was not fitted on it.
  - density fit / validation samples: the predefined split ('split.split': train fits, val selects)
  - density space: 57 principal components, standardised
  - flow hyperparameters: n_blocks=5, hidden_dim=256, noise_std=0.0, lr=0.001, batch_size=256, max_epochs=300, weight_decay=1e-05
  - density model: MAF

**11. predict** - `predict_targets` (new; new in Project 4 (DeLong after Sun & Xu 2014)). Predicts labels or multi-label targets from features (logistic or ridge + Platt), grouped cross-validation for penalties, a single test-split evaluation (or out-of-fold), DeLong AUROC intervals, the configured reliability rule, calibration and a leakage sensitivity check.
  - targets modelled: 300 of 1947
  - prediction model: ridge
  - fit / test separation: 7,921 fitting samples set aside
  - evaluation design: fit on the non-test splits, scored once on the test split

**12. predict_manual** - `predict_targets` (new; new in Project 4 (DeLong after Sun & Xu 2014)). Predicts labels or multi-label targets from features (logistic or ridge + Platt), grouped cross-validation for penalties, a single test-split evaluation (or out-of-fold), DeLong AUROC intervals, the configured reliability rule, calibration and a leakage sensitivity check.
  - targets modelled: 18 of 18
  - prediction model: logistic
  - fit / test separation: 7,811 fitting samples set aside
  - evaluation design: fit on the non-test splits, scored once on the test split

**13. map** - `dimreduce_map` (project1; Project 1 (dimreduce_agent): methods/, evaluate.py, registry.py copied; sub-agent rules in agents/dimreduce/agent.py are new). Two-dimensional map: feasible methods from different families are fitted and scored (trustworthiness, continuity, seed stability); the most faithful stable map is kept.
  - reduce before mapping: PCA to 50 dimensions
  - 2-D map method: umap

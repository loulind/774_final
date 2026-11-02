# Methods summary

Each step, the tool it ran (and where the tool comes from), its non-default settings and what it decided. Full rationale is in decision_log.json.

**1. qc** - `qc_counts` (new; new in Project 4 (the hash helper is Project 3's, in agents/multimodal/imaging.py)). Removes empty and low-quality samples of a count matrix (MAD outliers in log library size or log detected features, per library when libraries are large enough) and near-empty features.
  - count QC thresholds: per slide
  - features kept: 11,871 of 11,871

**2. split** - `count_split` (new; new in Project 4). Splits every count into two independent halves (Poisson or negative-binomial thinning): one to find groups, one to test them.
  - count splitting: poisson thinning, epsilon 0.5

**3. norm** - `normalize_counts` (new; new in Project 4). Normalises counts (log, analytic Pearson residuals or TF-IDF, chosen by how much depth signal is left in PC1) and selects variable features within batches.
  - variable features: 2,000 by Pearson residual variance within each slide
  - normalisation: log

**4. pca** - `dimreduce_pca` (project1; Project 1 (dimreduce_agent): methods/, evaluate.py, registry.py copied; sub-agent rules in agents/dimreduce/agent.py are new). Principal components for later steps; the number of components is chosen by parallel analysis (Horn), with the 90%-variance rule as the logged alternative.
  - standardise features before PCA: yes
  - analysis dimension: 50

**5. batch_assess** - `batch_assess` (new; new in Project 4 (Harmony through the harmonypy package; ComBat implemented after Johnson et al. 2007; variance share from Project 3's profiler)). Measures batch effects in a representation (silhouette, variance shares, neighbour mixing), within coarser groups when batches are nested, and how batches coincide with external labels.
  - batch effect: weak

**6. batch_vae** - `generative_batch_vae` (project2; Project 2 (genmodel_agent): models/base.py, train.py, vae.py, flows.py, registry.py and the NB-vs-ZINB fit copied; warrant, likelihood and hyperparameter rules adapted from planner/heuristic.py; tool wiring in agents/generative/ is new). Count VAE (NB or ZINB, chosen by per-gene likelihood) conditioned on a batch label; its latent means are a batch-corrected representation (scVI-style), a non-linear alternative to linear corrections.
  - use a batch-conditioned count VAE?: yes
  - count likelihood: NB
  - count VAE hyperparameters: latent_dim=10, hidden_dim=128, n_layers=1, beta=1.0, lr=0.001, batch_size=256, max_epochs=300, weight_decay=1e-06

**7. batch_correct** - `batch_correct` (new; new in Project 4 (Harmony through the harmonypy package; ComBat implemented after Johnson et al. 2007; variance share from Project 3's profiler)). Compares Harmony, ComBat, an optional candidate embedding and no correction; keeps the best batch mixing that preserves within-batch neighbourhoods (label-free), inside coarse groups when nested.
  Settings: within_outer=no.
  - correction scope: across all samples
  - batch correction: combat

**8. features** - `multimodal_features` (project3; Project 3 (multimodal_agent): encoders/registry.py, encoders/pretrained.py, preprocess/image.py and text.py, models/linear.py (CCA), the contrastive head and loss, retrieval and probe metrics and the image profiler copied; screening, colour and alignment rules adapted from encoders/screening.py and planner/heuristic.py; tool wiring in agents/multimodal/ is new). Frozen image (and caption) features from pretrained encoders. Candidates for the measured image domain are screened on held-out pairs by how well their features align with the paired modality (CCA or zero-shot recall@10, label-free); stain colour is normalised per group when it differs between groups.
  - patch size: 1 x spot diameter
  - image domain: stained_tissue
  - colour normalisation: none
  - image encoder: phikon

**9. domains** - `spatial_domains` (new; new in Project 4 (domain features after BANKSY; Moran's I as in Project 3's profiler, extended with per-library centring and the randomisation variance)). Spatial domains: features joined with their spatial neighbourhood mean (BANKSY-style) and clustered with the subsampling-stability rule; reports chance-corrected spatial coherence.
  - domain features: own + neighbourhood mean, lambda 0.8
  - spatial domains: method and number of clusters: kmeans, k=5

**10. validate_label** - `label_association` (new; new in Project 4). Agreement of found clusters or domains with external labels (ARI, AMI, NMI, purity, contingency), overall and per group; validation only.
  - validation against external labels: 3,123 labelled samples

**11. norm_all** - `normalize_counts` (new; new in Project 4). Normalises counts (log, analytic Pearson residuals or TF-IDF, chosen by how much depth signal is left in PC1) and selects variable features within batches.
  Settings: n_features=10000.
  - variable features: 10,000 by Pearson residual variance within each slide
  - normalisation: pearson_residuals

**12. svg** - `spatial_svg` (new; new in Project 4 (domain features after BANKSY; Moran's I as in Project 3's profiler, extended with per-library centring and the randomisation variance)). Spatially variable features: Moran's I per feature within each library, pooled, BH-corrected, with a permutation check of the test's calibration.
  - spatially variable features: 5,840 of 10,000 at FDR 0.05

**13. enrichment** - `spatial_enrichment` (new; new in Project 4 (domain features after BANKSY; Moran's I as in Project 3's profiler, extended with per-library centring and the randomisation variance)). Which labels (domains, cell types) border each other more or less than chance: neighbour-edge counts against within-library label permutations, BH-corrected.
  - neighbourhood enrichment test: 999 label permutations within each library

**14. histology** - `multimodal_predictability` (project3; Project 3 (multimodal_agent): encoders/registry.py, encoders/pretrained.py, preprocess/image.py and text.py, models/linear.py (CCA), the contrastive head and loss, retrieval and probe metrics and the image profiler copied; screening, colour and alignment rules adapted from encoders/screening.py and planner/heuristic.py; tool wiring in agents/multimodal/ is new). How well a categorical target is predicted from image features alone, another representation alone and both (early fusion): grouped nested cross-validation with a class-balanced logistic probe, macro-F1 and balanced accuracy against chance, with a group-bootstrap interval.
  - cross-validation: 5-fold, grouped by patient

**15. markers** - `differential_expression` (new; new in Project 4 (moderated t after limma's eBayes; log-CPM after edgeR)). Markers of every label against the rest with the replicate (patient) as the unit: pseudobulk log-CPM, paired moderated t, BH; sign consistency, leave-one-replicate-out stability, and the naive sample-level Wilcoxon test for comparison.
  - test unit: patient (pseudobulk)

**16. map** - `dimreduce_map` (project1; Project 1 (dimreduce_agent): methods/, evaluate.py, registry.py copied; sub-agent rules in agents/dimreduce/agent.py are new). Two-dimensional map: feasible methods from different families are fitted and scored (trustworthiness, continuity, seed stability); the most faithful stable map is kept.
  - 2-D map method: tsne

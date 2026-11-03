# Biomedical analysis agent

BIOS 774, Project 4. Given a biomedical dataset and a free-text objective, the agent

1. profiles the data,
2. reads the objective as a set of scientific questions,
3. lists the analytical challenges the data pose, with evidence and a severity for each (the
   **challenge register**),
4. plans an analysis across a registry of 27 tools, including the agents of Projects 1–3,
5. runs the plan and evaluates every step against criteria it declared beforehand,
6. lets a critic order at most one revision, and
7. writes a report.

Every choice is logged with its reason and the alternatives it rejected. The challenge
register, not the dataset, decides which steps the plan contains: nothing in the agent's
code, configuration or prompts refers to a particular dataset (dataset names appear only in
`scripts/prepare_data.py`).

The two reports in `outputs/` were produced with the commands under
[Reproducing the reports](#reproducing-the-reports).

| report | data | objective |
|---|---|---|
| [`outputs/generated_report_1/`](outputs/generated_report_1/report.md) | STimage-1K4M, HER2-positive breast cancer cohort: 13,620 spatial transcriptomics spots, 36 slides, 8 patients | "Characterize spatial tissue heterogeneity: identify spatial domains, relate them to histology, find marker genes, and assess slide-level batch effects." |
| [`outputs/generated_report_2/`](outputs/generated_report_2/report.md) | ROCOv2 v2.0.1: all 79,789 radiology images with captions and concept labels | "Determine which clinical concepts can be reliably predicted from images, and identify data-quality issues such as class imbalance, near-duplicates, and noisy captions." |

## Contents

- [Quick start](#quick-start)
- [Reproducing the reports](#reproducing-the-reports)
- [What a run writes](#what-a-run-writes)
- [How the agent works](#how-the-agent-works)
- [How the agent identifies challenges](#how-the-agent-identifies-challenges)
- [How the agent selects strategies](#how-the-agent-selects-strategies)
- [Tools](#tools)
- [Reused components and their sources](#reused-components-and-their-sources)
- [Configuration](#configuration)
- [Results of the two reports](#results-of-the-two-reports)
- [Tests](#tests)
- [Data, licences and third-party code](#data-licences-and-third-party-code)
- [Limitations](#limitations)
- [Repository layout](#repository-layout)

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv installs it if missing).
Developed on macOS with an Apple M2 Pro (16 GB, PyTorch on MPS); CUDA and CPU-only machines
are detected as well, and CPU runs get smaller training budgets.

```bash
uv sync                                                  # .venv from uv.lock
uv run bioagent hardware                                 # device, memory, disk and compute budgets
uv run python -I scripts/prepare_data.py --only quick    # small subsets of both datasets
uv run pytest -q                                         # offline: fake encoders, scripted LLM
```

One command per dataset and objective, in quick mode (small budgets; a smoke test, not
the analysis). `--no-llm` uses the rule-based planner, critic and writer:

```bash
uv run bioagent run --quick --no-llm --data data/stimage_her2_quick --out runs/quick_report_1 \
  --objective "Characterize spatial tissue heterogeneity: identify spatial domains, relate them to histology, find marker genes, and assess slide-level batch effects."
uv run bioagent run --quick --no-llm --data data/rocov2_quick --out runs/quick_report_2 \
  --objective "Determine which clinical concepts can be reliably predicted from images, and identify data-quality issues such as class imbalance, near-duplicates, and noisy captions."
```

Or from Python:

```python
from analysis_agent import run_agent
summary = run_agent("data/stimage_her2_quick", "<objective>", "runs/demo", quick=True, no_llm=True)
```

`requirements.txt` pins the same versions as `uv.lock` for
`pip install -r requirements.txt` (Python 3.12, which also installs this package and its
`bioagent` command).

**LLM.** Calls go through the locally installed, logged-in `claude` CLI (Claude Code) via
`claude-agent-sdk`; no API key is used. The model is configurable (default
`claude-opus-5-5`; `--model` or `llm.model` in the config), and the model that actually
answered is read back from each response and recorded. Without the CLI, or with
`--no-llm`, every LLM role falls back to its rule-based counterpart.

**Pretrained encoders** (open weights only: nothing that needs an account, a token or
acceptance of terms) are downloaded from Hugging Face on first use and cached there.

Other commands: `bioagent profile` (profile and challenge register only), `bioagent plan`
(the validated plan without running it), `bioagent tools` (the registry; `--name <tool>`
for one tool's full declaration), `bioagent schema` (the JSON schemas in `schemas/`).

## Reproducing the reports

```bash
uv run python -I scripts/prepare_data.py               # both datasets, full and quick subsets
uv run bioagent run --data data/stimage_her2 --out outputs/generated_report_1 \
  --objective "Characterize spatial tissue heterogeneity: identify spatial domains, relate them to histology, find marker genes, and assess slide-level batch effects."
uv run bioagent run --data data/rocov2 --out outputs/generated_report_2 \
  --objective "Determine which clinical concepts can be reliably predicted from images, and identify data-quality issues such as class imbalance, near-duplicates, and noisy captions."
```

Data sizes, as checked by the script (it compares them with free disk space first and
verifies checksums after downloading): the HER2 cohort downloads 0.73 GB and keeps
0.80 GB; ROCOv2 downloads 6.4 GB of image archives and 29 MB of tables and keeps about
2.2 GB of images (shorter side capped at 256 px). The archives stay in `data/raw/` unless
`--delete-archives` is given. `prepare_data.py --help` lists the other options
(`--dataset`, `--only quick|full`, `--reuse-from` to copy verified files from an earlier
download).

**Replay.** `--replay <run directory>` answers every LLM call from that run's
`llm_transcript.json`, so the planner's, critic's and writer's decisions are reproduced at
no cost:

```bash
uv run bioagent run --data data/stimage_her2 --out runs/replay_report_1 --replay outputs/generated_report_1 \
  --objective "Characterize spatial tissue heterogeneity: identify spatial domains, relate them to histology, find marker genes, and assess slide-level batch effects."
```

Calls are matched by their prompt and, when a prompt differs (for example in the last digits
of a GPU-computed number), by role and position. Every run uses seed 774; model fits on
tabular inputs run on the CPU, which is bit-for-bit reproducible, while pretrained-encoder
features computed on a GPU may differ in the last digits between machines. Replaying both
committed reports with the code in this repository, on the machine that made them,
reproduced every step metric exactly (177 and 170 values) apart from one wall-clock
training time.

## What a run writes

Everything goes to `--out`; caches and scratch go to the gitignored `runs/`.

| file | content |
|---|---|
| `report.md`, `report.html` | the report: objective, executive summary, data and challenges, plan, results with figures, critic and revision, limitations, reproducibility, and any number in the prose that was not found in the outputs |
| `methods_summary.md` | one entry per step: the tool and where it comes from, its non-default settings and what it decided |
| `profile.json` | the data profile |
| `objective.json` | question types, their targets and constraints |
| `challenge_register.json`, `.md` | every challenge type checked: evidence, severity, relevance, how it was handled, the risk that remains |
| `analysis_plan.json`, `.md`, `validation.json` | the plan that ran, its validation report and the planning decisions |
| `evaluation.json` | per step: metrics, criteria (with thresholds and pass/fail), time, errors |
| `decision_log.json` | every decision, with its source (`heuristic`, `llm`, `subagent`, `validator`, `critic`, `system`), rationale and rejected alternatives |
| `llm_transcript.json` | every LLM call: role, prompt, model requested and used, cost and response (used by `--replay`) |
| `run_manifest.json` | command, package versions, hardware, seed, budgets, criteria |
| `results_index.json` | the record the prose was written from, and the number check |
| `figures/`, `tables/`, `results/` | figures (PNG), step tables (CSV) and per-sample results such as domain labels, marker tables and quality flags (each capped in size) |

## How the agent works

```
dataset + objective
 -> load: format-generic readers; column roles (group, label, split, text, image path ...) inferred from content
 -> profile: measurements per modality, grouping, label set and split
 -> objective: question types with targets ("what to predict, from what"), constraints such as the reliability rule
 -> challenge register: evidence, severity from named thresholds, relevance to the questions, candidate tools
 -> plan: rule-based proposal -> LLM review and edits -> validator (one repair round, else the rule-based plan)
 -> execute: typed artifacts passed between tools, cached; each step scored against its declared criteria
 -> critic: deterministic triggers + LLM judgement; at most one revision, kept only if its criterion improves
 -> register update: how each challenge was handled, what risk remains, challenges only the results reveal
 -> figures, tables, results; prose (every number checked against the outputs); report
```

**One tool interface.** Every tool, ported or new, declares its origin, the typed artifacts
it consumes and produces (counts, normalised matrix, embedding, labels, groups,
coordinates, images, text, scores, tables ...), a parameter model with allowed ranges, the
question types it answers, the challenge types it addresses, a cost estimate and its
evaluation criteria. Each criterion has a direction (`max`, `min` or `report`) and says
whether it may select anything: criteria computed against external labels or the test split
are report-only. The planner, validator and critic see only these declarations.

**Projects 1–3 as sub-agents.** Their modules were copied into this repository and put
behind the common interface (see [Reused components](#reused-components-and-their-sources)).
Inside a step they make their own rule-based decisions (Project 1: how many components and
which map; Project 2: whether a generative model is warranted, NB or ZINB, hyperparameters;
Project 3: encoder screening, colour normalisation, alignment), logged with source
`subagent`. They make no LLM calls of their own: the LLM plans across tools.

**LLM roles.** The planner receives the profile, the register, the objective, the rule-based
proposal and the tool declarations, and may adopt the proposal, change parameters within
their ranges, swap tools or add steps; the critic judges the results and may propose one
revision; the writer writes the report's prose from a compact record of the run. Replies are
JSON validated against schemas (`schemas/`) and guardrails, repaired once, and otherwise
replaced by the rule-based result. Each call has a cap of 16,000 output tokens, and a run
stops calling the LLM once it has spent $3. Responses are cached per run.

**The critic.** Deterministic triggers come first: discovered groups confined to single
levels of the independent unit, untestable or unstable markers, unstable groups, a number of
groups at the edge of the searched range, an under-corrected batch effect, no target
passing the reliability rule, a large leakage gap, or a declared criterion that missed its
threshold. Most carry a suggested revision. With the LLM, the critic sees the triggers,
the results and every step's editable parameters, and decides: no revision, or one
revision of its own, which must name a criterion the step declares with a direction
(`max` or `min`). Without the LLM, the first trigger's suggestion is tried. A revision
changes parameters, swaps a tool or adds a step; it is validated, the changed step and
everything downstream re-run, and it is kept only if its criterion, computed without the
test split or labels reserved for validation, improves by more than 0.01 (absolute on 0–1
scales, relative otherwise). Otherwise the original results are restored.

**Writing.** The LLM writer's prose is checked number by number against the run's outputs;
numbers it cannot find are listed in the report rather than silently kept. Without the LLM,
templates write the same sections from the same record.

## How the agent identifies challenges

The register is computed from the profile. Every challenge type is always checked, and each
entry records:

- the challenge type and a one-sentence summary;
- its evidence (metric, value, the rule it was graded by);
- a severity (none, low, medium or high) from named thresholds in
  `analysis_agent/challenges.py`, printed with the register;
- the question types it affects (so whether it is relevant to this objective);
- the tools that address it.

After the run, each entry also records which steps handled it, their key numbers and the
risk that remains. Two challenges can only be judged from results and are added then: groups
that follow the independent unit (Cramér's V ≥ 0.5 between discovered groups and, say,
patients) and pretrained encoders that may have seen the evaluation pairs (zero-shot
retrieval recall@10 ≥ 0.5).

| challenge | what it means | tools that address it |
|---|---|---|
| `low_quality_samples` | samples too sparse, empty, unreadable or degenerate to analyse | `generative_density`, `qc_counts`, `qc_images` |
| `depth_variation` | technical variation in library size or sequencing depth | `generative_batch_vae`, `normalize_counts` |
| `high_dimensionality` | many more features than can be modelled directly; sparsity | `dimreduce_map`, `dimreduce_pca`, `generative_batch_vae`, `multimodal_features`, `normalize_counts` |
| `batch_effect` | technical groups (slides, sites) that separate the data, possibly nested | `batch_assess`, `batch_correct`, `generative_batch_vae`, `multimodal_features` |
| `batch_confounding` | technical groups that coincide with the biology of interest | `batch_assess` |
| `nested_sampling` | samples nested in groups (spots in slides in patients), so the independent unit is the group, not the sample; also raised when no grouping identifies independent units and groups are to be compared | `batch_assess`, `differential_expression`, `multimodal_predictability`, `predict_targets` |
| `double_dipping` | testing hypotheses on the same data that generated them | `count_split` |
| `spatial_autocorrelation` | neighbouring samples are not independent | `spatial_domains`, `spatial_svg` |
| `partial_labels` | labels exist for only some samples or groups | `label_association` |
| `class_imbalance` | classes or labels with very different frequencies; a long tail | `label_imbalance`, `multimodal_predictability`, `predict_targets` |
| `label_noise` | labels that disagree between sources or with the data | `caption_label_consistency`, `label_agreement`, `label_consistency`, `multimodal_agreement` |
| `near_duplicates` | repeated or near-identical samples, which can leak across splits | `near_duplicates`, `predict_targets` |
| `split_leakage` | related samples (one group, or near-copies) on both sides of a predefined split | `near_duplicates`, `predict_targets` |
| `text_noise` | empty, very short, duplicated or boilerplate text | `caption_label_consistency`, `multimodal_agreement`, `qc_text` |
| `missing_data` | missing values or missing modalities | `generative_impute` |
| `image_heterogeneity` | images that differ in size, colour, channels or staining | `generative_density`, `multimodal_features`, `qc_images` |
| `multiple_testing` | many hypotheses tested at once | `differential_expression`, `predict_targets`, `spatial_enrichment`, `spatial_svg` |
| `compute` | estimated cost close to or above the compute budget | (checked by the validator's cost estimate) |
| `pretraining_overlap` | a pretrained model may have seen the evaluation data | `multimodal_features` (flagged; reported as a limitation) |

## How the agent selects strategies

1. **Objective to questions.** Cue phrases mark question types in the objective; its nouns
   are then matched against the data (column names, modality words, the measured image
   domain) to name each question's targets: what to predict and from what, which grouping
   is the batch, which label source is the reference. "Reliably" fixes the configured
   reliability rule before any result is seen.

   | question type | meaning |
   |---|---|
   | `structure_discovery` | find groups of samples (clusters, domains) without using labels |
   | `spatial_structure` | how features and groups are arranged in space: domains, spatially variable features, neighbourhood relations |
   | `cross_modal_association` | how one modality relates to another (e.g. image to expression, image to text) |
   | `differential_analysis` | which features differ between groups, with error control |
   | `batch_assessment` | how strongly technical groups (slides, sites, runs) shape the data, and whether and how to correct for them |
   | `predictive_modelling` | how well an outcome can be predicted, estimated without leakage |
   | `data_quality_audit` | unusable samples, imbalance, duplicates, label and text noise |
   | `representation_learning` | a compact numeric representation of the samples for later steps |
   | `generative_modelling` | a probability model of the data: densities, imputation, conditional (batch-aware) latent spaces |
   | `visualization` | two-dimensional maps and figures for inspection |

2. **Recipes, chained by artifact kinds.** Each question type has a recipe; a tool is used
   only where earlier steps can supply its inputs. Challenges insert steps and set
   parameters, for example:
   - QC when samples fail the quality rule;
   - count splitting when groups found in counts are then tested for markers on them;
   - batch assessment and correction when a grouping separates the data, with the
     correction's scope set by where the replicate unit sits;
   - colour normalisation per group when image colour differs between groups;
   - cross-validation and test separation grouped by the independent unit and by
     near-duplicate clusters;
   - marker tests across samples, stated as such, when no grouping identifies replicates.

   Each step states its purpose, the challenges it addresses, the alternatives it rejected
   and its criteria before it runs.
3. **Guardrails** (`analysis_agent/planner/validator.py`), applied to every plan, whoever
   wrote it:
   - schema, tools and parameters stay within the declared ranges (repaired once);
   - every input exists and is of a kind its port accepts;
   - external labels the objective does not ask to predict or test are used only by
     validation-only steps, and nothing consumes a validation-only output;
   - prediction keeps the independent unit whole, and marker tests use it as the replicate;
   - markers of discovered groups are tested on the held-out half of a count split;
   - every family of tests the objective asks for controls the FDR;
   - every relevant high-severity challenge is handled by a step whose inputs derive from
     the challenge's data, or accepted as a limitation with a reason;
   - every question is answered, and the estimated cost fits the budget.
4. **"Reliably predicted"** is fixed in the configuration before any result: at least 20
   positive evaluation samples, a lower 95% DeLong bound of AUROC of at least 0.80, and a
   Benjamini–Hochberg-adjusted p-value below 0.05 against AUROC 0.5. Targets with fewer
   positives are reported as "too rare to assess", not as failures.

## Tools

27 tools: 8 from Projects 1–3 behind the common interface, 19 new.

| tool | origin | what it does | addresses |
|---|---|---|---|
| `dimreduce_pca` | Project 1 | principal components; how many by Horn's parallel analysis | high_dimensionality |
| `dimreduce_map` | Project 1 | 2-D map: feasible methods (PCA variants, MDS, Isomap, LLE, Laplacian eigenmaps, diffusion maps, t-SNE, UMAP) scored by trustworthiness, continuity and seed stability | high_dimensionality |
| `generative_batch_vae` | Project 2 | count VAE (NB or ZINB by per-gene likelihood) conditioned on a batch label (scVI-style) | batch_effect, depth_variation, high_dimensionality |
| `generative_density` | Project 2 | density model (MAF, kept only if it beats a Gaussian on held-out data) that flags atypical samples | low_quality_samples, image_heterogeneity |
| `generative_impute` | Project 2 | MIWAE imputation, used only if it beats mean/mode, k-NN and MICE on held-out entries | missing_data |
| `multimodal_features` | Project 3 | frozen image (and text) features; encoders screened on held-out pairs; colour normalisation by measured colour shifts | image_heterogeneity, batch_effect, high_dimensionality, pretraining_overlap |
| `multimodal_predictability` | Project 3 | how well a target is predicted from image features alone, another representation alone and both; grouped nested cross-validation | nested_sampling, class_imbalance |
| `multimodal_agreement` | Project 3 | per-pair agreement of two representations of the same samples (zero-shot, CCA or contrastive alignment); flags pairs that match worse than random partners | text_noise, label_noise |
| `qc_counts` | new | MAD outliers in depth and detected features, per library when needed | low_quality_samples |
| `qc_images` | new | unreadable, tiny, strip-shaped and blank images; sizes and hashes for later steps | low_quality_samples, image_heterogeneity |
| `qc_text` | new | empty, very short, repeated, boilerplate and panel-referencing texts | text_noise |
| `normalize_counts` | new | log, analytic Pearson residuals or TF-IDF, chosen by the depth signal left; variable features within batches | depth_variation, high_dimensionality |
| `count_split` | new | Poisson or negative-binomial data thinning into a half that finds groups and a half that tests them | double_dipping |
| `batch_assess` | new | silhouettes, variance shares and neighbour mixing, within the coarser grouping when batches are nested; batch-label association (report only) | batch_effect, batch_confounding, nested_sampling |
| `batch_correct` | new | Harmony, ComBat, an optional candidate embedding and no correction; the best mixing that preserves within-batch neighbourhoods | batch_effect |
| `cluster` | new | k-means, Gaussian mixtures and Leiden; the finest candidate whose every cluster is reproducible across subsamples | — |
| `spatial_domains` | new | features joined with their neighbourhood mean (BANKSY-style), clustered with the same stability rule | spatial_autocorrelation |
| `spatial_svg` | new | Moran's I per feature within libraries, BH FDR, with a permutation check of calibration | spatial_autocorrelation, multiple_testing |
| `spatial_enrichment` | new | which labels border each other more or less than chance (permutations within libraries) | multiple_testing |
| `label_association` | new | agreement of found groups with external labels or another partition (ARI, AMI, NMI, purity), overall and per group | partial_labels |
| `differential_expression` | new | markers per group: replicate pseudobulk, moderated t (limma-style), BH; leave-one-replicate-out stability; the naive per-sample test for comparison; per-sample tests when there are no replicates | nested_sampling, multiple_testing |
| `predict_targets` | new | logistic or ridge (+ Platt) for single- or multi-label targets; grouped cross-validation for penalties; one test-split evaluation (or within-fold AUROC); DeLong intervals; leakage sensitivity | class_imbalance, near_duplicates, split_leakage, nested_sampling, multiple_testing |
| `near_duplicates` | new | difference hashes, then embedding neighbours calibrated on the confirmed pixel pairs; clusters usable as CV groups; cross-split leakage | near_duplicates, split_leakage |
| `label_imbalance` | new | imbalance ratio, Gini, effective number of labels, labels with the support to be judged | class_imbalance |
| `label_agreement` | new | Cohen's kappa, precision and recall between two label sources | label_noise |
| `label_consistency` | new | agreement of one label within near-duplicate clusters (Fleiss' kappa) | label_noise |
| `caption_label_consistency` | new | labels a caption does not support, negates or names without the label (lexicons learned on training texts) | label_noise, text_noise |

## Reused components and their sources

Projects 1–3 are separate repositories; the modules this project needs were copied in and
adapted to the common tool interface, so this repository runs on its own. The earlier
repositories were not modified. Each copied module says where it came from in its first
line, and each sub-agent's tools carry a `SOURCE` string (shown by `bioagent tools --json`).

**Project 1**, dimension reduction: [github.com/loulind/774_dimreduce](https://github.com/loulind/774_dimreduce) (`dimreduce_agent/`)

| here (`analysis_agent/...`) | from | changes |
|---|---|---|
| `agents/dimreduce/methods/linear.py`, `manifold.py`, `neighbor.py`, `diffusion.py` | `methods/` | unchanged |
| `agents/dimreduce/methods/__init__.py` | `methods/__init__.py` | the GPLVM method is not copied |
| `agents/dimreduce/evaluate.py` | `evaluate.py` | unchanged |
| `agents/dimreduce/registry.py` | `registry.py` | no GPLVM; feasibility from n, p and sparsity |
| `agents/dimreduce/agent.py`, `adapter.py` | — | new: parallel analysis for the analysis dimension, a time-aware feasibility filter and the map-selection rule (Project 1 compared methods but left the choice to the reader) |

**Project 2**, generative models: [github.com/loulind/774_genmodel](https://github.com/loulind/774_genmodel) (`genmodel_agent/`)

| here | from | changes |
|---|---|---|
| `agents/generative/models/flows.py`, `train.py` | `models/flows.py`, `models/train.py` | unchanged |
| `agents/generative/models/vae.py` | `models/vae.py` | the convolutional image branch is not copied (images enter as encoder features) |
| `agents/generative/models/base.py` | `models/base.py` | `ModelSpec` is a small dataclass instead of Project 2's plan schema |
| `agents/generative/registry.py` | `models/registry.py` | only the models used here |
| `agents/generative/likelihood.py` | `profiler.py` (NB versus ZINB by per-gene likelihood) | moved into its own module |
| `agents/generative/agent.py`, `adapter.py` | rules adapted from `planner/heuristic.py` | warrant, likelihood and hyperparameter rules adapted; tool wiring new |

**Project 3**, multimodal learning: [github.com/loulind/774_multimodal](https://github.com/loulind/774_multimodal) (`multimodal_agent/`)

| here | from | changes |
|---|---|---|
| `data/dataset.py`, `loaders.py`, `roles.py`, `splits.py` | `data/` | format-generic loading and column-role inference |
| `config.py`, `hardware.py`, `llm.py`, `decision_log.py` | same names (Project 3 had adapted them from Projects 1–2) | budgets, criteria and LLM settings for this project; `llm.py` keeps the CLI backend, cache, replay and spend cap |
| `schema.py` (decision-log and split models) | `schema.py` | the profile, register, objective and plan models are new |
| `writer.py` | `writer.py` | number check and template fallback adapted to this report's sections |
| `figures.py` | `figures.py` | style and palette; the figures are new |
| `agents/multimodal/encoders/registry.py`, `pretrained.py` | `encoders/` | unchanged |
| `agents/multimodal/preprocess.py` | `preprocess/image.py`, `preprocess/text.py` | unchanged |
| `agents/multimodal/linear.py` | `models/linear.py` | the CCA class only |
| `agents/multimodal/align.py` | `models/heads.py`, the InfoNCE loss of `models/losses.py`, the loop of `models/train.py` | reduced to one loss |
| `agents/multimodal/metrics.py` | `evaluate/metrics.py`, `retrieval.py`, `probes.py` | the functions used here |
| `agents/multimodal/imaging.py` | `profiler.py` (image measurements) | functions of an image source and rows |
| `agents/multimodal/agent.py`, `adapter.py` | rules adapted from `encoders/screening.py`, `planner/heuristic.py`, `evaluate/probes.py` | new: grouped nested cross-validation with a group-bootstrap interval; cross-fitted pair scores |
| `variance_share` in `tools/stats.py`; the difference hash and text normalisation used by the QC, duplicate and text tools; Moran's I (extended here with per-library centring and the randomisation variance) | `profiler.py` | as noted in each module |

Everything else is new in Project 4: the profiler (`profiler/`), the challenge register
(`challenges.py`, `taxonomy.py`), objective reading (`objective.py`), the planners and
validator (`planner/`), the executor, critic and orchestrator, the report, the CLI, the
prompts and schemas, and the 19 new tools in `tools/`. Methods implemented from the
literature are cited in their modules: ComBat (Johnson et al. 2007), limma's moderated t
(Smyth 2004), edgeR-style log-CPM, DeLong's AUROC variance (Sun and Xu 2014),
BANKSY-style neighbourhood features, cluster stability (Hennig 2007) and the separation
test (after Liu et al. 2008).

## Configuration

`configs/default.yaml` (overridden by `--config <yaml>` and command-line flags) sets
how much compute a run gets, which LLM to ask, the seed and what counts as a finding:

- budgets for full and quick runs: estimated seconds per plan and per step, per-fit
  training time and epochs, folds, bootstrap and permutation counts, and the sample cap of
  quick mode;
- criteria: FDR 0.05, 95% intervals, |log2 fold change| ≥ 1 for markers, and the
  reliability rule above;
- LLM: model, output-token cap, effort per role, spend cap per run.

It never says which tool, method, number of clusters or correction to use: those come
only from the profile, the register and the objective, so a configuration cannot become a
dataset-specific branch in disguise.

## Results of the two reports

All numbers below are from the committed outputs (`evaluation.json` and the reports).

**Report 1**, spatial tissue heterogeneity (LLM planner, critic and writer; 3 calls, $0.79
recorded, `claude-opus-5-5`):

- QC removed 202 of 13,619 spots; markers were tested on the held-out half of a Poisson
  count split.
- Log normalisation lowered |rho(PC1, depth)| from 0.98 to 0.52.
- Slides separate the data. ComBat across slides, chosen over Harmony, the count VAE and no
  correction, raised neighbour mixing from 0.32 to 0.71 while keeping 0.75 of within-slide
  neighbourhoods.
- 5 spatial domains: subsample stability 0.84, spatial coherence 0.64 against 0.28 for the
  spots' own features. All 5 occur in all 8 patients (Cramér's V with patient 0.20).
- Agreement with the pathologist annotations (validation only, never used to choose): ARI
  0.066, NMI 0.12.
- 5,840 of 10,000 tested genes are spatially variable at FDR 5%.
- Histology predicts the domains with balanced accuracy 0.36 (95% CI 0.30–0.42) against
  chance 0.20, with patient-grouped cross-validation.
- 699 domain–gene marker pairs (582 genes) at FDR 5% and |log2 fold change| ≥ 1, every one
  stable when any single patient is left out. The naive per-spot test would report 941.
- The critic found two criteria just missed (|rho(PC1, depth)| 0.524 against a bar of 0.5;
  map trustworthiness 0.786 against 0.8) and did not revise.

**Report 2**, which clinical concepts images predict, and data quality (3 calls, $0.69
recorded, `claude-opus-5-5`):

- BiomedCLIP image and text encoders, chosen by screening (recall@10 0.955). Zero-shot
  retrieval was as high, so possible pretraining overlap is flagged.
- 1,947 automatically extracted concepts with an imbalance ratio of 27,798. 176 of the 214
  assessable concepts are reliably predicted (median test AUROC 0.94); 86 of the modelled
  concepts are too rare to assess.
- On the 18 curated concepts: 17 of 17 assessable are reliable (median AUROC 0.996).
- Near-duplicates: 3,373 clusters holding 8.5% of the images, 949 of them across splits.
  6.4% of test images have a duplicate in the training or validation split. Duplicate clusters and articles were
  kept together in folds and in the test separation (7,921 fitting images set aside), and a
  fit with the leaking images gains only 0.0005 median AUROC.
- Label noise: the two concept sources agree with kappa 0.964 on their 18 shared concepts.
  For 20.3% of the checked labels, the caption gives no support.
- The critic found no trigger and did not revise.

## Tests

```bash
uv run pytest -q
```

118 tests, offline (fake encoders and a scripted LLM backend), about a minute. They cover:

- known answers for the statistics (BH, DeLong and its within-fold variant, Moran's I, the
  moderated t, rank-sum tests, kappa, Cramér's V);
- every tool on synthetic data with planted effects;
- the profiler and register on planted and clean synthetic data: the right challenges at
  the right severity, and no false alarms;
- objective reading and the planner's recipes, and the guardrails (double dipping,
  grouping, reserved labels, FDR, challenge coverage);
- the LLM layer: validation, repair, fallback, caching and replay;
- the critic loop;
- end-to-end runs that write every output.

## Data, licences and third-party code

- **STimage-1K4M** (Chen et al. 2024, arXiv:2406.06393), HER2-positive breast cancer
  cohort, from Hugging Face `jiawennnn/STimage-1K4M` at a pinned revision (MIT licence tag on
  the dataset page): 36 slides from 8 patients, 13,620 spots and 11,871 genes, with
  pathologist annotations for 3,481 spots.
- **ROCOv2** v2.0.1 (Rückert et al., *Scientific Data* 2024,
  doi:[10.1038/s41597-024-03496-6](https://doi.org/10.1038/s41597-024-03496-6)), from Zenodo
  doi:[10.5281/zenodo.10821435](https://doi.org/10.5281/zenodo.10821435), CC BY-NC 4.0:
  79,789 images in the official splits (59,958 train, 9,904 validation, 9,927 test), with
  captions, automatically extracted UMLS concepts and manually curated ones. The dataset's
  per-image licence information is read by `prepare_data.py`.
- Data are never committed: `data/` is gitignored and `scripts/prepare_data.py` recreates it.
- `harmonypy` is GPL-3.0: it is installed as a dependency and called, never copied.
- Pretrained encoders have open weights; their licences and pretraining corpora are
  recorded in `analysis_agent/agents/multimodal/encoders/registry.py`.

## Limitations

- Question types are read from cue phrases; unusual wording can miss one. The LLM planner
  can add questions, and the plan notes any question no step answers.
- A categorical outcome stored as integer codes is read as a numeric column unless
  `--label-column` names it. The rule-based planner predicts the label the objective names
  (or the first); the LLM planner can add more.
- Without any grouping of samples into independent units, markers are tested across
  samples; the register and report say that such results describe one source.
- Pretrained encoders may have seen the evaluation data; this is flagged from zero-shot
  retrieval and cannot be ruled out.
- The critic makes at most one revision, judged only by label-free, non-test criteria.
- Quick-mode numbers are smoke tests: few subsamples, permutations and epochs.

## Repository layout

```
analysis_agent/
  __init__.py          run_agent(data_path, objective, out_dir, **overrides)
  cli.py               bioagent run | profile | plan | tools | schema | hardware
  orchestrator.py      the loop: load -> profile -> objective -> register -> plan -> execute -> critic -> report
  data/                format-generic loaders and column roles (Project 3)
  profiler/            measurements per modality, grouping, label set and split
  challenges.py        the challenge register: rules, thresholds, post-run update
  taxonomy.py          question and challenge types
  objective.py         objective -> question types, targets, constraints
  planner/             rule-based planner, LLM planner, validator
  tools/               the interface (base.py), registry and the new tools
  agents/              dimreduce (Project 1), generative (Project 2), multimodal (Project 3)
  executor.py critic.py writer.py report.py figures.py summaries.py
  config.py hardware.py llm.py decision_log.py schema.py
configs/default.yaml   budgets, seed, criteria, LLM settings
prompts/               planner, critic and writer instructions
schemas/               JSON schemas of the profile, register, objective, plan and LLM replies
scripts/prepare_data.py   downloads and converts the two datasets (the only place that names them)
tests/                 pytest suite
outputs/               the two generated reports
```

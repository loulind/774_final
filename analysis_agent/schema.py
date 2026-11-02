"""Typed records passed between the agent's steps.

Every object here is JSON-serialisable, because each one is shown to the LLM,
written to the run directory and quoted in the decision log. Profile fields are
measurements, not judgements: the profile says "silhouette by slide 0.21", never
"strong batch effect". Turning numbers into challenges is the challenge
register's job, and turning challenges into steps is the planner's.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Decision log (from Project 3)
# ---------------------------------------------------------------------------

# "subagent": a choice a ported Project 1-3 agent made by its own rules, inside a step.
DecisionSource = Literal["heuristic", "llm", "validator", "critic", "subagent", "system"]


class Alternative(BaseModel):
    option: str
    reason: str


class Decision(BaseModel):
    step: str                     # profile | objective | challenges | plan | <step id> | critic | ...
    decision: str                 # what was being decided
    chosen: str
    rationale: str
    alternatives_rejected: list[Alternative] = Field(default_factory=list)
    evidence: dict[str, float | int | str | bool | None | list | dict] = Field(default_factory=dict)
    source: DecisionSource = "heuristic"


# ---------------------------------------------------------------------------
# Data splits (from Project 3)
# ---------------------------------------------------------------------------


class SplitSpec(BaseModel):
    strategy: Literal["predefined", "grouped_kfold", "random"]
    group_column: str | None = None
    n_folds: int = 1                              # grouped_kfold: folds over groups (1 = one held-out set)
    val_fraction: float = 0.1                     # random / carved from train when no val split
    test_fraction: float = 0.2                    # random, and grouped with one fold
    stratify_column: str | None = None


# ---------------------------------------------------------------------------
# Data profile: measurements only (new in Project 4)
# ---------------------------------------------------------------------------

Number = float | int | None


class ModalityRecord(BaseModel):
    name: str
    kind: str
    n_present: int
    missing_fraction: float
    source: str = ""


class ArtifactRecord(BaseModel):
    name: str                                     # kind.qualifier, as the planner binds it
    kind: str
    n: int
    detail: dict[str, str | int | float | bool | None | list] = Field(default_factory=dict)


class MatrixProfile(BaseModel):
    artifact: str
    is_counts: bool
    n_samples: int
    n_features: int
    sparsity: float                               # share of zero entries
    features_per_sample: float                    # p / n
    library_size_median: Number = None
    library_size_log2_iqr: Number = None
    library_median_range: list[float] | None = None   # smallest and largest per-library median library size
    n_empty: int = 0
    qc_flagged_fraction: Number = None            # samples failing the default count-QC rule
    qc_per_library: bool | None = None
    detected_features_median: Number = None
    n_features_detected_5pct: int | None = None
    overdispersed_fraction: Number = None         # detected features with variance > mean
    pc1_depth_rho_raw: Number = None              # |Spearman rho| of PC1 with log depth, log1p of raw counts
    pc1_depth_rho_scaled: Number = None           # the same after library-size scaling
    components_50: int | None = None              # principal components for 50% / 90% of the variance
    components_90: int | None = None
    variance_top_components: Number = None        # share of the variance in the components computed
    measured_on: str = ""


class GroupStructureProfile(BaseModel):
    matrix: str
    group: str
    column: str
    n_levels: int
    silhouette: Number = None
    variance_share: Number = None
    mixing: Number = None
    outer: str | None = None                      # grouping this one is nested in
    within_outer_silhouette: Number = None
    within_outer_mixing: Number = None
    share_between_outer: Number = None
    share_between_inner_within_outer: Number = None
    n_outer_with_several_levels: int | None = None
    measured_on: str = ""


class SpatialProfile(BaseModel):
    artifact: str
    matrix: str | None = None
    library: str | None = None
    n_libraries: int
    samples_per_library: list[float]              # min, median, max
    nn_distance_median: Number = None
    features_tested: int = 0
    fraction_autocorrelated: Number = None        # BH q < fdr
    median_morans_i: Number = None
    measured_on: str = ""


class ImageProfile(BaseModel):
    artifact: str
    source_kind: str
    n_images: int
    n_measured: int
    unusable_fraction: Number = None
    flag_counts: dict[str, int] = Field(default_factory=dict)
    width_median: Number = None
    height_median: Number = None
    size_iqr_ratio: Number = None                 # Q3 / Q1 of the shorter side
    n_colour_sampled: int = 0
    domain_hint: str | None = None                # stained_tissue | greyscale_imaging | colour_other
    greyscale_fraction: Number = None
    stain_hue_fraction: Number = None
    colour_group: str | None = None
    colour_between_group_share: Number = None
    n_hashed: int = 0
    n_hash_candidates: int | None = None          # 64-bit hash matches before the 256-bit confirmation
    duplicate_fraction: Number = None             # images in a confirmed near-duplicate pair
    n_duplicate_pairs: int | None = None
    cross_split_duplicate_pairs: int | None = None
    measured_on: str = ""


class TextProfile(BaseModel):
    artifact: str
    n_texts: int
    flag_fractions: dict[str, float]
    flagged_fraction: float                       # any flag
    words_median: float
    words_p95: float
    cross_split_duplicate_fraction: Number = None


class TabularProfile(BaseModel):
    artifact: str
    n_columns: int
    n_numeric: int
    missing_cell_fraction: float
    columns_with_missing: int


class LabelProfile(BaseModel):
    artifact: str
    column: str
    kind: str                                     # labels | multilabel
    n_annotated: int
    annotated_fraction: float
    n_classes: int
    imbalance_ratio: Number = None
    largest: dict[str, int] = Field(default_factory=dict)
    smallest: dict[str, int] = Field(default_factory=dict)
    labels_per_sample_median: Number = None
    gini: Number = None
    n_modelable: int | None = None                # enough training positives to fit
    n_assessable: int | None = None               # ... and enough test positives to judge
    too_rare_fraction: Number = None
    annotated_groups: dict[str, list[int]] = Field(default_factory=dict)   # group -> [levels labelled, levels]
    cramers_v: dict[str, float] = Field(default_factory=dict)              # group -> V(group, label)
    variance_share: dict[str, float] = Field(default_factory=dict)         # matrix -> share explained by label


class LabelAgreementProfile(BaseModel):
    first: str
    second: str
    n_samples_both: int
    n_shared_labels: int
    kappa_micro: Number = None
    precision_micro: Number = None
    recall_micro: Number = None
    first_source_coverage: Number = None


class GroupProfile(BaseModel):
    artifact: str
    column: str
    role: str
    n_levels: int
    size_min: int
    size_median: float
    size_max: int
    samples_per_level: float
    shared_fraction: float                        # samples whose level holds at least one other sample
    nested_in: list[str] = Field(default_factory=list)
    split_overlap: dict[str, float] | None = None   # split -> share of its samples whose level is in another split


class SplitProfile(BaseModel):
    artifact: str
    counts: dict[str, int]


class CostEstimate(BaseModel):
    tool: str
    inputs: dict[str, str]
    seconds: float


class ComputeProfile(BaseModel):
    device: str
    step_seconds: float
    total_seconds: float
    estimates: list[CostEstimate] = Field(default_factory=list)


class DataProfile(BaseModel):
    """What the planner, the register and the LLM see about the data: structure and
    measurements, never the dataset's name or location."""
    n_samples: int
    format: str
    modalities: list[ModalityRecord]
    roles: dict[str, str]
    artifacts: list[ArtifactRecord]
    matrices: list[MatrixProfile] = Field(default_factory=list)
    group_structure: list[GroupStructureProfile] = Field(default_factory=list)
    spatial: list[SpatialProfile] = Field(default_factory=list)
    images: list[ImageProfile] = Field(default_factory=list)
    texts: list[TextProfile] = Field(default_factory=list)
    tabular: list[TabularProfile] = Field(default_factory=list)
    labels: list[LabelProfile] = Field(default_factory=list)
    label_agreement: list[LabelAgreementProfile] = Field(default_factory=list)
    groups: list[GroupProfile] = Field(default_factory=list)
    split: SplitProfile | None = None
    pairing: dict[str, int] = Field(default_factory=dict)
    compute: ComputeProfile | None = None
    seed: int = 774
    seconds: float = 0.0
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Challenge register (new in Project 4)
# ---------------------------------------------------------------------------

Severity = Literal["none", "low", "medium", "high"]
ChallengeStatus = Literal["detected", "not_detected", "not_applicable", "not_measurable"]


class Evidence(BaseModel):
    metric: str                                   # plain description of what was measured
    value: float | int | str | bool | None
    rule: str = ""                                # the named thresholds it was judged by
    source: str = ""                              # where in the profile (or a step's results) it came from


class Challenge(BaseModel):
    id: str                                       # type:subject, unique in a register
    type: str                                     # a key of taxonomy.CHALLENGE_TYPES
    subject: list[str] = Field(default_factory=list)   # artifacts concerned
    summary: str                                  # one plain-English sentence
    status: ChallengeStatus
    severity: Severity
    evidence: list[Evidence] = Field(default_factory=list)
    affects: list[str] = Field(default_factory=list)   # question types threatened
    relevant: bool = True                         # affects one of the objective's questions
    mitigations: list[str] = Field(default_factory=list)   # tools whose `addresses` include the type
    stage: Literal["profile", "execution"] = "profile"
    handled_by: list[str] = Field(default_factory=list)    # plan steps, filled by the planner
    handling: str | None = None                   # filled after execution
    residual_risk: str | None = None


class ChallengeRegister(BaseModel):
    questions: list[str] | None = None            # question types relevance was judged against (None: all)
    thresholds: dict[str, dict[str, str | float | bool]] = Field(default_factory=dict)
    entries: list[Challenge] = Field(default_factory=list)

    def get(self, challenge_id: str) -> Challenge | None:
        return next((c for c in self.entries if c.id == challenge_id), None)

    def of_type(self, kind: str) -> list[Challenge]:
        return [c for c in self.entries if c.type == kind]

    def detected(self, min_severity: Severity = "low", relevant_only: bool = True) -> list[Challenge]:
        order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        return [c for c in self.entries if c.status == "detected" and order[c.severity] >= order[min_severity]
                and (c.relevant or not relevant_only)]


# ---------------------------------------------------------------------------
# Objective: the free-text objective read as question types (new in Project 4)
# ---------------------------------------------------------------------------

class Question(BaseModel):
    type: str                                     # a key of taxonomy.QUESTION_TYPES
    phrase: str = ""                              # the part of the objective it was read from
    targets: dict[str, list[str]] = Field(default_factory=dict)   # role -> artifacts (features, labels, batch, ...)
    challenges: list[str] = Field(default_factory=list)           # challenge types the objective names
    implied: bool = False                         # added by the agent (e.g. maps for the report), not asked
    evidence: str = ""                            # which cue matched


class Objective(BaseModel):
    text: str
    questions: list[Question] = Field(default_factory=list)
    constraints: dict[str, str | float | int | bool | dict | list] = Field(default_factory=dict)
    source: Literal["rules", "llm"] = "rules"
    notes: list[str] = Field(default_factory=list)

    def types(self, include_implied: bool = True) -> list[str]:
        return list(dict.fromkeys(q.type for q in self.questions if include_implied or not q.implied))

    def get(self, kind: str) -> Question | None:
        return next((q for q in self.questions if q.type == kind), None)


# ---------------------------------------------------------------------------
# Analysis plan (new in Project 4)
# ---------------------------------------------------------------------------

class PlanStep(BaseModel):
    id: str                                       # short and unique; later steps refer to "@id.port"
    tool: str                                     # a registry tool
    inputs: dict[str, str] = Field(default_factory=dict)   # port -> first artifact name or "@step.port"
    params: dict[str, str | int | float | bool | list | dict | None] = Field(default_factory=dict)
    purpose: str = ""                             # why, in one sentence
    answers: list[str] = Field(default_factory=list)       # question types it serves
    addresses: list[str] = Field(default_factory=list)     # challenge ids it handles
    alternatives_rejected: list[Alternative] = Field(default_factory=list)
    criteria: list[str] = Field(default_factory=list)      # the tool's declared criteria, judged after the run
    validation_only: bool = False                 # external labels / test split: reported, never used to choose


class AcceptedChallenge(BaseModel):
    challenge: str                                # register id
    reason: str


class AnalysisPlan(BaseModel):
    objective: str
    questions: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)
    accepted: list[AcceptedChallenge] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    source: Literal["heuristic", "llm", "repaired"] = "heuristic"

    def step(self, sid: str) -> PlanStep | None:
        return next((s for s in self.steps if s.id == sid), None)


class ValidationIssue(BaseModel):
    rule: str                                     # schema | tools | inputs | params | flow | labels | grouping | ...
    severity: Literal["error", "warning"]
    message: str
    step: str | None = None
    repaired: bool = False


class ValidationReport(BaseModel):
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    estimated_seconds: float = 0.0
    step_seconds: dict[str, float] = Field(default_factory=dict)

    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error" and not i.repaired]


# ---------------------------------------------------------------------------
# LLM replies (new in Project 4): what the planner, critic and writer must return
# ---------------------------------------------------------------------------

class DecisionDraft(BaseModel):
    step: str = "plan"
    decision: str
    chosen: str
    rationale: str
    alternatives_rejected: list[Alternative] = Field(default_factory=list)


class PlannerReply(BaseModel):
    plan: AnalysisPlan
    decisions: list[DecisionDraft] = Field(default_factory=list)
    added_questions: list[str] = Field(default_factory=list)   # question types the rules missed


class Revision(BaseModel):
    """One change to a plan that ran: new parameters for a step, another tool for it, or an
    added step; kept only if ``criterion`` of ``criterion_step`` improves."""
    action: Literal["set_params", "swap_tool", "add_step"]
    step: str                                     # the step changed, or the step the new one follows
    params: dict[str, str | int | float | bool | list | dict | None] = Field(default_factory=dict)
    tool: str | None = None                       # swap_tool
    new_step: PlanStep | None = None              # add_step
    criterion_step: str
    criterion: str                                # a label-free, non-test criterion of that step
    goal: Literal["max", "min"]
    trigger: str = ""
    reason: str = ""


class CriticReply(BaseModel):
    assessment: str
    revise: bool
    revision: Revision | None = None
    alternatives_rejected: list[Alternative] = Field(default_factory=list)


class WriterReply(BaseModel):
    executive_summary: str
    data_and_challenges: str
    plan_explained: str
    results_interpretation: str
    critic_and_revision: str
    limitations: str


# JSON schemas written to schemas/ (``bioagent schema --write``); a test keeps them in step.
SCHEMA_MODELS: dict[str, type[BaseModel]] = {"data_profile": DataProfile, "challenge_register": ChallengeRegister,
                                             "objective": Objective, "analysis_plan": AnalysisPlan,
                                             "planner_reply": PlannerReply, "critic_reply": CriticReply,
                                             "writer_reply": WriterReply}


def json_schemas() -> dict[str, dict]:
    return {name: model.model_json_schema() for name, model in SCHEMA_MODELS.items()}

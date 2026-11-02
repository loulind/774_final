"""The generative model library as data (from Project 2): what exists and its legal ranges.

A model may only be built from this table, with hyperparameters clipped into these
ranges. Adapted: only the models this project uses are listed -- the Gaussian VAE, the
negative-binomial and zero-inflated count VAEs, the masked VAE (MIWAE) for imputation,
and the RealNVP and MAF flows -- and ``build()`` points at this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# (low, high) inclusive. Integers stay integers when clipped.
Range = tuple[float, float]

COMMON_TRAINING: dict[str, Range] = {
    "lr": (1e-5, 1e-2),
    "batch_size": (16, 1024),
    "max_epochs": (1, 2000),
    "weight_decay": (0.0, 1e-2),
}


@dataclass(frozen=True)
class ModelInfo:
    name: str
    family: str                     # vae | gan | flow | diffusion
    modalities: tuple[str, ...]
    tasks: tuple[str, ...]
    conditional_capable: bool
    hp_ranges: dict[str, Range]
    defaults: dict[str, float | int | str | bool]
    representations: tuple[str, ...]        # data spaces the model can be trained in
    module: str                             # implementation: .models.<module>
    cls: str
    summary: str
    has_likelihood: bool = False            # exact or bounded log p(x) for scoring
    notes: tuple[str, ...] = field(default_factory=tuple)


GENERATIVE_TASKS = ("generation", "augmentation", "class_balancing")
BATCH_TASKS = ("batch_correction",)

REGISTRY: dict[str, ModelInfo] = {m.name: m for m in [
    ModelInfo(
        name="vae", family="vae", modalities=("tabular", "counts"),
        tasks=GENERATIVE_TASKS + ("representation_learning", "anomaly_detection"),
        conditional_capable=True,
        hp_ranges={"latent_dim": (2, 256), "hidden_dim": (16, 1024), "n_layers": (1, 4),
                   "beta": (0.05, 4.0), **COMMON_TRAINING},
        defaults={"latent_dim": 16, "hidden_dim": 256, "n_layers": 2, "beta": 1.0,
                  "lr": 1e-3, "batch_size": 128, "max_epochs": 200, "weight_decay": 0.0},
        representations=("standardized", "lognorm_hvg"),
        module="vae", cls="VAE",
        summary="Gaussian/categorical-likelihood (conditional) VAE on vectors",
        has_likelihood=True,
    ),
    ModelInfo(
        name="nb_vae", family="vae", modalities=("counts",),
        tasks=GENERATIVE_TASKS + ("representation_learning",) + BATCH_TASKS,
        conditional_capable=True,
        hp_ranges={"latent_dim": (2, 64), "hidden_dim": (32, 1024), "n_layers": (1, 3),
                   "beta": (0.05, 4.0), **COMMON_TRAINING},
        defaults={"latent_dim": 10, "hidden_dim": 128, "n_layers": 1, "beta": 1.0,
                  "lr": 1e-3, "batch_size": 128, "max_epochs": 400, "weight_decay": 1e-6},
        representations=("counts_hvg",),
        module="vae", cls="CountVAE",
        summary="scVI-style VAE with negative-binomial likelihood and observed library size",
        has_likelihood=True,
    ),
    ModelInfo(
        name="zinb_vae", family="vae", modalities=("counts",),
        tasks=GENERATIVE_TASKS + ("representation_learning",) + BATCH_TASKS,
        conditional_capable=True,
        hp_ranges={"latent_dim": (2, 64), "hidden_dim": (32, 1024), "n_layers": (1, 3),
                   "beta": (0.05, 4.0), **COMMON_TRAINING},
        defaults={"latent_dim": 10, "hidden_dim": 128, "n_layers": 1, "beta": 1.0,
                  "lr": 1e-3, "batch_size": 128, "max_epochs": 400, "weight_decay": 1e-6},
        representations=("counts_hvg",),
        module="vae", cls="CountVAE",
        summary="as nb_vae, with a per-gene zero-inflation component (ZINB)",
        has_likelihood=True,
    ),
    ModelInfo(
        name="masked_vae", family="vae", modalities=("tabular",),
        tasks=("imputation", "uncertainty_quantification"),
        conditional_capable=False,
        hp_ranges={"latent_dim": (2, 64), "hidden_dim": (16, 512), "n_layers": (1, 3),
                   "n_importance": (1, 50), **COMMON_TRAINING},
        defaults={"latent_dim": 8, "hidden_dim": 128, "n_layers": 2, "n_importance": 10,
                  "lr": 1e-3, "batch_size": 64, "max_epochs": 500, "weight_decay": 1e-5},
        representations=("standardized_onehot",),
        module="vae", cls="MaskedVAE",
        summary="MIWAE-style VAE trained on observed entries only; imputes by importance-weighted draws",
        has_likelihood=True,
    ),
    ModelInfo(
        name="realnvp", family="flow", modalities=("tabular",),
        tasks=GENERATIVE_TASKS + ("anomaly_detection",), conditional_capable=False,
        hp_ranges={"n_blocks": (2, 16), "hidden_dim": (16, 512), "noise_std": (0.0, 0.5),
                   **COMMON_TRAINING},
        defaults={"n_blocks": 6, "hidden_dim": 128, "noise_std": 0.0, "lr": 1e-3,
                  "batch_size": 256, "max_epochs": 300, "weight_decay": 1e-5},
        representations=("standardized",),
        module="flows", cls="RealNVP",
        summary="RealNVP affine coupling flow; exact log-likelihood",
        has_likelihood=True, notes=("ambient dimension must be modest (<= 200)",),
    ),
    ModelInfo(
        name="maf", family="flow", modalities=("tabular",),
        tasks=GENERATIVE_TASKS + ("anomaly_detection",), conditional_capable=False,
        hp_ranges={"n_blocks": (2, 16), "hidden_dim": (16, 512), "noise_std": (0.0, 0.5),
                   **COMMON_TRAINING},
        defaults={"n_blocks": 5, "hidden_dim": 128, "noise_std": 0.0, "lr": 1e-3,
                  "batch_size": 256, "max_epochs": 300, "weight_decay": 1e-5},
        representations=("standardized",),
        module="flows", cls="MAF",
        summary="Masked autoregressive flow; exact log-likelihood, slow sampling",
        has_likelihood=True, notes=("ambient dimension must be modest (<= 200)",),
    ),
]}

# Maximum ambient dimension for models that work directly in feature space with
# full-rank transforms; beyond this the planner must reduce dimension first.
MAX_FLOW_DIM = 200
MAX_MODELS_PER_PLAN = 3


def allowed(modality: str, task: str) -> list[str]:
    return [m.name for m in REGISTRY.values() if modality in m.modalities and task in m.tasks]


def clip_hyperparameters(name: str, hps: dict) -> tuple[dict, list[str]]:
    """Clip to registry ranges and drop unknown keys. Returns (clean, notes)."""
    info = REGISTRY[name]
    ranges = info.hp_ranges
    clean, notes = {}, []
    for key, value in {**info.defaults, **hps}.items():
        if key not in ranges:
            if key in hps and key not in info.defaults:
                notes.append(f"{name}: dropped unknown hyperparameter '{key}'")
                continue
            clean[key] = value
            continue
        lo, hi = ranges[key]
        try:
            v = float(value)
        except (TypeError, ValueError):
            notes.append(f"{name}: non-numeric {key}={value!r} replaced by default")
            v = float(info.defaults.get(key, lo))
        clipped = min(max(v, lo), hi)
        if clipped != v:
            notes.append(f"{name}: {key}={value} outside [{lo}, {hi}], clipped to {clipped:g}")
        is_int = isinstance(info.defaults.get(key), int) and not isinstance(
            info.defaults.get(key), bool)
        clean[key] = int(round(clipped)) if is_int else clipped
    return clean, notes


def build(spec, **kwargs):
    """Instantiate the implementation of ``spec.name`` (lazy torch import)."""
    import importlib

    info = REGISTRY[spec.name]
    module = importlib.import_module(f"{__package__}.models.{info.module}")
    return getattr(module, info.cls)(spec=spec, **kwargs)

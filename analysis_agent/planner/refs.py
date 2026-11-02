"""How plan steps refer to data, shared by the planner and the validator.

A step input is either a first artifact (``counts.expression``, ``groups.slide``) or the output
of an earlier step, written ``@step.port``. From the plan alone - before anything runs - this
module resolves every reference to an artifact kind, traces which first artifacts each output
derives from (so a challenge about ``counts.expression`` is known to be handled by a step that
consumes its normalised, corrected embedding), and propagates sizes to estimate each step's
cost with the tool's declared cost function.
"""

from __future__ import annotations

from ..config import Budget
from ..schema import AnalysisPlan, DataProfile
from ..tools import registry

# Width of outputs whose size the plan cannot know before running; the declared parameter
# gives it where there is one (variable features kept, components at most).
OUTPUT_WIDTH = {("generative_batch_vae", "embedding"): 30, ("multimodal_features", "image_features"): 768,
                ("multimodal_features", "text_features"): 768, ("dimreduce_map", "map"): 2}
WIDTH_PARAM = {("normalize_counts", "normalized"): "n_features", ("dimreduce_pca", "embedding"): "max_components"}


# Inputs that only guide a step (the partner a feature screen aligns to, features that seed a
# model) and so do not make it a handler of challenges about their data.
AUXILIARY_PORTS = {("multimodal_features", "partner"), ("multimodal_features", "target"),
                   ("generative_batch_vae", "features")}
GROUPING_KINDS = ("groups", "split")


def is_ref(value: str) -> bool:
    return isinstance(value, str) and value.startswith("@") and "." in value


def split_ref(value: str) -> tuple[str, str]:
    sid, port = value[1:].split(".", 1)
    return sid, port


class Trace:
    """Kinds, origins and sizes of every first artifact and planned output."""

    def __init__(self, plan: AnalysisPlan, profile: DataProfile):
        self.kind: dict[str, str] = {a.name: a.kind for a in profile.artifacts}
        self.origin: dict[str, set[str]] = {a.name: {a.name} for a in profile.artifacts}
        self.size: dict[str, tuple[int, int]] = {}
        self.external = {a.name for a in profile.artifacts if a.detail.get("external")}
        for a in profile.artifacts:
            width = a.detail.get("n_features") or a.detail.get("n_labels") or 1
            self.size[a.name] = (int(a.n or profile.n_samples), int(width))
        self.producer: dict[str, str] = {}
        self.unknown: list[tuple[str, str, str]] = []          # (step, port, reference) that did not resolve
        for step in plan.steps:
            self.add_step(step)

    def add_step(self, step) -> None:
        try:
            spec = registry.get(step.tool)
        except KeyError:
            return
        origins: set[str] = set()
        first = None
        for port in spec.consumes:
            ref = step.inputs.get(port.name)
            if ref is None:
                continue
            if ref not in self.kind:
                self.unknown.append((step.id, port.name, ref))
                continue
            origins |= self.origin[ref]
            if first is None:
                first = self.size.get(ref)
        n, width = first or (0, 1)
        try:
            params = spec.parse_params(step.params)
        except ValueError:
            params = spec.params()
        for port in spec.produces:
            ref = f"@{step.id}.{port.name}"
            self.kind[ref] = port.kinds[0]
            self.origin[ref] = origins | {ref}
            self.producer[ref] = step.id
            w = OUTPUT_WIDTH.get((step.tool, port.name))
            if (step.tool, port.name) in WIDTH_PARAM:
                w = getattr(params, WIDTH_PARAM[(step.tool, port.name)], None) or w
            if port.kinds[0] in ("labels", "groups", "mask", "table"):
                w = 1
            self.size[ref] = (n, int(w or width))

    def derives_from(self, ref: str, artifacts) -> bool:
        return bool(self.origin.get(ref, set()) & set(artifacts))

    def handles(self, step, challenge) -> bool:
        """Whether a step's inputs concern a challenge: a (non-auxiliary) input derives from the
        challenge's data, and a grouping the challenge is about is bound to the step. Challenges
        about the whole dataset concern every step whose tool addresses their type."""
        subject = [x for x in challenge.subject if x in self.kind]
        data = [x for x in subject if self.kind[x] not in GROUPING_KINDS]
        grouping = [x for x in subject if self.kind[x] in GROUPING_KINDS]
        refs = [r for p, r in step.inputs.items() if (step.tool, p) not in AUXILIARY_PORTS]
        if data and not any(self.derives_from(r, data) for r in refs):
            return False
        return not grouping or bool(set(grouping) & set(refs))


def step_costs(plan: AnalysisPlan, profile: DataProfile, budget: Budget) -> dict[str, float]:
    """Each step's estimated seconds: the tool's declared cost at the size of its first input."""
    tr = Trace(AnalysisPlan(objective=plan.objective), profile)
    out = {}
    for step in plan.steps:
        try:
            spec = registry.get(step.tool)
            params = spec.parse_params(step.params)
        except (KeyError, ValueError):
            tr.add_step(step)
            continue
        first = next((tr.size[step.inputs[p.name]] for p in spec.consumes
                      if step.inputs.get(p.name) in tr.size), None)
        n, p = first or (profile.n_samples, 1)
        if budget.max_samples:
            n = min(n, budget.max_samples)
        out[step.id] = float(spec.cost(n, p, params, budget))
        tr.add_step(step)
    return out

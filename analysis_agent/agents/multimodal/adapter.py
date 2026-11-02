"""Project 3 as three tools: screened frozen features, grouped predictability, image-text agreement."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from ...tools.base import Artifact, Criterion, Port, StepResult, ToolParams, ToolSpec, check_time, dense, digest
from ...tools.registry import register
from . import agent, imaging
from .encoders.registry import REGISTRY, candidates, is_dual_pair
from .metrics import recall_at_k

SOURCE = ("Project 3 (multimodal_agent): encoders/registry.py, encoders/pretrained.py, preprocess/image.py "
          "and text.py, models/linear.py (CCA), the contrastive head and loss, retrieval and probe "
          "metrics and the image profiler copied; screening, colour and alignment rules adapted from "
          "encoders/screening.py and planner/heuristic.py; tool wiring in agents/multimodal/ is new")

ImageEncoder = Literal[tuple(n for n, e in REGISTRY.items() if "image" in e.kinds)]
TextEncoder = Literal[tuple(n for n, e in REGISTRY.items() if "text" in e.kinds)]
VAL_GALLERY = 2000               # validation pairs behind a recall@10 comparison


def _values_on(rows: np.ndarray, art: Artifact | None) -> np.ndarray | None:
    """An optional artifact's values (as strings) for ``rows``; "" where it has none."""
    if art is None:
        return None
    out = np.full(len(rows), "", object)
    common = np.intersect1d(rows, art.rows, assume_unique=True)
    out[np.searchsorted(rows, common)] = np.asarray(art.take(common)).astype(str)
    return out


def _release():
    from .encoders.pretrained import release_all

    release_all()


def _cache(ctx):
    from .encoders.pretrained import FeatureCache

    return FeatureCache(ctx.cache_dir / "features") if ctx.cache_dir else None


# ---------------------------------------------------------------------------
# screened frozen features
# ---------------------------------------------------------------------------

class FeaturesParams(ToolParams):
    image_encoders: list[ImageEncoder] | None = Field(
        None, description="image encoders to screen; None = the registry's candidates for the measured domain")
    text_encoders: list[TextEncoder] | None = Field(
        None, description="text encoders to screen; None = the registry's candidates for the image domain")
    max_candidates: int = Field(3, ge=1, le=6, description="candidates per modality (capped by the budget)")
    domain: Literal["auto", "stained_tissue", "greyscale_imaging", "colour_other"] = Field(
        "auto", description="image domain; auto = measured on a sample of images")
    colour_normalisation: Literal["auto", "reinhard", "none"] = Field(
        "auto", description="auto = Reinhard per library group when the between-group colour share is "
                            f">= {agent.COLOUR_ICC_NORMALISE}")
    patch_context: float = Field(1.0, ge=0.5, le=4.0, description="patch side as a multiple of the spot "
                                                                  "diameter (spot patches only)")
    rank_by: Literal["auto", "alignment", "probe", "prior"] = Field(
        "auto", description="auto = alignment with the paired text or partner features (label-free); "
                            "else a probe on the target's training and validation samples; else the "
                            "registry's prior order")


def _run_features(ctx, inputs, p: FeaturesParams) -> StepResult:
    img, text, partner = inputs["images"], inputs.get("text"), inputs.get("partner")
    target, library, split = inputs.get("target"), inputs.get("library"), inputs.get("split")
    source, rows = img.data, img.rows
    spots = hasattr(source, "with_context")
    if spots:
        source = source.with_context(p.patch_context)
        ctx.decide("patch size", f"{p.patch_context:g} x spot diameter",
                   "Each patch covers the spot's own footprint, the tissue its RNA was captured from; "
                   "it is resized to the encoder's input size." if p.patch_context == 1.0 else "Set by the plan.",
                   [("2.0 x spot diameter", "adds context but also cells outside the spot")]
                   if p.patch_context == 1.0 else [], {"patch_context": p.patch_context})

    # what the images look like: domain and between-group colour share
    lib = _values_on(rows, library)
    has_lib = None if lib is None else lib != ""
    summ = (imaging.image_summary(source, rows[has_lib], lib[has_lib], ctx.rng(11)) if lib is not None
            else imaging.image_summary(source, rows, None, ctx.rng(11)))
    domain = summ["domain_hint"] if p.domain == "auto" else p.domain
    ctx.decide("image domain", domain, summ["evidence"] if p.domain == "auto" else "Set by the plan.", [],
               {k: v for k, v in summ.items() if k != "evidence"})
    use_norm, why = agent.colour_normalisation(summ, p.colour_normalisation, lib is not None)
    normalizer = agent.fit_normalizer(source, rows[has_lib], lib[has_lib], ctx.rng(12)) if use_norm else None
    lib_name = library.meta.get("column", library.name) if library is not None else None
    ctx.decide("colour normalisation", f"Reinhard per {lib_name}" if use_norm else "none", why,
               [("none", "group-level colour shifts would act as a batch effect")] if use_norm else [],
               {"colour_between_group_share": summ["colour_between_group_share"],
                **(normalizer.describe() if normalizer else {})})
    group_of = None if lib is None else dict(zip(rows.tolist(), lib.tolist()))
    data_key = img.meta.get("dataset_key") or img.provenance

    def key_parts(r):
        return {"data": data_key, "rows": digest({"rows": r}), "n": len(r),
                "context": p.patch_context if spots else None, "colour": "reinhard" if use_norm else "none",
                "normaliser": normalizer.describe() if normalizer else None}

    rank = p.rank_by
    if rank == "auto":
        rank = "alignment" if (text is not None or partner is not None) else ("probe" if target is not None
                                                                             else "prior")
    if (rank == "alignment" and text is None and partner is None) or (rank == "probe" and target is None):
        raise ValueError(f"rank_by='{rank}' needs {'text or partner features' if rank == 'alignment' else 'a target'}")
    n_cand = int(min(p.max_candidates, ctx.budget.max_candidates))
    img_cands = list(p.image_encoders or candidates("image", domain, n_cand, ctx.quick))[: ctx.budget.max_candidates]
    txt_cands = ([] if text is None else
                 list(p.text_encoders or candidates("text", domain, n_cand, ctx.quick))[: ctx.budget.max_candidates])
    cache, failed, table, screen = _cache(ctx), {}, [], {}

    if rank != "prior":
        paired = text if rank == "alignment" and text is not None else (partner if rank == "alignment" else target)
        srows = np.intersect1d(rows, paired.rows, assume_unique=True)
        tr, va, how = agent.holdout(len(srows), _values_on(srows, split), _values_on(srows, inputs.get("holdout")),
                                    ctx.rng(13))
        tr, va = agent.cap(tr, ctx.budget.screening_train, ctx.rng(14)), agent.cap(va, ctx.budget.screening_val,
                                                                                  ctx.rng(15))
        if len(tr) < agent.MIN_SCREEN_PAIRS or len(va) < 10:
            ctx.decide("screen encoders?", "no", f"only {len(tr)} training and {len(va)} validation samples "
                       "to screen on; the registry's prior order is used", [("screen anyway", "too few "
                                                                                   "samples to rank")], {})
            rank = "prior"
    if rank != "prior":
        s_idx = np.concatenate([srows[tr], srows[va]])                # dataset positions, training first
        memo: dict = {}

        def screening_source():
            if "src" not in memo:                                     # loaded once for all candidates
                from ...data.dataset import MaterializedSource

                ctx.log(f"    loading {len(s_idx)} screening images once for all candidates")
                memo["src"] = MaterializedSource(source, s_idx, normalizer, group_of)
            return memo["src"]

        img_feats = {}
        for enc in img_cands:
            check_time(ctx, "encoder screening")
            ctx.log(f"    screening image encoder {enc}")
            try:
                F = agent.encode_images(enc, screening_source, s_idx, device=ctx.device,
                                        batch_size=ctx.budget.feature_batch_size, cache=cache,
                                        key_parts=key_parts(s_idx), progress=ctx.log)
                img_feats[enc] = (F[: len(tr)], F[len(tr):])
            except Exception as err:                                  # a failed candidate is reported
                failed[enc] = f"{type(err).__name__}: {str(err)[:200]}"
            finally:
                _release()
        partners = {}
        if rank == "alignment" and text is not None:
            texts = list(text.take(s_idx))
            for enc in txt_cands:
                check_time(ctx, "encoder screening")
                ctx.log(f"    screening text encoder {enc}")
                try:
                    F = agent.encode_texts(enc, texts, device=ctx.device, batch_size=ctx.budget.feature_batch_size,
                                           cache=cache, key_parts={"data": data_key, "rows": digest({"rows": s_idx}),
                                                                   "n": len(s_idx), "cleaning": "basic"})
                    partners[enc] = (F[: len(tr)], F[len(tr):])
                except Exception as err:
                    failed[enc] = f"{type(err).__name__}: {str(err)[:200]}"
                finally:
                    _release()
        elif rank == "alignment":
            P = dense(partner.take(s_idx))
            partners[partner.name] = (P[: len(tr)], P[len(tr):])
        if rank == "alignment":
            metric = "alignment_recall_at_10"
            for ea, (Atr, Ava) in img_feats.items():
                for eb, (Btr, Bva) in partners.items():
                    table.append({"image_encoder": ea, "partner": eb,
                                  **agent.alignment_scores(Atr, Ava, Btr, Bva, is_dual_pair(ea, eb))})
        else:
            metric = "probe_macro_f1"
            y = np.asarray(target.take(s_idx)).astype(str)
            for ea, (Atr, Ava) in img_feats.items():
                table.append({"image_encoder": ea, "probe_macro_f1": agent.probe_score(Atr, y[: len(tr)], Ava,
                                                                                       y[len(tr):])})
        ranked = agent.best_per_candidate(table, "image_encoder", metric)
        if not ranked:
            raise RuntimeError(f"no image encoder could be screened: {failed}")
        chance = min(1.0, 10 / len(va)) if rank == "alignment" else None
        clear = None if chance is None else agent.clearly_above(ranked[0][1][metric], chance, len(va))
        scored = dict(ranked)
        if clear is False:                           # the screen cannot tell candidates apart: prior decides
            img_enc = next(e for e in img_cands if e in scored)
            why = (f"No candidate is clearly above chance on {len(va)} held-out screening samples ({how}): "
                   f"best {ranked[0][1][metric]:.3f} vs chance {chance:.3f}, within two standard errors. "
                   "The screen cannot tell the candidates apart, so the registry's order for this domain "
                   "decides.")
        else:
            img_enc = ranked[0][0]
            why = (f"Highest {metric} on {len(va)} held-out screening samples, {how}: "
                   f"{scored[img_enc][metric]:.3f}" + ("" if chance is None else f" (chance {chance:.3f})") + ".")
        ctx.decide("image encoder", img_enc, f"{why} Pretrained on: {REGISTRY[img_enc].pretraining}.",
                   [(e, f"{metric} {r[metric]:.3f}") for e, r in ranked if e != img_enc]
                   + [(e, f"failed to load: {m}") for e, m in failed.items() if e in img_cands],
                   {"n_train": len(tr), "n_val": len(va), "metric": metric, "chance": chance,
                    "clearly_above_chance": clear})
        best, text_enc = scored[img_enc], None
        if text is not None and rank == "alignment":
            text_enc = next(e for e in txt_cands if e in partners) if clear is False else best["partner"]
            best = next(r for r in table if r["image_encoder"] == img_enc and r["partner"] == text_enc)
            pairs = [r for r in table if r["image_encoder"] == img_enc and r["partner"] != text_enc]
            ctx.decide("text encoder", text_enc,
                       ("The registry's first text candidate (the screen is inconclusive)" if clear is False
                        else f"Best partner of {img_enc}") + f": {metric} {best[metric]:.3f}.",
                       [(r["partner"], f"{metric} {r[metric]:.3f} with {img_enc}") for r in pairs]
                       + [(e, f"failed to load: {m}") for e, m in failed.items() if e in txt_cands], {})
        screen = {"screen_metric": metric, "screen_score": best[metric], "screen_chance": chance,
                  "screen_clear_of_chance": clear,
                  "screen_recall_at_10": best.get("alignment_recall_at_10"),
                  "zero_shot_recall_at_10": best.get("zero_shot_recall_at_10"),
                  "n_screen_train": len(tr), "n_screen_val": len(va)}
    else:
        img_enc = img_cands[0]
        ctx.decide("image encoder", img_enc, f"Nothing paired with the images to screen against: the "
                   f"registry's first candidate for '{domain}' images (pretrained on: "
                   f"{REGISTRY[img_enc].pretraining}).", [(c, "lower prior; not screened") for c in img_cands[1:]], {})
        text_enc = txt_cands[0] if txt_cands else None

    check_time(ctx, "feature extraction")
    ctx.log(f"    {img_enc}: features for {len(rows):,} images")
    F_img = agent.encode_images(img_enc, source, rows, device=ctx.device, batch_size=ctx.budget.feature_batch_size,
                                cache=cache, key_parts=key_parts(rows), normalizer=normalizer, group_of=group_of,
                                progress=ctx.log)
    _release()
    info = REGISTRY[img_enc]
    arts = {"image_features": Artifact("embedding", F_img, rows, {
        "method": "frozen encoder", "encoder": img_enc, "partner_encoder": info.partner,
        "pretraining": info.pretraining, "domain": domain, "source": img.name,
        "colour_normalisation": "reinhard" if use_norm else "none",
        "patch_context": p.patch_context if spots else None, "dim": int(F_img.shape[1])})}
    if text is not None and text_enc:
        ctx.log(f"    {text_enc}: features for {len(text.rows):,} texts")
        F_txt = agent.encode_texts(text_enc, list(text.data), device=ctx.device,
                                   batch_size=ctx.budget.feature_batch_size, cache=cache,
                                   key_parts={"data": data_key, "rows": digest({"rows": text.rows}),
                                              "n": len(text.rows), "cleaning": "basic"})
        _release()
        tinfo = REGISTRY[text_enc]
        arts["text_features"] = Artifact("embedding", F_txt, text.rows, {
            "method": "frozen encoder", "encoder": text_enc, "partner_encoder": tinfo.partner,
            "pretraining": tinfo.pretraining, "source": text.name, "dim": int(F_txt.shape[1])})
    metrics = {"image_encoder": img_enc, "text_encoder": text_enc, "domain": domain, "rank_by": rank,
               "greyscale_fraction": summ["greyscale_fraction"], "stain_hue_fraction": summ["stain_hue_fraction"],
               "colour_between_group_share": summ["colour_between_group_share"], "colour_normalised": use_norm,
               "n_candidates": len(img_cands) + len(txt_cands), "n_failed": len(failed), "n_images": len(rows),
               **screen}
    tables = {"encoder_screening": pd.DataFrame(table + [{"image_encoder" if e in img_cands else "partner": e,
                                                          "error": m} for e, m in failed.items()])}
    return StepResult(artifacts=arts, metrics=metrics, tables=tables)


def _features_cost(n, p, prm, budget) -> float:
    k = min(getattr(prm, "max_candidates", 3), budget.max_candidates)
    per_item = 0.02                                   # seconds per image or text on the GPU, loading included
    return float(2 * per_item * (k * (budget.screening_train + budget.screening_val) + n) + 30)


register(ToolSpec(
    name="multimodal_features", origin="project3", source=SOURCE,
    summary="Frozen image (and caption) features from pretrained encoders. Candidates for the measured "
            "image domain are screened on held-out pairs by how well their features align with the paired "
            "modality (CCA or zero-shot recall@10, label-free); stain colour is normalised per group when "
            "it differs between groups.",
    consumes=(Port("images", ("images",), doc="images to encode"),
              Port("text", ("text",), required=False, doc="paired captions (also encoded)"),
              Port("partner", ("embedding", "normalized"), required=False,
                   doc="paired features of another modality to screen against"),
              Port("target", ("labels",), required=False, doc="labels for probe screening (train/val only)"),
              Port("library", ("groups",), required=False, doc="staining or scanning batches (colour)"),
              Port("holdout", ("groups",), required=False, doc="groups kept apart between screening "
                                                               "training and validation"),
              Port("split", ("split",), required=False, doc="predefined split; test samples are not screened")),
    produces=(Port("image_features", ("embedding",), doc="frozen image features"),
              Port("text_features", ("embedding",), required=False, doc="frozen caption features")),
    params=FeaturesParams, answers=("representation_learning", "cross_modal_association"),
    addresses=("image_heterogeneity", "batch_effect", "high_dimensionality", "pretraining_overlap"),
    criteria=(Criterion("screen_recall_at_10", "max", None, "held-out recall@10 of the chosen encoder pair "
                                                             "after CCA, or zero-shot (label-free)"),
              Criterion("zero_shot_recall_at_10", "report", doc="recall@10 in a dual encoder's own space; a "
                                                                "very high value suggests pretraining overlap")),
    run=_run_features, cost=_features_cost,
    notes=("Encoders are open-weight models only; the first use of each downloads its weights.",),
))


# ---------------------------------------------------------------------------
# grouped predictability
# ---------------------------------------------------------------------------

class PredictParams(ToolParams):
    min_class_size: int = Field(10, ge=2, le=1000, description="classes with fewer samples are left out")
    max_classes: int = Field(30, ge=2, le=200, description="most frequent classes kept")


def _run_predictability(ctx, inputs, p: PredictParams) -> StepResult:
    from sklearn.metrics import f1_score

    target, groups, image, other = inputs["target"], inputs["groups"], inputs["image"], inputs.get("other")
    rows = target.rows
    for a in [groups, image] + ([other] if other is not None else []):
        rows = np.intersect1d(rows, a.rows, assume_unique=True)
    y_raw = np.asarray(target.take(rows)).astype(str)
    counts = pd.Series(y_raw).value_counts()
    kept = list(counts[counts >= p.min_class_size].index[: p.max_classes])
    if len(kept) < len(counts):
        ctx.decide("classes assessed", f"{len(kept)} of {len(counts)}",
                   f"classes with fewer than {p.min_class_size} samples (or beyond the {p.max_classes} most "
                   "frequent) cannot be judged by cross-validation", [("keep all", "too rare to assess")],
                   {"dropped": {str(c): int(counts[c]) for c in counts.index if c not in kept}})
    keep = np.isin(y_raw, kept)
    rows, y_raw = rows[keep], y_raw[keep]
    classes = sorted(kept)
    y = np.searchsorted(classes, y_raw)
    g = np.asarray(groups.take(rows)).astype(str)
    n_groups = len(np.unique(g))
    if len(classes) < 2 or n_groups < 2:
        raise RuntimeError(f"predictability needs >= 2 classes and >= 2 groups (have {len(classes)} and {n_groups})")
    n_folds = int(min(ctx.budget.max_folds, n_groups))
    col = groups.meta.get("column", groups.name)
    ctx.decide("cross-validation", f"{n_folds}-fold, grouped by {col}",
               f"Samples of one {col} are not independent, so whole groups are held out and performance is "
               f"judged on unseen groups ({n_groups} groups). Within each training part the penalty is chosen "
               f"by grouped inner cross-validation ({ctx.budget.inner_folds} folds, macro-F1).",
               [("random folds", "samples of one group on both sides would inflate the scores")],
               {"n_samples": len(rows), "n_groups": n_groups, "n_classes": len(classes)})
    sets = {"image": dense(image.take(rows))}
    if other is not None:
        sets["other"] = dense(other.take(rows))
        sets["both"] = np.hstack([sets["image"], sets["other"]])
    chance = 1 / len(classes)
    res, table = {}, []
    for name, X in sets.items():
        check_time(ctx, "predictability")
        ctx.log(f"    grouped CV on {name} features ({X.shape[1]} dims, {len(rows):,} samples)")
        res[name] = r = agent.grouped_predictability(X, y, g, n_folds, ctx.budget.inner_folds, ctx.seed,
                                                     ctx.budget.permutations,
                                                     check=lambda: check_time(ctx, "predictability"))
        table.append({"features": name, "dims": X.shape[1], **{k: r.get(k) for k in (
            "macro_f1", "balanced_accuracy", "ba_ci_low", "ba_ci_high", "accuracy", "n_test", "n_folds")},
                      "chance_balanced_accuracy": chance,
                      "above_chance": r.get("ba_ci_low") is not None and r["ba_ci_low"] > chance,
                      "C_per_fold": r.get("C_per_fold")})
    metrics = {"n_samples": len(rows), "n_groups": n_groups, "n_classes": len(classes), "n_folds": n_folds,
               "chance_balanced_accuracy": chance}
    for row in table:
        for k in ("macro_f1", "balanced_accuracy", "ba_ci_low", "ba_ci_high", "above_chance"):
            metrics[f"{k}_{row['features']}"] = row[k]
    pred = res["image"]["pred"]
    ok = pred >= 0
    per_class = pd.DataFrame({"class": classes, "n": np.bincount(y, minlength=len(classes)),
                              "f1_image": f1_score(y[ok], pred[ok], average=None, labels=range(len(classes)),
                                                   zero_division=0)})
    out = np.array([classes[c] if c >= 0 else None for c in pred], dtype=object)
    arts = {"predicted": Artifact("labels", out[ok], rows[ok],
                                  {"external": False, "source": f"out-of-fold predictions from '{image.name}'",
                                   "n_classes": len(classes)})}
    return StepResult(artifacts=arts, metrics=metrics,
                      tables={"predictability": pd.DataFrame(table), "per_class_f1": per_class})


def _predictability_cost(n, p, prm, budget) -> float:
    fits = budget.max_folds * (budget.inner_folds * len(agent.C_GRID) + 1) * 3
    return float(fits * (2e-8 * min(n, agent.PROBE_MAX_TRAIN) * p + 0.05) + 5)


register(ToolSpec(
    name="multimodal_predictability", origin="project3", source=SOURCE,
    summary="How well a categorical target is predicted from image features alone, another "
            "representation alone and both (early fusion): grouped nested cross-validation with a "
            "class-balanced logistic probe, macro-F1 and balanced accuracy against chance, with a "
            "group-bootstrap interval.",
    consumes=(Port("target", ("labels",), doc="categorical target (agent-made or external)"),
              Port("groups", ("groups",), doc="independent units held out together (e.g. patients)"),
              Port("image", ("embedding",), doc="image features"),
              Port("other", ("embedding", "normalized"), required=False, doc="a second representation")),
    produces=(Port("predicted", ("labels",), required=False, doc="out-of-fold predictions from image features"),),
    params=PredictParams, answers=("cross_modal_association", "predictive_modelling"),
    addresses=("nested_sampling", "class_imbalance"),
    criteria=(Criterion("balanced_accuracy_image", "max", None, "out-of-fold balanced accuracy from image "
                                                                 "features", selection=False),
              Criterion("macro_f1_image", "report", doc="out-of-fold macro-F1 from image features",
                        selection=False),
              Criterion("balanced_accuracy_other", "report", doc="from the second representation",
                        selection=False),
              Criterion("balanced_accuracy_both", "report", doc="from both (early fusion)", selection=False),
              Criterion("chance_balanced_accuracy", "report", doc="1 / number of classes", selection=False)),
    run=_run_predictability, cost=_predictability_cost,
))


# ---------------------------------------------------------------------------
# image-text agreement
# ---------------------------------------------------------------------------

class AgreementParams(ToolParams):
    aligners: list[Literal["zero_shot", "cca", "contrastive"]] | None = Field(
        None, description="alignments compared on validation pairs; None = every one that applies")
    n_random: int = Field(100, ge=10, le=1000, description="random partners each pair is compared with")
    flag_below: float = Field(agent.AGREEMENT_FLAG, ge=0.05, le=0.95,
                              description="pairs whose own partner beats fewer than this share of random "
                                          "partners are flagged")


def _run_agreement(ctx, inputs, p: AgreementParams) -> StepResult:
    a_art, b_art, split, groups = inputs["image"], inputs["text"], inputs.get("split"), inputs.get("groups")
    rows = np.intersect1d(a_art.rows, b_art.rows, assume_unique=True)
    A, B = dense(a_art.take(rows)), dense(b_art.take(rows))
    s, g = _values_on(rows, split), _values_on(rows, groups)
    tr, va, how = agent.holdout(len(rows), s, g, ctx.rng(21))
    ctx.decide("alignment training / validation pairs", how, "Alignments are fitted on training pairs and "
               "compared on validation pairs; no pair is scored by a model trained on it.", [],
               {"n_train": len(tr), "n_val": len(va)})
    ea, eb = a_art.meta.get("encoder"), b_art.meta.get("encoder")
    dual = bool(ea and eb and is_dual_pair(ea, eb) and A.shape[1] == B.shape[1])
    options, skipped = list(p.aligners or ["zero_shot", "cca", "contrastive"]), []
    if "zero_shot" in options and not dual:
        options.remove("zero_shot")
        skipped.append(("zero_shot", f"'{ea}' and '{eb}' are not the two towers of one dual encoder"))
    if "contrastive" in options and len(tr) < agent.MIN_PAIRS_CONTRASTIVE:
        options.remove("contrastive")
        skipped.append(("contrastive", f"{len(tr)} training pairs < {agent.MIN_PAIRS_CONTRASTIVE}"))
    options = options or ["cca"]
    hp, why_hp = agent.aligner_hyperparameters(len(tr), dual, ctx.budget.max_epochs, ctx.budget.train_seconds_per_fit)
    if "contrastive" in options:
        ctx.decide("contrastive head and training", ", ".join(f"{k}={v}" for k, v in hp.items()), why_hp, [], {})
    gallery = agent.cap(va, VAL_GALLERY, ctx.rng(22))
    fits, comparison = {}, []
    for m in options:
        check_time(ctx, "image-text agreement")
        ctx.log(f"    alignment {m} on {len(tr):,} training pairs")
        f, hist = agent.fit_alignment(m, A[tr], B[tr], A[va], B[va], hp if m == "contrastive" else None,
                                      ctx.seed, ctx.fit_device)
        fits[m] = f
        comparison.append({"aligner": m, "val_recall_at_10": recall_at_k(*f(A[gallery], B[gallery]), 10),
                           **{k: hist.get(k) for k in ("best_epoch", "epochs_run", "stopped_because", "k")
                              if k in hist}})
    best = max(comparison, key=lambda r: r["val_recall_at_10"])
    chance = min(1.0, 10 / len(gallery))
    ctx.decide("image-text alignment", best["aligner"],
               f"Highest validation recall@10 on {len(gallery)} held-out pairs: {best['val_recall_at_10']:.3f} "
               f"(chance {chance:.3f}).",
               [(r["aligner"], f"validation recall@10 {r['val_recall_at_10']:.3f}") for r in comparison
                if r is not best] + skipped, {"comparison": comparison})

    # score every pair with a model that did not see it: the training pairs by 2-fold cross-fitting
    m = best["aligner"]
    if m == "zero_shot":
        scorers = [(np.arange(len(rows)), fits[m])]
    else:
        h1, h2 = agent.halves(tr, g, ctx.rng(23))
        fit_half = [agent.fit_alignment(m, A[h], B[h], A[va], B[va], hp if m == "contrastive" else None,
                                        ctx.seed, ctx.fit_device)[0] for h in (h1, h2)]
        scorers = [(h2, fit_half[0]), (h1, fit_half[1]), (np.setdiff1d(np.arange(len(rows)), tr), fits[m])]
    pct, cos = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    rng = ctx.rng(24)
    for idx, f in scorers:
        keys = s[idx] if s is not None else np.zeros(len(idx), object)
        for key in pd.unique(keys):                    # random partners come from the same split
            sub = idx[keys == key]
            pct[sub], cos[sub] = agent.partner_percentiles(*f(A[sub], B[sub]), p.n_random, rng)
    flags = pct < p.flag_below
    metrics = {"aligner": m, "val_recall_at_10": best["val_recall_at_10"], "chance_recall_at_10": chance,
               "cross_fitted": m != "zero_shot", "n_pairs": len(rows), "n_flagged": int(flags.sum()),
               "flagged_fraction": float(flags.mean()), "median_percentile": float(np.nanmedian(pct))}
    if s is not None:
        for key, frac in pd.Series(flags).groupby(s).mean().items():
            metrics[f"flagged_fraction_{key or 'none'}"] = float(frac)
    arts = {"scores": Artifact("scores", np.column_stack([pct, cos]).astype(np.float32), rows,
                               {"columns": ["partner_percentile", "partner_cosine"],
                                "direction": "lower = the pair agrees less", "aligner": m}),
            "flags": Artifact("mask", flags, rows, {"reason": f"own partner beats fewer than {p.flag_below:.0%} "
                                                              f"of {p.n_random} random partners"})}
    return StepResult(artifacts=arts, metrics=metrics, tables={"aligner_comparison": pd.DataFrame(comparison)})


def _agreement_cost(n, p, prm, budget) -> float:
    return float(3 * budget.train_seconds_per_fit + 2e-6 * n * p + 10)


register(ToolSpec(
    name="multimodal_agreement", origin="project3", source=SOURCE,
    summary="Image-text agreement per pair: zero-shot, CCA and contrastive alignments are compared by "
            "validation recall@10 (label-free); the best scores each pair against random partners "
            "(cross-fitted), and pairs that agree less than an unrelated partner would are flagged.",
    consumes=(Port("image", ("embedding",), doc="image features"),
              Port("text", ("embedding",), doc="text features of the same samples"),
              Port("split", ("split",), required=False, doc="predefined split: train fits, val selects"),
              Port("groups", ("groups",), required=False, doc="groups kept apart between fit and validation")),
    produces=(Port("scores", ("scores",), doc="percentile among random partners, and cosine"),
              Port("flags", ("mask",), doc="pairs that agree less than a random partner")),
    params=AgreementParams, answers=("data_quality_audit", "cross_modal_association"),
    addresses=("text_noise", "label_noise"),
    criteria=(Criterion("val_recall_at_10", "max", None, "validation recall@10 of the chosen alignment"),
              Criterion("flagged_fraction", "report", doc="share of pairs flagged as disagreeing")),
    run=_run_agreement, cost=_agreement_cost,
))

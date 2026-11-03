"""The run's outputs: the record behind the prose, rule-based prose, the report (Markdown and
HTML), the methods summary, tables, exported results and the run manifest.

The record is the single source of numbers: the writer (LLM or template) reads it, the number
check compares the prose with it, and evaluation.json holds the same step metrics. Tables are
capped at ``TABLE_ROWS`` rows and exports at ``EXPORT_MAX_BYTES`` so outputs stay small; the
report says when something was cut. HTML styling is Project 3's.
"""

from __future__ import annotations

import json
import platform
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from .challenges import SEVERITY_ORDER
from .summaries import fmt, headline
from .tools import registry
from .tools.base import to_jsonable

TABLE_ROWS = 5000
EXPORT_MAX_BYTES = 5_000_000
PACKAGES = ["numpy", "scipy", "pandas", "scikit-learn", "torch", "transformers", "open-clip-torch", "timm", "anndata",
            "scanpy", "leidenalg", "harmonypy", "matplotlib", "pydantic", "claude-agent-sdk"]


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=1, default=str) + "\n")


def _cell(x) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def md_table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return ""
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_cell(fmt(r.get(c)) if not isinstance(r.get(c), str) else r.get(c))
                                     for c in cols) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------

def build_record(run: dict) -> dict:
    """Everything the prose may cite, compactly: objective, data, challenges, plan, decisions,
    step results, critic. ``run`` holds the run's objects (see orchestrator)."""
    prof, obj, reg, plan, records, log = (run[k] for k in ("profile", "objective", "register", "plan", "records",
                                                         "log"))
    challenges = [{"id": c.id, "severity": c.severity, "relevant": c.relevant, "stage": c.stage, "summary": c.summary,
                   "evidence": [{"metric": e.metric, "value": e.value} for e in c.evidence[:4]],
                   "handled_by": c.handled_by, "handling": c.handling, "residual_risk": c.residual_risk}
                  for c in reg.entries if c.status == "detected"]
    decisions = [{"step": d.step, "decision": d.decision, "chosen": d.chosen, "rationale": d.rationale[:500],
                  "source": d.source} for d in log.entries]
    results = {}
    for s in plan.steps:
        r = records.get(s.id) or {}
        scal = {k: v for k, v in (r.get("metrics") or {}).items() if isinstance(v, (int, float, str, bool))
                or v is None}
        results[s.id] = {"tool": s.tool, "answers": s.answers, "metrics": scal, "error": r.get("error"),
                         "skipped": r.get("skipped"),
                         "criteria": [{k: c.get(k) for k in ("metric", "goal", "threshold", "value", "passed",
                                                             "selection")} for c in r.get("criteria") or []]}
    return {"objective": obj.text, "mode": "quick" if run["quick"] else "full",
            "questions": [{"type": q.type, "phrase": q.phrase, "implied": q.implied,
                           "targets": q.targets} for q in obj.questions],
            "constraints": obj.constraints,
            "data": {"n_samples": prof.n_samples, "n_used": run.get("n_used"), "format": prof.format,
                     "modalities": [m.model_dump() for m in prof.modalities],
                     "groups": [{"column": g.column, "n_levels": g.n_levels, "samples_per_level": g.samples_per_level,
                                 "nested_in": g.nested_in} for g in prof.groups],
                     "split": prof.split.counts if prof.split else None},
            "challenges": challenges,
            "plan": {"source": plan.source, "steps": [{"id": s.id, "tool": s.tool, "purpose": s.purpose,
                                                        "params": s.params, "handles": s.addresses,
                                                        "validation_only": s.validation_only} for s in plan.steps],
                     "accepted": [a.model_dump() for a in plan.accepted]},
            "decisions": decisions, "results": results, "critic": run.get("critic") or {},
            "counts": {"questions": len([q for q in obj.questions if not q.implied]), "steps": len(plan.steps),
                       "relevant_challenges": len([c for c in challenges if c["relevant"]]),
                       "high_challenges": len([c for c in challenges if c["relevant"] and c["severity"] == "high"]),
                       "not_completed": len([r for r in results.values() if r["error"] or r["skipped"]])}}


# ---------------------------------------------------------------------------
# rule-based prose (used without an LLM, and for sections the LLM left empty)
# ---------------------------------------------------------------------------

def _answering(record: dict, qtype: str) -> list[str]:
    return [sid for sid, r in record["results"].items() if qtype in (r.get("answers") or [])]


def template_sections(record: dict) -> dict[str, str]:
    res, ch = record["results"], [c for c in record["challenges"] if c["relevant"]]
    high = [c for c in ch if c["severity"] == "high"]
    failed = [sid for sid, r in res.items() if r.get("error") or r.get("skipped")]
    qs = [q for q in record["questions"] if not q["implied"]]
    lines = []
    for q in qs:
        for sid in _answering(record, q["type"])[:2]:
            r = res[sid]
            if not r.get("error") and not r.get("skipped"):
                lines.append(f"{q['type'].replace('_', ' ')}: {sid} ({r['tool']}) - "
                             f"{headline(r['tool'], r['metrics'], 4)}")
    crit = record.get("critic") or {}
    exec_sum = (f"The objective was read as {len(qs)} question types ({', '.join(q['type'] for q in qs)}). The profile "
                f"raised {len(ch)} relevant challenges, {len(high)} of high severity"
                + (f" ({', '.join(sorted({c['id'].split(':')[0].replace('_', ' ') for c in high}))})" if high else "")
                + f". A plan of {len(res)} steps ({record['plan']['source']}) ran"
                + (f"; {len(failed)} did not complete ({', '.join(failed)})" if failed else "") + ".\n\n"
                + "\n".join(f"- {x}" for x in lines[:8])
                + (f"\n\nCritic: {crit.get('summary')}" if crit.get("summary") else ""))
    data = ("\n".join(f"- **{c['id']}** ({c['severity']}): {c['summary']}" for c in
                      sorted(ch, key=lambda c: -SEVERITY_ORDER[c["severity"]])[:12]) or "No relevant challenge.")
    plan_lines = [f"- **{d['decision']}**: {d['chosen']}. {d['rationale']}" for d in record["decisions"]
                  if d["step"] in ("plan", "objective") and d["source"] in ("heuristic", "llm")][:14]
    results = []
    for sid, r in res.items():
        if r.get("error"):
            results.append(f"- {sid} ({r['tool']}) failed: {r['error']}")
        elif r.get("skipped"):
            results.append(f"- {sid} ({r['tool']}) was skipped: it {r['skipped']}.")
        else:
            missed = [c["metric"] for c in r["criteria"] if c.get("passed") is False]
            results.append(f"- {sid} ({r['tool']}): {headline(r['tool'], r['metrics'], 5)}"
                           + (f"; criteria missed: {', '.join(missed)}" if missed else ""))
    limits = [f"- {a['challenge']}: accepted ({a['reason']})" for a in record["plan"]["accepted"]]
    limits += [f"- {c['id']}: {c['residual_risk']}" for c in ch if c.get("residual_risk")
               and not str(c["residual_risk"]).startswith("low") and c["severity"] in ("high", "medium")][:10]
    if record["mode"] == "quick":
        limits.append("- Quick mode: small budgets (few subsamples, permutations and epochs); numbers are a smoke "
                      "test, not the analysis.")
    limits += [f"- {sid} did not complete" for sid in failed]
    return {"executive_summary": exec_sum, "data_and_challenges": data,
            "plan_explained": "\n".join(plan_lines) or "The rule-based plan ran as proposed.",
            "results_interpretation": "\n".join(results),
            "critic_and_revision": crit.get("summary") or "The critic found nothing to revise.",
            "limitations": "\n".join(limits) or "- None recorded."}


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

def write_tables(results: dict, out: Path) -> dict[str, dict]:
    """Every step's tables as CSV (first TABLE_ROWS rows)."""
    index = {}
    for sid, r in results.items():
        for name, t in r.tables.items():
            if not isinstance(t, pd.DataFrame) or t.empty:
                continue
            path = out / "tables" / f"{sid}__{name}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            t.head(TABLE_ROWS).to_csv(path, index=False)
            index[f"{sid}__{name}"] = {"path": f"tables/{path.name}", "rows": int(len(t)),
                                       "truncated": bool(len(t) > TABLE_ROWS)}
    return index


def export_results(ws, plan, results: dict, out: Path) -> dict[str, str]:
    """Key per-sample results: discovered groups, quality flags, duplicate clusters, markers and
    target tables, each capped at EXPORT_MAX_BYTES."""
    files, ids = {}, ws.sample_ids
    for s in plan.steps:
        r = results.get(s.id)
        if r is None:
            continue
        frames = {}
        for port, name in r.outputs.items():
            a = ws.get(name)
            if a.kind == "labels" and s.tool in ("spatial_domains", "cluster", "multimodal_predictability"):
                frames[f"{s.id}_{port}"] = pd.DataFrame({"sample": ids[a.rows], "label": np.asarray(a.data)})
            elif a.kind == "mask" and s.tool.startswith("qc_"):
                bad = a.rows[~np.asarray(a.data, bool)]
                frames[f"{s.id}_removed"] = pd.DataFrame({"sample": ids[bad]})
            elif a.kind == "groups" and s.tool == "near_duplicates":
                v = np.asarray(a.data).astype(str)
                keep = np.char.startswith(v.astype("U"), "D")
                frames[f"{s.id}_clusters"] = pd.DataFrame({"sample": ids[a.rows[keep]], "cluster": v[keep]})
        if s.tool == "differential_expression" and "markers" in r.tables:
            t = r.tables["markers"]
            frames[f"{s.id}_significant"] = t[t["significant"]] if "significant" in t else t
        if s.tool == "predict_targets" and "target_predictability" in r.tables:
            frames[f"{s.id}_targets"] = r.tables["target_predictability"]
        for name, df in frames.items():
            path = out / "results" / f"{name}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            text = df.to_csv(index=False)
            if len(text.encode()) > EXPORT_MAX_BYTES:
                files[name] = f"not exported: {len(text.encode()) / 1e6:.1f} MB > {EXPORT_MAX_BYTES / 1e6:.0f} MB"
                continue
            path.write_text(text)
            files[name] = f"results/{path.name}"
    return files


def methods_summary(plan, records: dict) -> str:
    lines = ["# Methods summary", "", "Each step, the tool it ran (and where the tool comes from), its non-default "
             "settings and what it decided. Full rationale is in decision_log.json.", ""]
    for i, s in enumerate(plan.steps, 1):
        spec = registry.get(s.tool)
        r = records.get(s.id) or {}
        defaults = spec.params().model_dump()
        set_ = {k: v for k, v in (r.get("params") or s.params).items() if defaults.get(k) != v}
        lines.append(f"**{i}. {s.id}** - `{s.tool}` ({spec.origin}; {spec.source}). {spec.summary}")
        if set_:
            lines.append(f"  Settings: {', '.join(f'{k}={v}' for k, v in set_.items())}.")
        for d in (r.get("decisions") or [])[:4]:
            lines.append(f"  - {d['decision']}: {d['chosen']}")
        if r.get("error"):
            lines.append(f"  - failed: {r['error']}")
        lines.append("")
    return "\n".join(lines)


def manifest(run: dict, started: float) -> dict:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    cfg = run["config"]
    return {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.perf_counter() - started, 1),
            "python": platform.python_version(), "platform": platform.platform(), "packages": versions,
            "hardware": run["hardware"], "seed": cfg.seed, "quick": cfg.quick, "budget": run["budget"].model_dump(),
            "criteria": cfg.criteria.model_dump(), "llm": run["client"].summary() | {"calls": len(run["client"].calls)},
            "planner": run["plan"].source, "n_samples_used": run.get("n_used"),
            "command": run.get("command")}


CSS = """
:root { color-scheme: light; --bg: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e;
  --muted: #898781; --rule: #e1e0d9; --accent: #2a78d6; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --bg: #0d0d0d;
  --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781; --rule: #2c2c2a; --accent: #3987e5; } }
:root[data-theme="dark"] { color-scheme: dark; --bg: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7;
  --muted: #898781; --rule: #2c2c2a; --accent: #3987e5; }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
  margin: 0; padding: 24px 16px; }
main { max-width: 980px; margin: 0 auto; background: var(--surface); padding: 24px 28px; border-radius: 8px; }
h1 { font-size: 1.6em; line-height: 1.25; } h2 { margin-top: 1.8em; border-bottom: 1px solid var(--rule);
  padding-bottom: 4px; } h3 { margin-top: 1.4em; }
a { color: var(--accent); } code { font-size: 0.92em; }
table { border-collapse: collapse; font-size: 13px; margin: 10px 0 18px; display: block; overflow-x: auto; }
th, td { border-bottom: 1px solid var(--rule); padding: 4px 10px; text-align: left; vertical-align: top;
  font-variant-numeric: tabular-nums; } th { color: var(--ink2); font-weight: 600; }
img { max-width: 100%; height: auto; background: #fcfcfb; border-radius: 6px; }
blockquote { margin: 0; padding: 4px 14px; border-left: 3px solid var(--rule); color: var(--ink2); }
@media (max-width: 640px) { body { padding: 0; } main { padding: 16px; border-radius: 0; } }
"""


def to_html(md_text: str, title: str) -> str:
    import markdown

    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    return (f"<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>{title}</title>"
            f"<style>{CSS}</style></head><body><main>\n{body}\n</main></body></html>\n")


def report_markdown(run: dict, record: dict, prose: dict, figs: dict, tables: dict, exports: dict,
                    man: dict) -> str:
    plan, reg, obj = run["plan"], run["register"], run["objective"]
    sec = prose["sections"]
    llm = man["llm"]
    lines = ["# Analysis report", "",
             f"{record['data']['format']} data, {record['data']['n_samples']:,} samples"
             + (f" ({run['n_used']:,} used in quick mode)" if run.get("n_used") else "")
             + f"; {record['mode']} mode; plan by the {plan.source} planner; LLM: "
             + (f"{', '.join(llm['models_used']) or llm['model_requested']}, {llm['n_calls']} calls, "
                f"${llm['recorded_cost_usd']:.2f}"
                + (f" ({llm['n_from_cache']} replayed from the run's transcript at no new cost)"
                   if llm["n_from_cache"] else "") if llm["n_calls"] else "not used") + f"; {man['created']}.", "",
             "## Objective", "", f"> {obj.text}", "",
             md_table([{"question": q.type.replace("_", " ") + (" (implied)" if q.implied else ""),
                        "read from": q.phrase,
                        "about": "; ".join(f"{k}: {', '.join(v)}" for k, v in q.targets.items() if v)}
                       for q in obj.questions], ["question", "read from", "about"])]
    for k, v in obj.constraints.items():
        lines += ["", f"Fixed before any result ({k.replace('_', ' ')}): {v}."]
    lines += ["", "## Executive summary", "", sec["executive_summary"], "", "## Data and challenges", "",
              sec["data_and_challenges"], ""]
    if "challenges" in figs:
        lines += [f"![{figs['challenges']['caption']}]({figs['challenges']['path']})", ""]
    rows = [{"challenge": c.id, "severity": c.severity, "evidence": "; ".join(
                f"{e.metric} {fmt(e.value)}" for e in c.evidence[:2]),
             "handled by": ", ".join(c.handled_by) or "-", "remaining risk": c.residual_risk or "-"}
            for c in sorted([c for c in reg.entries if c.status == "detected" and c.relevant],
                            key=lambda c: (-SEVERITY_ORDER[c.severity], c.id))]
    lines += ["Relevant challenges (all checks, including those that found nothing, are in "
              "challenge_register.md):", "", md_table(rows, ["challenge", "severity", "evidence", "handled by",
                                                            "remaining risk"]), "",
              "## Plan", "", sec["plan_explained"], "",
              md_table([{"#": i, "step": s.id + (" (validation only)" if s.validation_only else ""), "tool": s.tool,
                         "purpose": s.purpose, "handles": ", ".join(s.addresses) or "-"}
                        for i, s in enumerate(plan.steps, 1)], ["#", "step", "tool", "purpose", "handles"])]
    if plan.accepted:
        lines += ["", "Accepted limitations:"] + [f"- {a.challenge}: {a.reason}" for a in plan.accepted]
    lines += ["", "## Results", "", sec["results_interpretation"], ""]
    records = run["records"]
    shown: dict[str, str] = {}                                # step -> section it is reported in
    for q in obj.questions:
        sids = [s.id for s in plan.steps if q.type in s.answers]
        if not sids:
            continue
        title = q.type.replace("_", " ").capitalize()
        lines += [f"### {title}", ""]
        for sid in sids:
            if sid in shown:
                lines += [f"- **{sid}**: reported under {shown[sid]} above."]
                continue
            shown[sid] = title
            r = records.get(sid) or {}
            tool = next(s.tool for s in plan.steps if s.id == sid)
            if r.get("error") or r.get("skipped"):
                what = f"failed: {r['error']}" if r.get("error") else f"skipped: {r['skipped']}"
                lines += [f"- **{sid}** ({tool}): {what}"]
                continue
            crit = "; ".join(f"{c['metric']} {fmt(c.get('value'))}" + ("" if c.get("passed") is None else
                                                                     (" (met)" if c["passed"] else " (missed)"))
                             + ("" if c.get("selection") else " [report only]") for c in r.get("criteria", []))
            lines += [f"- **{sid}** ({tool}): {headline(tool, r.get('metrics', {}), 6)}",
                      f"  - criteria: {crit or 'none declared'}"]
            for f in figs.values():
                if f.get("step") == sid and f.get("path"):
                    lines += ["", f"![{f['caption']}]({f['path']})", ""]
        lines.append("")
    lines += ["## Critic and revision", "", sec["critic_and_revision"], ""]
    crit = record.get("critic") or {}
    if crit.get("revision"):
        rv = crit["revision"]
        lines += [f"Revision tried: {rv.get('action')} on {rv.get('step')} {rv.get('params') or rv.get('tool') or ''}; "
                  f"judged by {rv.get('criterion_step')}.{rv.get('criterion')} ({rv.get('goal')}): "
                  f"{crit.get('verdict', '')}.", ""]
    lines += ["## Limitations", "", sec["limitations"], "", "## Reproducibility", "",
              f"- Seed {man['seed']}; {'quick' if man['quick'] else 'full'} budgets; device "
              f"{man['hardware'].get('device')}; Python {man['python']}; run time {man['seconds']:,.0f} s.",
              f"- Planner: {man['planner']}; LLM calls {llm['n_calls']} ({llm['n_from_cache']} from cache); replay "
              "with `--replay` on this run's directory or transcript.",
              f"- Files: profile.json, objective.json, challenge_register.(json|md), analysis_plan.(json|md), "
              f"evaluation.json, decision_log.json, methods_summary.md, {len(tables)} tables in tables/, "
              f"{len([f for f in figs.values() if f.get('path')])} figures in figures/, "
              f"{len([v for v in exports.values() if v.startswith('results/')])} result files in results/.", ""]
    check = prose["number_check"]
    lines += ["## Numbers not found in the outputs", "",
              f"{check['n_checked']} numbers in the prose were checked against the computed record "
              f"({prose['source']}); {check['n_unmatched']} not found."]
    lines += [f"- {u['section']}: {u['number']} in \"{u['context']}\"" for u in check["unmatched"][:30]]
    return "\n".join(lines) + "\n"

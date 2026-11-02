"""The agent loop, from a dataset and an objective to a finished report.

    load -> profile -> objective -> challenge register -> plan (rules, then the LLM's review,
    validated) -> execute and evaluate each step -> critic (at most one revision, kept only if
    its label-free criterion improves) -> register update -> figures -> prose -> report

Every choice lands in ``decision_log.json`` as it is made, so even a run that stops early
leaves a record of how far it got and why.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .challenges import assess, render_markdown, update_after_run
from .config import load_config
from .critic import apply_revision, critique, downstream, find_triggers, keep_revision
from .data.loaders import load_path
from .decision_log import DecisionLog
from .executor import Executor, subsample
from .figures import make_figures
from .hardware import detect
from .llm import make_client
from .objective import interpret, objective_decisions
from .planner import choose_plan, plan_analysis, render_plan_markdown, validate
from .planner.refs import step_costs
from .profiler import profile_dataset
from .report import (build_record, dump, export_results, manifest, methods_summary, report_markdown,
                     template_sections, to_html, write_tables)
from .tools.ingest import workspace_from_dataset
from .writer import write_sections


def critic_loop(plan, records, ex, prof, reg, obj, budget, client, log, max_revisions: int):
    """At most one revision: apply, re-run what it touches, keep it only if its criterion improves."""
    triggers = find_triggers(plan, records, reg, obj)
    info = {"triggers": [t.to_record() for t in triggers], "revision": None}
    if max_revisions < 1:
        info["summary"] = "The critic is switched off for this run."
        return plan, records, info
    spent = sum(r.get("seconds") or 0.0 for r in records.values())
    rev = critique(plan, records, obj, reg, client, log, budget_left=budget.total_seconds - spent)
    if rev is None:
        info["summary"] = (f"{len(triggers)} trigger(s) ({', '.join(t.kind for t in triggers) or 'none'}); "
                           "no revision was tried.")
        return plan, records, info
    info["revision"] = rev.model_dump()
    revised = apply_revision(plan, rev)
    checked, report = validate(revised, prof, reg, obj, budget)
    if checked is None or not report.valid:
        why = "; ".join(i.message for i in report.errors()[:4])
        log.add("critic", "revision", "not run", f"the revised plan fails validation: {why}", source="validator")
        info.update(verdict="not run (invalid)", summary=f"The revision of {rev.step} failed validation: {why}")
        return plan, records, info
    touched = [rev.step] + downstream(checked, rev.step)
    cost = sum(v for k, v in step_costs(checked, prof, budget).items() if k in touched)
    if spent + cost > budget.total_seconds:
        log.add("critic", "revision", "not run", f"re-running {len(touched)} steps (~{cost:,.0f} s) would exceed the "
                f"budget ({spent:,.0f} s used of {budget.total_seconds:,.0f})", source="system")
        info.update(verdict="not run (budget)", summary=f"The revision of {rev.step} would exceed the budget.")
        return plan, records, info
    new = ex.run(checked, "revision")
    before = ((records.get(rev.criterion_step) or {}).get("metrics") or {}).get(rev.criterion)
    after = ((new.get(rev.criterion_step) or {}).get("metrics") or {}).get(rev.criterion)
    keep, why = keep_revision(before, after, rev.goal)
    info.update(before=before, after=after, kept=keep, verdict=f"{'kept' if keep else 'discarded'}: {why}",
                rerun=touched)
    log.add("critic", "revision kept" if keep else "revision discarded", f"{rev.action} on {rev.step}",
            f"{rev.criterion_step}.{rev.criterion} {why}; re-ran {', '.join(touched)}", [],
            {"before": before, "after": after, "criterion": rev.criterion}, source="critic")
    info["summary"] = (f"Trigger {rev.trigger}: {rev.reason}. {rev.action} on {rev.step} "
                       f"({rev.params or rev.tool or ''}); {rev.criterion_step}.{rev.criterion} {why}; the "
                       f"revision was {'kept' if keep else 'discarded'}.")
    if keep:
        return checked, new, info
    ex.run(plan, "restore")                                  # cached: puts the original artifacts back
    return plan, records, info


def run(data_path, objective: str, out_dir, **overrides) -> dict:
    started = time.perf_counter()
    quick = bool(overrides.pop("quick", False))
    no_llm = bool(overrides.pop("no_llm", False))
    replay = overrides.pop("replay", None)
    label_column, group_column = overrides.pop("label_column", None), overrides.pop("group_column", None)
    config_path, device = overrides.pop("config", None), overrides.pop("device", None)
    cfg = load_config(config_path, quick=quick, **{k: v for k, v in overrides.items() if v is not None})
    want = device or cfg.device
    hw = detect(None if want in (None, "auto") else want)
    budget = cfg.budget(hw.device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_dir = Path(cfg.runs_dir) / out.name
    log = DecisionLog(out / "decision_log.json")
    log.add("system", "hardware", hw.device, f"{hw.to_record()}; {'quick' if quick else 'full'} budgets",
            source="system")

    ds = load_path(data_path, label_column, group_column)
    ws = workspace_from_dataset(ds)
    n_used = None
    if budget.max_samples:
        small = subsample(ws, budget.max_samples, cfg.seed, log)
        if small is not ws:
            ws = small
            n_used = len({int(r) for n in ws.names() for r in (ws.get(n).rows if ws.get(n).rows is not None
                                                                else [])})
    prof = profile_dataset(ds, ws, cfg, budget, cache_dir=Path(cfg.cache_dir))
    if n_used:
        prof.notes.append(f"quick mode: {n_used:,} of {ds.n:,} samples used")
    obj = interpret(objective, prof, cfg.criteria)
    log.extend(objective_decisions(obj))
    reg = assess(prof, obj.types())
    high = [c.id for c in reg.detected("high")]
    log.add("challenges", "challenge register", f"{len(reg.detected('low'))} relevant challenges, {len(high)} high",
            "; ".join(high) or "none high", [], {"detected": [c.id for c in reg.entries if c.status == "detected"]},
            source="heuristic")
    client = make_client(cfg.llm, run_dir / "llm_cache", replay, enabled=not no_llm)
    proposal, _, decisions = plan_analysis(prof, reg, obj, budget)
    log.extend(decisions)
    plan, vreport, obj, reg = choose_plan(prof, reg, obj, budget, proposal, decisions, client, log)
    if not vreport.valid:
        raise RuntimeError("no valid plan: " + "; ".join(i.message for i in vreport.errors()[:6]))
    dump(plan.model_dump(), out / "analysis_plan.json")

    ex = Executor(ws, cfg, budget, hw.device, log, out)
    records = ex.run(plan, "run")
    plan, records, critic_info = critic_loop(plan, records, ex, prof, reg, obj, budget, client, log,
                                             cfg.max_critic_revisions)
    reg = update_after_run(reg, plan, records)
    figs = make_figures(ws, plan, ex.results, reg, out, cfg.seed, cfg.criteria.reliable_min_auroc_lower)
    tables = write_tables({k: v for k, v in ex.results.items() if k in records}, out)
    exports = export_results(ws, plan, ex.results, out)
    runinfo = {"profile": prof, "objective": obj, "register": reg, "plan": plan, "records": records, "log": log,
               "quick": quick, "n_used": n_used, "critic": critic_info, "config": cfg, "budget": budget,
               "hardware": hw.to_record(), "client": client, "command": " ".join(sys.argv)}
    record = build_record(runinfo)
    prose = write_sections(record, client, log, template_sections)
    man = manifest(runinfo, started)
    md = report_markdown(runinfo, record, prose, figs, tables, exports, man)
    (out / "report.md").write_text(md)
    (out / "report.html").write_text(to_html(md, "Analysis report"))
    (out / "methods_summary.md").write_text(methods_summary(plan, records))
    (out / "analysis_plan.md").write_text(render_plan_markdown(plan, vreport, [d for d in log.entries
                                                                              if d.step in ("plan", "validator")]))
    (out / "challenge_register.md").write_text(render_markdown(reg))
    for name, obj_ in (("profile", prof), ("objective", obj), ("challenge_register", reg), ("analysis_plan", plan),
                       ("validation", vreport)):
        dump(obj_.model_dump(), out / f"{name}.json")
    dump({sid: {k: r.get(k) for k in ("tool", "metrics", "criteria", "seconds", "cached", "error", "skipped")}
          for sid, r in records.items()}, out / "evaluation.json")
    dump({"record": record, "prose_source": prose["source"], "number_check": prose["number_check"],
          "figures": figs, "tables": tables, "exports": exports}, out / "results_index.json")
    dump(man, out / "run_manifest.json")
    summary = {k: v for k, v in client.summary().items() if k != "calls"}
    dump({"summary": summary, "calls": client.records}, out / "llm_transcript.json")
    failed = [sid for sid, r in records.items() if r.get("error") or r.get("skipped")]
    return {"out_dir": str(out), "planner": plan.source, "steps": len(plan.steps), "failed": failed,
            "critic": critic_info.get("verdict") or critic_info.get("summary"), "llm_calls": len(client.calls),
            "llm_cost_usd": round(client.spent_usd, 4), "seconds": round(time.perf_counter() - started, 1),
            "numbers_unmatched": prose["number_check"]["n_unmatched"]}

"""Command line: one command per dataset and objective.

    bioagent run --data data/<dataset> --objective "..." --out outputs/generated_report_1
    bioagent run ... --quick          # smoke test: small subsample, seconds per step
    bioagent run ... --no-llm         # rule-based planner, critic and writer (no Claude CLI needed)
    bioagent run ... --replay runs/<run>/llm_cache   # reproduce LLM decisions from cache
    bioagent profile --data data/<dataset> [--out DIR]   # measure the data, list its challenges
    bioagent schema [--write]         # JSON schemas of the profile and the challenge register
    bioagent hardware                 # report detected device, disk and budgets
    bioagent tools [--json | --name TOOL]            # the tool registry: the planner's menu
"""

from __future__ import annotations

import argparse
import json


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="bioagent", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the agent on one dataset and objective")
    r.add_argument("--data", required=True, help="dataset file or directory")
    r.add_argument("--objective", required=True, help="free-text objective, in quotes")
    r.add_argument("--out", required=True, help="output directory for the report and artefacts")
    r.add_argument("--config", help="YAML with budgets/seed/LLM settings (never analysis choices)")
    r.add_argument("--quick", action="store_true", help="smoke-test budgets")
    r.add_argument("--no-llm", action="store_true", help="rule-based planner/critic/writer only")
    r.add_argument("--replay", help="directory of cached LLM responses to replay")
    r.add_argument("--label-column", help="override the detected label column")
    r.add_argument("--group-column", help="override the detected group (slide/patient/source) column")
    r.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"])
    r.add_argument("--seed", type=int)
    r.add_argument("--model", help="LLM model id (default from config)")
    sub.add_parser("hardware", help="print detected hardware and budgets")
    pr = sub.add_parser("profile", help="measure a dataset and list its analytical challenges")
    pr.add_argument("--data", required=True, help="dataset file or directory")
    pr.add_argument("--out", help="directory for profile.json, challenge_register.json and challenge_register.md")
    pr.add_argument("--quick", action="store_true", help="quick-mode budgets for the cost estimates")
    pr.add_argument("--json", action="store_true", help="print the profile as JSON instead of the register")
    pr.add_argument("--label-column")
    pr.add_argument("--group-column")
    sc = sub.add_parser("schema", help="JSON schemas of the profile and the challenge register")
    sc.add_argument("--write", action="store_true", help="(re)write schemas/*.schema.json")
    t = sub.add_parser("tools", help="list the tools the planner can use")
    t.add_argument("--json", action="store_true", help="every tool's full declaration as JSON")
    t.add_argument("--name", help="the full declaration of one tool")
    args = ap.parse_args(argv)

    if args.cmd == "tools":
        from .tools import registry

        if args.name or args.json:
            spec = registry.get(args.name).describe() if args.name else registry.describe_all()
            print(json.dumps(spec, indent=2, default=str))
            return
        for row in registry.table():
            print(f"{row['tool']}  [{row['origin']}]")
            print(f"    consumes:  {row['consumes']}")
            print(f"    produces:  {row['produces']}")
            print(f"    answers:   {row['answers']}")
            print(f"    addresses: {row['addresses']}")
        return

    if args.cmd == "hardware":
        from .config import load_config
        from .hardware import detect

        hw = detect()
        cfg = load_config()
        print(json.dumps({"hardware": hw.to_record(),
                          "full_budget": cfg.budget(hw.device).model_dump(),
                          "quick_budget": load_config(quick=True).budget(hw.device).model_dump(),
                          "fit_device": cfg.fit_device,
                          "llm": {"backend": "claude CLI (claude-agent-sdk)", "model": cfg.llm.model,
                                  "max_output_tokens": cfg.llm.max_output_tokens,
                                  "max_budget_usd": cfg.llm.max_budget_usd}},
                         indent=2))
        return

    if args.cmd == "profile":
        from pathlib import Path

        from .challenges import assess, render_markdown
        from .config import load_config
        from .data.loaders import load_path
        from .profiler import profile_dataset

        cfg = load_config(quick=args.quick)
        ds = load_path(args.data, args.label_column, args.group_column)
        prof = profile_dataset(ds, config=cfg, cache_dir=Path(cfg.cache_dir))
        reg = assess(prof)
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / "profile.json").write_text(prof.model_dump_json(indent=2) + "\n")
            (out / "challenge_register.json").write_text(reg.model_dump_json(indent=2) + "\n")
            (out / "challenge_register.md").write_text(render_markdown(reg))
        print(prof.model_dump_json(indent=2) if args.json else render_markdown(reg))
        print(f"profiled in {prof.seconds:.1f} s")
        return

    if args.cmd == "schema":
        from .config import REPO_ROOT
        from .schema import json_schemas

        schemas = json_schemas()
        if args.write:
            d = REPO_ROOT / "schemas"
            d.mkdir(exist_ok=True)
            for name, sch in schemas.items():
                (d / f"{name}.schema.json").write_text(json.dumps(sch, indent=2) + "\n")
                print(f"wrote schemas/{name}.schema.json")
        else:
            print(json.dumps(schemas, indent=2))
        return

    from .orchestrator import run

    overrides = {"quick": args.quick, "no_llm": args.no_llm, "replay": args.replay,
                 "label_column": args.label_column, "group_column": args.group_column,
                 "device": args.device, "seed": args.seed, "config": args.config}
    if args.model:
        overrides["llm"] = {"model": args.model}
    result = run(args.data, args.objective, args.out,
                 **{k: v for k, v in overrides.items() if v not in (None, False)})
    print(f"\nreport: {result['out_dir']}/report.html")


if __name__ == "__main__":
    main()

"""Scaffold: device detection, budgets and configuration."""

from analysis_agent import hardware
from analysis_agent.cli import main
from analysis_agent.config import RunConfig, load_config


def test_detect_returns_known_device():
    info = hardware.detect()
    assert info.device in {"cuda", "mps", "cpu"}
    assert hardware.detect("cpu").device == "cpu"
    assert info.memory_gb is None or info.memory_gb > 0


def test_quick_budget_is_smaller_and_cpu_is_scaled():
    full = load_config().budget("mps")
    quick = load_config(quick=True).budget("mps")
    cpu = load_config().budget("cpu")
    assert quick.total_seconds < full.total_seconds and quick.step_seconds < full.step_seconds
    assert quick.train_seconds_per_fit < full.train_seconds_per_fit
    assert quick.max_epochs <= 5 and quick.max_samples and quick.bootstrap_reps < full.bootstrap_reps
    # a slower device trains each fit for less time; the run-level caps stay the same
    assert cpu.train_seconds_per_fit < full.train_seconds_per_fit
    assert cpu.total_seconds == full.total_seconds and cpu.device == "cpu"


def test_config_overrides_and_llm_defaults():
    cfg = load_config(seed=1, llm={"enabled": False}, criteria={"fdr": 0.1})
    assert cfg.seed == 1 and cfg.llm.enabled is False and cfg.criteria.fdr == 0.1
    assert cfg.llm.model == RunConfig().llm.model                  # untouched keys keep defaults
    default = load_config().llm
    assert default.model and default.max_output_tokens <= 16_000
    assert set(default.effort) == {"objective", "planner", "critic", "writer"}


def test_default_yaml_matches_the_code_defaults():
    assert load_config().model_dump() == RunConfig().model_dump()


def test_config_holds_no_analysis_choices():
    """Budgets, seeds, LLM settings and reporting thresholds only: tools, methods and their
    parameters come from the profile, the challenge register and the objective."""
    def keys(d):
        for k, v in d.items():
            yield k
            if isinstance(v, dict):
                yield from keys(v)

    banned = {"tool", "tools", "method", "methods", "encoder", "encoders", "steps", "plan",
              "objective_text", "strategy", "correction", "normalization", "n_clusters",
              "resolution", "k", "loss", "model_name", "target", "targets"}
    assert not set(keys(load_config().model_dump())) & banned


def test_hardware_command_prints_json(capsys):
    main(["hardware"])
    out = capsys.readouterr().out
    assert '"device"' in out and '"full_budget"' in out and '"quick_budget"' in out

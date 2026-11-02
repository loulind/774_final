"""A general biomedical analysis agent.

Given a dataset and a free-text objective it profiles the data, turns the objective
into scientific questions, lists the analytical challenges the data pose (the
challenge register), plans an analysis across a registry of tools -- including the
Project 1-3 agents -- runs it, lets a critic order at most one revision, and writes
a report.

    from analysis_agent import run_agent
    run_agent("data/<dataset>", "<objective>", "outputs/<run>", quick=True)
"""

__version__ = "0.1.0"


def run_agent(data_path, objective: str, out_dir, **overrides) -> dict:
    """Run the agent end to end on one dataset and objective.

    ``overrides`` are run settings, not analysis choices: ``quick``, ``no_llm``,
    ``replay``, ``seed``, ``device``, ``config`` (a YAML file), ``label_column``,
    ``group_column``, or nested sections such as ``llm={"model": ...}``.
    Returns a summary with the output directory and the files written.
    """
    from .orchestrator import run

    return run(data_path, objective, out_dir, **overrides)


__all__ = ["run_agent", "__version__"]

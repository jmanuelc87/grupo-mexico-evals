"""Convert a deepeval run file (``DEEPEVAL_RESULTS_FOLDER``) into a standalone HTML report."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

from grupo_mexico_evals.agent import CLAUDE_MODELS, GEMINI_MODELS, OPENAI_CHAT_MODELS

RESULTS_DIR = Path("reports")
TITLE_QUESTION_CHARS = 80  # question text kept in a test case's dropdown title
KNOWN_SPECS = [
    f"{provider}:{model}"
    for provider, models in (
        ("openai", OPENAI_CHAT_MODELS),
        ("claude", CLAUDE_MODELS),
        ("gemini", GEMINI_MODELS),
    )
    for model in models
]

CSS = """
body { font-family: system-ui, sans-serif; margin: 2rem; color: #1f2933; }
h1, h2 { font-weight: 600; }
table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; font-size: 0.9rem; }
th, td { border: 1px solid #d9e2ec; padding: 0.4rem 0.6rem; text-align: left; vertical-align: top; }
th { background: #f0f4f8; }
.pass { color: #0f7b3f; font-weight: 600; }
.fail { color: #b42318; font-weight: 600; }
details { margin: 0.3rem 0; }
pre { white-space: pre-wrap; background: #f8fafc; padding: 0.5rem; margin: 0.3rem 0; }
.meta { color: #616e7c; font-size: 0.9rem; }
"""


def latest_run(results_dir: Path = RESULTS_DIR) -> Path:
    """The most recently written run file in ``results_dir``."""
    files = [p for p in results_dir.iterdir() if p.is_file() and p.suffix != ".html"]
    if not files:
        raise FileNotFoundError(f"no run files in {results_dir}")
    return max(files, key=lambda p: p.stat().st_mtime)


def split_case(case: dict) -> tuple[str, str]:
    """``(<provider>:<model>, question)`` for a test case.

    The pytest id is ``test_rag_quality[<provider>:<model>-<question>]``. The question is the case's
    input, so the model is whatever comes before it. Ids escape non-ASCII characters, so when the
    input is not found in the id, the model is matched against the known model IDs instead.
    """
    name = case["name"]
    inner = name[name.find("[") + 1 : -1] if "[" in name else name
    question = case.get("input") or ""
    if question and inner.endswith("-" + question):
        spec = inner[: -len(question) - 1]
    else:
        spec = max(
            (s for s in KNOWN_SPECS if inner.startswith(s + "-")), key=len, default=inner
        )
    return spec, question


def case_title(case: dict) -> str:
    """Dropdown title: ``<provider>:<model> · <question>``, with the question shortened."""
    spec, question = split_case(case)
    question = " ".join(question.split())
    if len(question) > TITLE_QUESTION_CHARS:
        question = question[: TITLE_QUESTION_CHARS - 1].rstrip() + "…"
    return f"{spec} · {question}"


def _score(value) -> str:
    return "–" if value is None else f"{value:.2f}"


def _status(success: bool) -> str:
    cls, text = ("pass", "PASS") if success else ("fail", "FAIL")
    return f'<span class="{cls}">{text}</span>'


def _duration(value) -> str:
    return "–" if value is None else f"{value:.1f}s"


def _money(value) -> str:
    return "–" if value is None else f"${value:.4f}"


def _summary_table(cases: list[dict]) -> str:
    """One row per model: pass rate, avg time, judge cost and mean score for each metric."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        by_model[split_case(case)[0]].append(case)

    metric_names: list[str] = []
    for case in cases:
        for metric in case.get("metricsData") or []:
            if metric["name"] not in metric_names:
                metric_names.append(metric["name"])

    header = "".join(f"<th>{html.escape(m)} (mean)</th>" for m in metric_names)
    rows = []
    for model, model_cases in by_model.items():
        passed = sum(1 for c in model_cases if c["success"])
        durations = [c["runDuration"] for c in model_cases if c.get("runDuration") is not None]
        costs = [c["evaluationCost"] for c in model_cases if c.get("evaluationCost") is not None]
        avg_time = sum(durations) / len(durations) if durations else None
        total_cost = sum(costs) if costs else None
        avg_cost = sum(costs) / len(costs) if costs else None
        cells = []
        for name in metric_names:
            scores = [
                m["score"]
                for c in model_cases
                for m in c.get("metricsData") or []
                if m["name"] == name and m.get("score") is not None
            ]
            mean = sum(scores) / len(scores) if scores else None
            cells.append(f"<td>{_score(mean)}</td>")
        rows.append(
            f"<tr><td>{html.escape(model)}</td>"
            f"<td>{passed}/{len(model_cases)}</td>"
            f"<td>{_duration(avg_time)}</td>"
            f"<td>{_money(total_cost)}</td>"
            f"<td>{_money(avg_cost)}</td>"
            f"{''.join(cells)}</tr>"
        )
    return (
        "<table><thead><tr><th>Model</th><th>Passed</th><th>Avg time</th>"
        "<th>Total cost</th><th>Avg cost</th>"
        f"{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _case_section(case: dict) -> str:
    metrics = "".join(
        f"<tr><td>{html.escape(m['name'])}</td>"
        f"<td>{_score(m.get('score'))}</td>"
        f"<td>{_score(m.get('threshold'))}</td>"
        f"<td>{_status(m['success'])}</td>"
        f"<td>{html.escape(m.get('reason') or '')}</td></tr>"
        for m in case.get("metricsData") or []
    )
    contexts = "".join(
        f"<pre>{html.escape(text)}</pre>" for text in case.get("retrievalContext") or []
    )
    return (
        f"<details><summary>{_status(case['success'])} "
        f"{html.escape(case_title(case))} · {_duration(case.get('runDuration'))} · "
        f"{_money(case.get('evaluationCost'))}</summary>"
        f"<p><b>Question:</b> {html.escape(case.get('input') or '')}</p>"
        f"<p><b>Answer:</b></p><pre>{html.escape(case.get('actualOutput') or '')}</pre>"
        "<table><thead><tr><th>Metric</th><th>Score</th><th>Threshold</th>"
        f"<th>Result</th><th>Reason</th></tr></thead><tbody>{metrics}</tbody></table>"
        f"<p><b>Retrieved passages:</b></p>{contexts or '<p>none</p>'}"
        "</details>"
    )


def render(run: dict, source_name: str) -> str:
    cases = run.get("testCases") or []
    passed = sum(1 for c in cases if c["success"])
    body = (
        f"<h1>RAG eval report</h1>"
        f"<p class='meta'>Source: {html.escape(source_name)} · "
        f"test file: {html.escape(run.get('testFile') or '')} · "
        f"{passed}/{len(cases)} test cases passed · "
        f"run duration {run.get('runDuration', 0):.1f}s · "
        f"evaluation cost ${run.get('evaluationCost') or 0:.4f}</p>"
        "<h2>Summary by model</h2>"
        f"{_summary_table(cases)}"
        "<h2>Test cases</h2>"
        + "".join(_case_section(case) for case in cases)
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>RAG eval report</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )


def convert(run_file: Path, output: Path | None = None) -> Path:
    """Write the HTML report for ``run_file`` next to it (``<run_file>.html``) unless ``output`` is given."""
    run = json.loads(run_file.read_text(encoding="utf-8"))
    output = output or run_file.with_name(run_file.name + ".html")
    output.write_text(render(run, run_file.name), encoding="utf-8")
    return output

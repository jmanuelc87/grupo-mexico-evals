"""RAG quality eval: every OpenAI, Claude and Gemini chat model, same questions, same index, same judge.

Calls paid APIs (agent + LLM judge), so it is skipped by default. A provider's models are skipped
when its API key is not set. Run it with:

    uv run pytest -m eval                                  # all models
    uv run pytest -m eval -k "gpt-5.5 and not gpt-5.5-pro" # one model
    uv run pytest -m eval -k claude                        # one provider
    EVAL_FAMILY=claude uv run pytest -m eval               # one family, same as the rag-benchmark switch
"""

import atexit
import csv
import os
import time
from pathlib import Path

import numpy as np
import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from grupo_mexico_evals.agent import (
    CLAUDE_MODELS,
    GEMINI_MODELS,
    OPENAI_CHAT_MODELS,
    PROVIDERS,
    ask_with_context,
    build_agent,
    build_vector_store,
)
from grupo_mexico_evals.report import RESULTS_DIR, convert, latest_run
from grupo_mexico_evals.rubrics import RUBRICS, parse_rubric

GOLDENS = Path("data/usable/rh_goldens.csv")
NUM_QUESTIONS = 7  # per model
JUDGE_MODEL = os.environ.get(
    "EVAL_JUDGE_MODEL", "gpt-4.1"
)  # fixed, so scores are comparable across models
FAITHFULNESS_THRESHOLD = 0.8
RELEVANCY_THRESHOLD = 0.8
HALLUCINATION_THRESHOLD = 0.2  # a maximum: lower is better, unlike the other two
GEVAL_THRESHOLD = 0.7  # deepeval's default; GEval scores are 0-1 (rubric 0-10 / 10)

# Fields the GEval judge sees. Every rubric refers to the retrieved context; precision and
# completeness are also checked against the golden answer.
GEVAL_PARAMS = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.RETRIEVAL_CONTEXT,
]
GEVAL_EXTRA_PARAMS = {
    "precision": [LLMTestCaseParams.EXPECTED_OUTPUT],
    "completitud": [LLMTestCaseParams.EXPECTED_OUTPUT],
}


def geval_metrics() -> list[GEval]:
    """One GEval metric per rubric in ``grupo_mexico_evals.rubrics``."""
    metrics = []
    for name, prompt in RUBRICS.items():
        criteria, rubric = parse_rubric(prompt)
        metrics.append(
            GEval(
                name=name,
                criteria=criteria,
                rubric=rubric,
                evaluation_params=GEVAL_PARAMS + GEVAL_EXTRA_PARAMS.get(name, []),
                threshold=GEVAL_THRESHOLD,
                model=JUDGE_MODEL,
            )
        )
    return metrics

# Every chat model under test, as "<provider>:<model>" specs. The OpenAI key is always needed:
# it is the judge and the embeddings, whichever model is being tested.
ALL_MODEL_SPECS = {
    "openai": [f"openai:{m}" for m in OPENAI_CHAT_MODELS],
    "claude": [f"claude:{m}" for m in CLAUDE_MODELS],
    "gemini": [f"gemini:{m}" for m in GEMINI_MODELS],
}
# EVAL_FAMILY picks one family ("openai", "claude" or "gemini"); unset or "all" runs every family.
FAMILY = os.environ.get("EVAL_FAMILY", "all").strip().lower()
if FAMILY not in {"all", *ALL_MODEL_SPECS}:
    raise ValueError(f"EVAL_FAMILY must be all, openai, claude or gemini; got {FAMILY!r}")
MODEL_SPECS = [
    spec
    for family, specs in ALL_MODEL_SPECS.items()
    if FAMILY in {"all", family}
    for spec in specs
]


def model_params() -> list:
    """One pytest param per model spec, skipped when that provider's API key is missing."""
    params = []
    for spec in MODEL_SPECS:
        provider = PROVIDERS[spec.split(":", 1)[0]]
        params.append(
            pytest.param(
                spec,
                id=spec,
                marks=pytest.mark.skipif(
                    not os.environ.get(provider.api_key_env),
                    reason=f"{provider.api_key_env} is not set",
                ),
            )
        )
    return params


pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set (judge)"
    ),
]


def load_questions(
    path: Path = GOLDENS, limit: int = NUM_QUESTIONS
) -> list[tuple[str, str, str]]:
    """Pick ``limit`` distinct generated (question, expected answer, context) rows at random.

    Every model sees the same rows.
    """
    with path.open(encoding="utf-8", newline="") as file:
        rows = [
            (row["input"], row["expected_output"], row["context"])
            for row in csv.DictReader(file)
        ]
    indexes = np.random.choice(len(rows), size=limit, replace=False)
    return [rows[i] for i in indexes]


@pytest.fixture(scope="session")
def store():
    return build_vector_store()


@pytest.fixture(scope="session")
def agents(store):
    """One agent per model, built on first use and shared by that model's questions."""
    cache = {}

    def get(spec: str):
        if spec not in cache:
            cache[spec] = build_agent(spec, store)
        return cache[spec]

    return get


@pytest.mark.parametrize("question, expected_output, golden_context", load_questions())
@pytest.mark.parametrize("model", model_params())
def test_rag_quality(agents, model, question, expected_output, golden_context):
    answer, contexts = ask_with_context(agents(model), question)
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=expected_output,
        retrieval_context=contexts,  # what the agent actually retrieved
        context=[golden_context],  # ground truth from the goldens, used by HallucinationMetric
    )
    assert_test(
        test_case,
        [
            FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, model=JUDGE_MODEL),
            AnswerRelevancyMetric(threshold=RELEVANCY_THRESHOLD, model=JUDGE_MODEL),
            HallucinationMetric(threshold=HALLUCINATION_THRESHOLD, model=JUDGE_MODEL),
            *geval_metrics(),
        ],
    )


def write_html_report(started: float) -> None:
    """Convert the run file this process saved to HTML.

    ``deepeval test run`` writes the run file only after pytest has finished, so this runs at
    process exit (see the ``atexit`` hook below). A run file older than ``started`` came from an
    earlier run, so it is skipped.
    """
    folder = Path(os.environ.get("DEEPEVAL_RESULTS_FOLDER", RESULTS_DIR))
    try:
        run_file = latest_run(folder)
    except FileNotFoundError:
        return  # nothing saved, e.g. plain pytest without DEEPEVAL_RESULTS_FOLDER
    if run_file.stat().st_mtime < started:
        return
    print(f"\nHTML report: {convert(run_file)}")


atexit.register(write_html_report, time.time())

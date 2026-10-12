"""Run the RAG benchmark (chat models from ``tests/test_benchmark.py``) through deepeval.

deepeval only writes its run file when run through ``deepeval test run``, so this wraps that
command and points it at the reports folder. ``--family`` limits the run to one provider's models
(via ``EVAL_FAMILY``, read by the test file); other arguments are passed on to deepeval.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from grupo_mexico_evals.report import RESULTS_DIR

BENCHMARK_FILE = "tests/test_benchmark.py"
FAMILIES = ("all", "openai", "claude", "gemini")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the RAG benchmark; extra arguments go to deepeval."
    )
    parser.add_argument(
        "-f",
        "--family",
        choices=FAMILIES,
        default="all",
        help="model family to run (default: %(default)s)",
    )
    args, deepeval_args = parser.parse_known_args(argv)

    RESULTS_DIR.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.setdefault("DEEPEVAL_RESULTS_FOLDER", str(RESULTS_DIR))
    env["EVAL_FAMILY"] = args.family
    cmd = ["deepeval", "test", "run", BENCHMARK_FILE, "-m", "eval", *deepeval_args]
    return subprocess.run(cmd, env=env).returncode


def cli() -> None:
    """Entry point for the ``rag-benchmark`` command."""
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":
    cli()

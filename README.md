# Grupo Mexico Evals

Document pipeline and RAG evaluation for the Grupo México reports (financial statements, sustainability reports, code of conduct, and a BHP comparison). The project has three parts:

1. **Pipeline** (`grupo-mexico-evals`): turns PDFs into per-page text.
2. **RAG agent** (`rag-agent`): answers questions over that text, with a configurable chat model from OpenAI, Claude or Gemini.
3. **Evals** (`tests/test_benchmark.py`): scores the agent's answers with deepeval metrics, across the OpenAI, Claude and Gemini chat models.

## Layout

```
src/grupo_mexico_evals/
  pipeline.py    PDF -> pages -> layout detection -> OCR / vision -> markdown
  dataset.py     generates eval goldens from the processed documents (deepeval Synthesizer)
  agent.py       RAG agent: Chroma index, retrieval tool, model providers, CLI
  benchmark.py   rag-benchmark: runs the eval through deepeval, optionally one model family (-f)
  rubrics.py     Spanish GEval rubrics (relevancia, precision, ...)
  report.py      converts a deepeval run file (JSON) into an HTML report
tests/
  test_agent.py      unit tests (no API calls)
  test_benchmark.py  RAG quality eval (paid API calls)
data/
  raw/           input PDFs (not in git)
  processed/     per-page text, one .txt per page, e.g. 2024-Q4-FS-MDA_p001.txt
  usable/        generated eval goldens (rh_goldens.csv)
reports/         deepeval run files and HTML reports (not in git)
```

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create `.env` in the project root with the keys for the providers you use:

```bash
OPENAI_API_KEY=...        # required: embeddings, the OpenAI models, and the eval judge
ANTHROPIC_API_KEY=...     # optional: claude models
GOOGLE_API_KEY=...        # optional: gemini models
RAG_MODEL=openai          # optional: default model for rag-agent
```

## 1. Pipeline: PDFs to text

```bash
uv run grupo-mexico-evals -i data/raw -o data/processed
```

For each PDF in the input folder, every page is rendered and then:

- **Layout detection** with `cmarkea/detr-layout-detection`, keeping regions above a score threshold (default 0.4).
- **Text regions** (titles, paragraphs, lists, captions, headers, footers) go through EasyOCR, in Spanish and English.
- **Pictures, tables and formulas** go to Claude (`claude-haiku-5-5`) through the Claude Agent SDK, which describes or transcribes them.

Each page is written as `<pdf-stem>_p<page>.md`. Without `-o`, the results are printed. Options: `-t` threshold, `-s` render scale.

> **Known mismatch:** the pipeline writes `.md` files, but the RAG agent only indexes `.txt` files (`SUPPORTED_EXTENSIONS` in `agent.py`), and the existing `data/processed` folder holds `.txt` files. Either rename the pipeline output to `.txt`, or change `SUPPORTED_EXTENSIONS` before rebuilding the index.

## 2. RAG agent

```bash
uv run rag-agent "¿Qué dice el informe sobre el agua?"
uv run rag-agent -m claude:claude-opus-5-5 "question"
uv run rag-agent -m gemini "question"
uv run rag-agent                     # interactive prompt; type "exit" to quit
uv run rag-agent --reindex "question" # rebuild the vector index first
```

The model is chosen with `-m` or `$RAG_MODEL`, in the form `<provider>` or `<provider>:<model>`:

| Provider | Default model | Key |
|---|---|---|
| `openai` | `gpt-6-astra` | `OPENAI_API_KEY` |
| `claude` | `claude-sonnet-5-5` | `ANTHROPIC_API_KEY` |
| `gemini` | `gemini-3.8-flash` | `GOOGLE_API_KEY` |

How it works:

- The processed `.txt` pages are split into 1000-character chunks (150 overlap) and stored in a persistent Chroma index in `.vector_db/`. The index is built on first use.
- Embeddings are always OpenAI (`text-embedding-3-small`), whichever chat model answers, so one index serves every model.
- The chat model gets one tool, `search_documents`, which returns the top 5 passages. It cites them as `[source p.N]`, and says so when the documents don't contain the answer.
- OpenAI models use the Responses API, because some reasoning models reject function tools on Chat Completions.

The model lists in `agent.py` (`OPENAI_CHAT_MODELS`, `CLAUDE_MODELS`, `GEMINI_MODELS`) are reference lists of current models: the OpenAI and Gemini lists come from the providers' model pages, and the Claude list is the set of active models from Anthropic's docs, including dated IDs. Any model ID is accepted, not only those in the lists.

## 3. Evals

The eval checks the chat models in `agent.py` against the same questions, drawn from `data/usable/rh_goldens.csv`. By default it covers every family (OpenAI, Claude and Gemini); `-f` / `EVAL_FAMILY` limits it to one. A model is skipped when its provider's API key is not set. The OpenAI key is always required, because the judge and the embeddings use it.

Each test case runs the agent on one question, then scores the answer with:

| Metric | Threshold | Notes |
|---|---|---|
| Faithfulness | ≥ 0.8 | answer vs. retrieved passages |
| Answer relevancy | ≥ 0.7 | |
| Hallucination | ≤ 0.2 | answer vs. the golden context; lower is better |
| GEval × 9 | ≥ 0.5 | Spanish rubrics in `rubrics.py`: relevancia, precision, completitud, claridad, razonamiento_logico, contextualizacion, accionabilidad, estructura, profundidad_analitica |

The judge is `gpt-4.1` (`EVAL_JUDGE_MODEL` overrides it), for every model, so scores are comparable. Each test case makes one agent run plus 12 judge calls (3 metrics and 9 GEval), so the full eval is expensive.

Evals are skipped by default (`addopts = "-m 'not eval'"` in `pyproject.toml`). Run them with the `eval` marker:

```bash
uv run rag-benchmark                                     # benchmark: every family, saves JSON + HTML report to reports/
uv run rag-benchmark -f claude                           # benchmark: one family (openai, claude or gemini)
uv run rag-benchmark -f claude -m claude-opus-5-5        # benchmark: one model of a family
uv run pytest                                            # unit tests only (no API calls)
uv run pytest -m eval -k "gpt-5.5 and not gpt-5.5-pro"   # one OpenAI model
uv run pytest -m eval -k claude                          # one provider
EVAL_FAMILY=gemini uv run pytest -m eval                 # one family, without a report
```

### Reports

`rag-benchmark` runs the benchmark through `deepeval test run` and saves its report. It sets `DEEPEVAL_RESULTS_FOLDER` to `reports/` unless that is already set, and passes any other arguments on to deepeval.

deepeval writes a JSON run file to `reports/`. `test_benchmark.py` registers an `atexit` hook that converts the newest run file to an HTML report (`<run-file>.html`) in the same folder, after the process exits. It only converts files written during the current process. `deepeval test run` has no `-k` option, so `rag-benchmark` selects a family with `-f` and a model with `-m` (which needs `-f`) instead. Plain `pytest` also works for a single model, but it does not write a run file.

To convert an existing run file by hand:

```bash
uv run python -c "from pathlib import Path; from grupo_mexico_evals.report import convert; print(convert(Path('reports/<run-file>')))"
```

## Tests

```bash
uv run pytest      # unit tests (tests/test_agent.py): agent helpers, provider selection, document loading
```

## Dependencies notes

- `deepeval==2.9.3` pins `anthropic<0.50`, which blocks `langchain-anthropic`. `pyproject.toml` overrides this with `anthropic>=0.50` under `[tool.uv]`.
- `requires-python` is capped at `<3.14`, because the dependency resolver fails on 3.14.

## Git-ignored

`data/`, `.vector_db/`, `.deepeval/`, `.env`, `.venv/`, `.pytest_cache/`. The `reports/` folder is not ignored yet, so add it to `.gitignore` before committing run files.

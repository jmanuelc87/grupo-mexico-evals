import pytest

from grupo_mexico_evals.agent import PROVIDERS, build_model, load_documents, parse_model_spec


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("openai", ("openai", PROVIDERS["openai"].default_model)),
        ("Claude", ("claude", PROVIDERS["claude"].default_model)),
        ("anthropic", ("claude", PROVIDERS["claude"].default_model)),
        ("gemini:gemini-custom", ("gemini", "gemini-custom")),
        ("claude:claude-opus-5-5", ("claude", "claude-opus-5-5")),
    ],
)
def test_parse_model_spec(spec, expected):
    assert parse_model_spec(spec) == expected


def test_parse_model_spec_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        parse_model_spec("llama")


@pytest.mark.parametrize(
    ("spec", "env_var", "class_name"),
    [
        ("openai", "OPENAI_API_KEY", "ChatOpenAI"),
        ("claude", "ANTHROPIC_API_KEY", "ChatAnthropic"),
        ("gemini", "GOOGLE_API_KEY", "ChatGoogleGenerativeAI"),
    ],
)
def test_build_model_picks_provider_class(monkeypatch, spec, env_var, class_name):
    monkeypatch.setenv(env_var, "test-key")
    assert type(build_model(spec)).__name__ == class_name


def test_build_model_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_model("claude")


def test_load_documents_parses_pages_and_skips_empty(tmp_path):
    (tmp_path / "report_p001.txt").write_text("first page")
    (tmp_path / "report_p002.txt").write_text("  \n")
    (tmp_path / "notes.txt").write_text("no page suffix")
    (tmp_path / "ignored.md").write_text("wrong extension")

    docs = {d.metadata["source"]: d for d in load_documents(tmp_path)}

    assert docs["report"].metadata == {"source": "report", "page": 1}
    assert docs["notes"].metadata == {"source": "notes"}
    assert len(docs) == 2

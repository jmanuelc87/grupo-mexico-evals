"""RAG agent over the processed documents, with a configurable chat model (OpenAI, Claude or Gemini).

The chat model is picked at runtime with a spec such as ``"claude"`` or ``"gemini:gemini-3.8-flash"``.
Embeddings are always OpenAI, whatever the chat model, so the vector store is built once and shared
by every model being compared.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

load_dotenv()

DATA_DIR = Path("data/processed")
VECTOR_DB_DIR = Path(".vector_db")  # git-ignored
COLLECTION = "grupo_mexico"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 8
INDEX_BATCH_SIZE = 100
SUPPORTED_EXTENSIONS = (".txt",)
DEFAULT_MODEL_ENV = "RAG_MODEL"

SYSTEM_PROMPT = (
    "You answer questions about Grupo Mexico, using the search_documents tool to look up its reports "
    "(financial statements, sustainability reports, code of conduct, and a comparison with BHP). "
    "Base every answer on what the tool returns, and cite the sources you used as [source p.N]. "
    "The documents are a mix of Spanish and English, and a query in one language does not retrieve "
    "passages written in the other, so search in both languages and rephrase when results are off-target. "
    "If the documents still don't contain the answer, say so instead of guessing. "
    "Answer in the language of the question."
)


@dataclass(frozen=True)
class Provider:
    langchain_id: str  # provider id understood by langchain's init_chat_model
    default_model: str
    api_key_env: str


# Current OpenAI chat/reasoning models, from https://developers.openai.com/api/docs/models/all.
# Image, audio, embedding and deprecated models are left out. Any ID is accepted; this is for reference.
OPENAI_CHAT_MODELS = (
    # Flagship
    "gpt-6-astra",  # most capable, for the most demanding work
    "gpt-6.1-sol",  # near-Astra performance at lower cost
    "gpt-6-luna",  # most efficient, for high-volume tasks
    # General purpose
    "gpt-6-sol",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    # "gpt-5.5-pro",
    "gpt-5.4",
    # "gpt-5.4-pro",
    "gpt-5.4-mini",
)

# Active Claude models. Any ID is accepted; this is for reference.
CLAUDE_MODELS = (
    "claude-fable-5-1",  # 1M context, 128K output
    "claude-opus-5-5",  # 1M context, 128K output
    "claude-sonnet-5-5",  # 1M context, 128K output
    "claude-haiku-5-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-5-20251101",
    "claude-haiku-4-5-20251001",  # 200K context, 64K output
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-1-20250805",
)

# Current Gemini text-output models, from https://ai.google.dev/gemini-api/docs/models.
# Live, TTS, image, video, embedding, legacy 2.x and shut-down models are left out.
# Any ID is accepted; this is for reference.
GEMINI_MODELS = (
    # Stable
    "gemini-3.8-flash",  # most capable Flash, for long-running coding and agents
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",  # fastest, most cost-effective 3.5 model
    "gemini-3.1-flash-lite",
    # Preview
    "gemini-3.1-pro-preview",  # advanced reasoning and agentic coding
    "gemini-3-flash-preview",
)

PROVIDERS: dict[str, Provider] = {
    "openai": Provider("openai", "gpt-6-astra", "OPENAI_API_KEY"),
    "claude": Provider("anthropic", "claude-sonnet-5-5", "ANTHROPIC_API_KEY"),
    "gemini": Provider("google_genai", "gemini-3.8-flash", "GOOGLE_API_KEY"),
}
# Alternative spellings accepted in a model spec.
PROVIDER_ALIASES = {"anthropic": "claude", "google": "gemini", "google_genai": "gemini"}


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Split ``"<provider>"`` or ``"<provider>:<model>"`` into (provider key, model name)."""
    name, _, model = spec.strip().partition(":")
    key = PROVIDER_ALIASES.get(name.lower(), name.lower())
    if key not in PROVIDERS:
        raise ValueError(
            f"unknown provider {name!r}; choose one of: {', '.join(PROVIDERS)}"
        )
    return key, model.strip() or PROVIDERS[key].default_model


def build_model(spec: str, **kwargs) -> BaseChatModel:
    """Chat model for ``spec``; extra kwargs (temperature, max_tokens, ...) go to the provider class."""
    key, model = parse_model_spec(spec)
    provider = PROVIDERS[key]
    if not os.environ.get(provider.api_key_env):
        raise RuntimeError(
            f"{provider.api_key_env} is not set (needed for {key}); add it to .env"
        )
    if key == "openai":
        # Newer OpenAI reasoning models reject function tools on /v1/chat/completions.
        kwargs.setdefault("use_responses_api", True)
    return init_chat_model(model, model_provider=provider.langchain_id, **kwargs)


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """One Document per non-empty page file, named ``<pdf-stem>_p<page>.txt`` by the pipeline."""
    documents = []
    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        source, _, page = path.stem.rpartition("_p")
        metadata = (
            {"source": source, "page": int(page)}
            if source and page.isdigit()
            else {"source": path.stem}
        )
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def build_vector_store(
    data_dir: Path = DATA_DIR,
    persist_dir: Path = VECTOR_DB_DIR,
    reindex: bool = False,
) -> Chroma:
    """Open the persisted Chroma store, indexing ``data_dir`` first if it is empty (or ``reindex``)."""
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(persist_dir),
    )
    if reindex:
        store.reset_collection()
    if not store.get(limit=1)["ids"]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(load_documents(data_dir))
        if not chunks:
            raise FileNotFoundError(f"no documents to index in {data_dir}")
        for start in tqdm(
            range(0, len(chunks), INDEX_BATCH_SIZE),
            desc="Indexing",
            unit="batch",
            file=sys.stderr,
        ):
            store.add_documents(chunks[start : start + INDEX_BATCH_SIZE])
    return store


def make_search_tool(store: Chroma, k: int = TOP_K):
    @tool(response_format="content_and_artifact")
    def search_documents(query: str):
        """Search the Grupo Mexico documents and return the most relevant passages with their sources."""
        docs = store.similarity_search(query, k=k)
        content = "\n\n".join(
            f"[{doc.metadata.get('source')} p.{doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in docs
        )
        return content or "No relevant passages found.", docs

    return search_documents


def build_agent(
    model: str | BaseChatModel | None = None,
    store: Chroma | None = None,
    k: int = TOP_K,
):
    """RAG agent: the model decides when to call the retrieval tool.

    ``model`` is a spec (see :func:`parse_model_spec`) or a ready chat model; it defaults to
    ``$RAG_MODEL`` and then to ``"openai"``.
    """
    if model is None:
        model = os.environ.get(DEFAULT_MODEL_ENV, "openai")
    if isinstance(model, str):
        model = build_model(model)
    store = store or build_vector_store()
    return create_agent(
        model, tools=[make_search_tool(store, k)], system_prompt=SYSTEM_PROMPT
    )


def ask(agent, question: str) -> str:
    """Run one question through the agent and return the final answer as text."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].text


def ask_with_context(agent, question: str) -> tuple[str, list[str]]:
    """Like :func:`ask`, but also return the text of every passage the agent retrieved (for evals)."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    contexts = [
        doc.page_content
        for message in result["messages"]
        if isinstance(message, ToolMessage)
        for doc in message.artifact or []
    ]
    return result["messages"][-1].text, contexts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask questions about the Grupo Mexico documents."
    )
    parser.add_argument(
        "question", nargs="?", help="question to ask (default: interactive prompt)"
    )
    parser.add_argument(
        "-m",
        "--model",
        default=os.environ.get(DEFAULT_MODEL_ENV, "openai"),
        help=(
            f"openai | claude | gemini, optionally with a model name, e.g. claude:claude-opus-5-5 "
            f"(default: ${DEFAULT_MODEL_ENV} or %(default)s)"
        ),
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=TOP_K,
        help="passages retrieved per search (default: %(default)s)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="rebuild the vector store from the documents",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = build_vector_store(reindex=args.reindex)
    agent = build_agent(args.model, store, k=args.top_k)
    if args.question:
        print(ask(agent, args.question))
        return 0
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            return 0
        if question in {"exit", "quit"}:
            return 0
        if question:
            print(ask(agent, question), end="\n\n")


def cli() -> None:
    """Entry point for the ``rag-agent`` command and ``python -m``: friendly errors, no traceback."""
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()

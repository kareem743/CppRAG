import os
from pathlib import Path
from typing import List, Optional

import typer

from rag.config import AppConfig, build_config
from rag.embedders import FastEmbedEmbedder
from rag.errors import IngestionError, QueryError
from rag.index import Index
from rag.llm import OllamaLLM
from rag.logging_utils import setup_logging, timed
from rag.rag import RAGSystem
from rag.vector_store import LanceDBVectorStore


app = typer.Typer(help="Local RAG CLI")


def _parse_extensions(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return parts or None


def _list_files(directory: Path, extensions: Optional[List[str]]) -> List[str]:
    valid_exts = None
    if extensions:
        valid_exts = {ext if ext.startswith(".") else f".{ext}" for ext in extensions}

    all_files: List[str] = []
    for root, _, files in os.walk(directory):
        for file in files:
            ext = Path(file).suffix.lower()
            if valid_exts is None or ext in valid_exts:
                all_files.append(os.path.join(root, file))
    return all_files


def _build_overrides(
    chunk_size: Optional[int],
    overlap: Optional[int],
    extensions: Optional[str],
    top_k: Optional[int],
    model: Optional[str],
    db_path: Optional[str],
    table_name: Optional[str],
    files_per_batch: Optional[int],
    adaptive_batching: Optional[bool],
    min_files_per_batch: Optional[int],
    max_files_per_batch: Optional[int],
    target_batch_seconds: Optional[float],
    prefer_gpu: Optional[bool],
    max_retries: Optional[int],
    gpu_batch_size: Optional[int],
    cpu_batch_size: Optional[int],
    log_level: Optional[str],
) -> dict:
    overrides: dict = {}

    def _set(path: List[str], value) -> None:
        node = overrides
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value

    if chunk_size is not None:
        _set(["ingestion", "chunk_size"], chunk_size)
    if overlap is not None:
        _set(["ingestion", "overlap"], overlap)
    if extensions is not None:
        _set(["ingestion", "extensions"], _parse_extensions(extensions))
    if files_per_batch is not None:
        _set(["ingestion", "files_per_batch"], files_per_batch)
    if adaptive_batching is not None:
        _set(["ingestion", "adaptive_batching"], adaptive_batching)
    if min_files_per_batch is not None:
        _set(["ingestion", "min_files_per_batch"], min_files_per_batch)
    if max_files_per_batch is not None:
        _set(["ingestion", "max_files_per_batch"], max_files_per_batch)
    if target_batch_seconds is not None:
        _set(["ingestion", "target_batch_seconds"], target_batch_seconds)
    if top_k is not None:
        _set(["query", "top_k"], top_k)
    if model is not None:
        _set(["llm", "model"], model)
    if db_path is not None:
        _set(["vector_store", "db_path"], db_path)
    if table_name is not None:
        _set(["vector_store", "table_name"], table_name)
    if prefer_gpu is not None:
        _set(["embedding", "prefer_gpu"], prefer_gpu)
    if max_retries is not None:
        _set(["embedding", "max_retries"], max_retries)
    if gpu_batch_size is not None:
        _set(["embedding", "gpu_batch_size"], gpu_batch_size)
    if cpu_batch_size is not None:
        _set(["embedding", "cpu_batch_size"], cpu_batch_size)
    if log_level is not None:
        _set(["logging", "level"], log_level)

    return overrides


def _resolve_config(config_path: Optional[str], overrides: dict) -> AppConfig:
    path = config_path or os.environ.get("RAG_CONFIG")
    return build_config(path, overrides)


@app.callback()
def main(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    overrides = {}
    cfg = _resolve_config(config, overrides)
    setup_logging(cfg.logging.level, verbose=verbose)


@app.command()
@timed("ingest")
def ingest(
    directory: Path = typer.Argument(..., help="Directory to ingest"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to YAML config"),
    chunk_size: Optional[int] = typer.Option(None, "--chunk-size", help="Tokens per chunk"),
    overlap: Optional[int] = typer.Option(None, "--overlap", help="Token overlap per chunk"),
    extensions: Optional[str] = typer.Option(None, "--extensions", help="Comma-separated extensions"),
    files_per_batch: Optional[int] = typer.Option(None, "--files-per-batch", help="Initial batch size"),
    adaptive_batching: Optional[bool] = typer.Option(None, "--adaptive-batching/--no-adaptive-batching", help="Enable adaptive batch sizing"),
    min_files_per_batch: Optional[int] = typer.Option(None, "--min-files-per-batch", help="Minimum batch size"),
    max_files_per_batch: Optional[int] = typer.Option(None, "--max-files-per-batch", help="Maximum batch size"),
    target_batch_seconds: Optional[float] = typer.Option(None, "--target-batch-seconds", help="Target batch duration"),
    db_path: Optional[str] = typer.Option(None, "--db-path", help="LanceDB storage path"),
    table_name: Optional[str] = typer.Option(None, "--table-name", help="LanceDB table name"),
    prefer_gpu: Optional[bool] = typer.Option(None, "--prefer-gpu/--no-prefer-gpu", help="Prefer GPU embeddings"),
    max_retries: Optional[int] = typer.Option(None, "--embed-retries", help="Embedding retry count"),
    gpu_batch_size: Optional[int] = typer.Option(None, "--embed-gpu-batch", help="GPU batch size"),
    cpu_batch_size: Optional[int] = typer.Option(None, "--embed-cpu-batch", help="CPU batch size"),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="Logging level"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config and show actions only"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    overrides = _build_overrides(
        chunk_size,
        overlap,
        extensions,
        top_k=None,
        model=None,
        db_path=db_path,
        table_name=table_name,
        files_per_batch=files_per_batch,
        adaptive_batching=adaptive_batching,
        min_files_per_batch=min_files_per_batch,
        max_files_per_batch=max_files_per_batch,
        target_batch_seconds=target_batch_seconds,
        prefer_gpu=prefer_gpu,
        max_retries=max_retries,
        gpu_batch_size=gpu_batch_size,
        cpu_batch_size=cpu_batch_size,
        log_level=log_level,
    )
    cfg = _resolve_config(config, overrides)
    setup_logging(cfg.logging.level, verbose=verbose)

    if dry_run:
        files = _list_files(directory, cfg.ingestion.extensions)
        typer.echo(f"Dry run: would ingest {len(files)} files from {directory}")
        return

    embedder = FastEmbedEmbedder(
        prefer_gpu=cfg.embedding.prefer_gpu,
        max_retries=cfg.embedding.max_retries,
        gpu_batch_size=cfg.embedding.gpu_batch_size,
        cpu_batch_size=cfg.embedding.cpu_batch_size,
    )
    vector_store = LanceDBVectorStore(
        db_path=cfg.vector_store.db_path,
        table_name=cfg.vector_store.table_name,
    )
    try:
        Index(
            directory=directory,
            chunk_size=cfg.ingestion.chunk_size,
            overlap=cfg.ingestion.overlap,
            extensions=cfg.ingestion.extensions,
            embedder=embedder,
            vector_store=vector_store,
            files_per_batch=cfg.ingestion.files_per_batch,
            adaptive_batching=cfg.ingestion.adaptive_batching,
            min_files_per_batch=cfg.ingestion.min_files_per_batch,
            max_files_per_batch=cfg.ingestion.max_files_per_batch,
            target_batch_seconds=cfg.ingestion.target_batch_seconds,
        )
    except IngestionError as exc:
        raise typer.Exit(code=1) from exc


@app.command()
@timed("query")
def query(
    directory: Path = typer.Argument(..., help="Directory to query against"),
    question: str = typer.Argument(..., help="Question to answer"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to YAML config"),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Top K chunks"),
    model: Optional[str] = typer.Option(None, "--model", help="Ollama model name"),
    chunk_size: Optional[int] = typer.Option(None, "--chunk-size", help="Tokens per chunk"),
    overlap: Optional[int] = typer.Option(None, "--overlap", help="Token overlap per chunk"),
    extensions: Optional[str] = typer.Option(None, "--extensions", help="Comma-separated extensions"),
    files_per_batch: Optional[int] = typer.Option(None, "--files-per-batch", help="Initial batch size"),
    adaptive_batching: Optional[bool] = typer.Option(None, "--adaptive-batching/--no-adaptive-batching", help="Enable adaptive batch sizing"),
    min_files_per_batch: Optional[int] = typer.Option(None, "--min-files-per-batch", help="Minimum batch size"),
    max_files_per_batch: Optional[int] = typer.Option(None, "--max-files-per-batch", help="Maximum batch size"),
    target_batch_seconds: Optional[float] = typer.Option(None, "--target-batch-seconds", help="Target batch duration"),
    db_path: Optional[str] = typer.Option(None, "--db-path", help="LanceDB storage path"),
    table_name: Optional[str] = typer.Option(None, "--table-name", help="LanceDB table name"),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="Logging level"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config and show actions only"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    overrides = _build_overrides(
        chunk_size,
        overlap,
        extensions,
        top_k=top_k,
        model=model,
        db_path=db_path,
        table_name=table_name,
        files_per_batch=files_per_batch,
        adaptive_batching=adaptive_batching,
        min_files_per_batch=min_files_per_batch,
        max_files_per_batch=max_files_per_batch,
        target_batch_seconds=target_batch_seconds,
        prefer_gpu=None,
        max_retries=None,
        gpu_batch_size=None,
        cpu_batch_size=None,
        log_level=log_level,
    )
    cfg = _resolve_config(config, overrides)
    setup_logging(cfg.logging.level, verbose=verbose)

    if dry_run:
        typer.echo("Dry run: would run query with configured index and LLM")
        return

    embedder = FastEmbedEmbedder(
        prefer_gpu=cfg.embedding.prefer_gpu,
        max_retries=cfg.embedding.max_retries,
        gpu_batch_size=cfg.embedding.gpu_batch_size,
        cpu_batch_size=cfg.embedding.cpu_batch_size,
    )
    vector_store = LanceDBVectorStore(
        db_path=cfg.vector_store.db_path,
        table_name=cfg.vector_store.table_name,
    )
    index = Index(
        directory=directory,
        chunk_size=cfg.ingestion.chunk_size,
        overlap=cfg.ingestion.overlap,
        extensions=cfg.ingestion.extensions,
        embedder=embedder,
        vector_store=vector_store,
        files_per_batch=cfg.ingestion.files_per_batch,
        adaptive_batching=cfg.ingestion.adaptive_batching,
        min_files_per_batch=cfg.ingestion.min_files_per_batch,
        max_files_per_batch=cfg.ingestion.max_files_per_batch,
        target_batch_seconds=cfg.ingestion.target_batch_seconds,
    )
    llm = OllamaLLM(model=cfg.llm.model, timeout_seconds=cfg.llm.timeout_seconds)
    rag = RAGSystem(index=index, llm=llm, top_k=cfg.query.top_k)
    try:
        answer = rag.answer(question)
    except QueryError as exc:
        raise typer.Exit(code=1) from exc
    typer.echo(answer)


@app.command()
@timed("serve")
def serve(
    directory: Path = typer.Argument(..., help="Directory to serve against"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to YAML config"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config and show actions only"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    overrides = {}
    cfg = _resolve_config(config, overrides)
    setup_logging(cfg.logging.level, verbose=verbose)

    if dry_run:
        typer.echo("Dry run: would start interactive query loop")
        return

    embedder = FastEmbedEmbedder(
        prefer_gpu=cfg.embedding.prefer_gpu,
        max_retries=cfg.embedding.max_retries,
        gpu_batch_size=cfg.embedding.gpu_batch_size,
        cpu_batch_size=cfg.embedding.cpu_batch_size,
    )
    vector_store = LanceDBVectorStore(
        db_path=cfg.vector_store.db_path,
        table_name=cfg.vector_store.table_name,
    )
    index = Index(
        directory=directory,
        chunk_size=cfg.ingestion.chunk_size,
        overlap=cfg.ingestion.overlap,
        extensions=cfg.ingestion.extensions,
        embedder=embedder,
        vector_store=vector_store,
        files_per_batch=cfg.ingestion.files_per_batch,
        adaptive_batching=cfg.ingestion.adaptive_batching,
        min_files_per_batch=cfg.ingestion.min_files_per_batch,
        max_files_per_batch=cfg.ingestion.max_files_per_batch,
        target_batch_seconds=cfg.ingestion.target_batch_seconds,
    )
    llm = OllamaLLM(model=cfg.llm.model, timeout_seconds=cfg.llm.timeout_seconds)
    rag = RAGSystem(index=index, llm=llm, top_k=cfg.query.top_k)

    typer.echo("Interactive mode. Type 'exit' to quit.")
    while True:
        prompt = typer.prompt("Question")
        if prompt.strip().lower() in {"exit", "quit"}:
            break
        answer = rag.answer(prompt)
        typer.echo(answer)


if __name__ == "__main__":
    app()

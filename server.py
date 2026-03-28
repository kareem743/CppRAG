import json
import os
import threading
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Literal, Dict, Iterable, Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.chunker import RagCoreChunker
from rag.config import AppConfig, build_config
from rag.embedders import FastEmbedEmbedder
from rag.errors import QueryError
from rag.ingestion_state import load_state, plan_files
from rag.index import Index
from rag.llm import OllamaLLM
from rag.models import Chunk
from rag.rag import RAGSystem
from rag.rerankers import FlashRankReranker
from rag.vector_store import LanceDBVectorStore
from run_eval import evaluate as run_evaluate
from run_eval import _build_overrides as build_eval_overrides


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(None, ge=1)


class ChunkResponse(BaseModel):
    text: str
    source: str
    score: Optional[float] = None


class MetricsResponse(BaseModel):
    latency_ms: float
    embed_ms: float
    retrieve_ms: float
    rerank_ms: float = 0.0
    generate_ms: float
    engine: Literal["cpp-accelerated"]


class ChatResponse(BaseModel):
    answer: str
    sources: List[ChunkResponse]
    metrics: MetricsResponse


class StatsResponse(BaseModel):
    ingestion_rate: float
    gpu_active: bool
    vector_count: int
    embedding_model: str
    llm_model: str
    embedding_device: Literal["gpu", "cpu", "unknown"]
    storage_db_bytes: int
    storage_cache_bytes: int


class VisualizeQuery(BaseModel):
    filepath: str = Field(..., min_length=1)


class ChunkSpan(BaseModel):
    start: int
    end: int


class VisualizeResponse(BaseModel):
    text: str
    chunks: List[ChunkSpan]


class IngestRequest(BaseModel):
    directory: Optional[str] = None
    chunk_size: Optional[int] = Field(None, ge=1)
    overlap: Optional[int] = Field(None, ge=0)
    extensions: Optional[List[str]] = None
    files_per_batch: Optional[int] = Field(None, ge=1)
    adaptive_batching: Optional[bool] = None
    min_files_per_batch: Optional[int] = Field(None, ge=1)
    max_files_per_batch: Optional[int] = Field(None, ge=1)
    target_batch_seconds: Optional[float] = Field(None, gt=0)
    db_path: Optional[str] = None
    table_name: Optional[str] = None
    prefer_gpu: Optional[bool] = None
    max_retries: Optional[int] = Field(None, ge=0)
    gpu_batch_size: Optional[int] = Field(None, ge=1)
    cpu_batch_size: Optional[int] = Field(None, ge=1)
    log_level: Optional[str] = None
    dry_run: bool = False


class IngestStatus(BaseModel):
    status: Literal["idle", "running", "complete", "error"]
    message: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    directory: Optional[str] = None
    total_files: Optional[int] = None
    changed_files: Optional[int] = None
    last_duration_ms: Optional[float] = None
    dry_run: Optional[bool] = None


class EvaluationResponse(BaseModel):
    run_timestamp: str
    config: Dict[str, Any]
    retrieval_metrics: Dict[str, Any]
    generation_metrics: Dict[str, Any]
    end_to_end_metrics: Dict[str, Any]
    per_question: List[Dict[str, Any]]


class EvaluationRunRequest(BaseModel):
    dataset: Optional[str] = None
    index_dir: Optional[str] = None
    output_dir: str = "eval_results"
    compare_baseline: bool = False
    set_baseline: bool = False
    config: Optional[str] = None
    chunk_size: Optional[int] = Field(None, ge=1)
    overlap: Optional[int] = Field(None, ge=0)
    extensions: Optional[str] = None
    files_per_batch: Optional[int] = Field(None, ge=1)
    adaptive_batching: Optional[bool] = None
    min_files_per_batch: Optional[int] = Field(None, ge=1)
    max_files_per_batch: Optional[int] = Field(None, ge=1)
    target_batch_seconds: Optional[float] = Field(None, gt=0)
    top_k: Optional[int] = Field(None, ge=1)
    model: Optional[str] = None
    db_path: Optional[str] = None
    table_name: Optional[str] = None
    prefer_gpu: Optional[bool] = None
    embed_retries: Optional[int] = Field(None, ge=0)
    embed_gpu_batch: Optional[int] = Field(None, ge=1)
    embed_cpu_batch: Optional[int] = Field(None, ge=1)
    log_level: Optional[str] = None


class EvaluationRunStatus(BaseModel):
    status: Literal["idle", "running", "complete", "error"]
    message: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    last_duration_ms: Optional[float] = None
    exit_code: Optional[int] = None
    latest_results_path: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


def _resolve_directory() -> Path:
    return Path(os.environ.get("RAG_DIRECTORY", os.getcwd())).resolve()


def _init_config() -> AppConfig:
    config_path = os.environ.get("RAG_CONFIG")
    return build_config(config_path, {})


def _init_rag() -> tuple[RAGSystem, Index, AppConfig]:
    cfg = _init_config()
    directory = _resolve_directory()

    embedder = FastEmbedEmbedder(
        model_name=cfg.embedding.model_name,
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
        auto_ingest=False,
        ensure_schema=False,
    )
    llm = OllamaLLM(model=cfg.llm.model, timeout_seconds=cfg.llm.timeout_seconds)
    reranker = None
    if cfg.reranker.enabled:
        reranker = FlashRankReranker(
            model_name=cfg.reranker.model,
            cache_dir=cfg.reranker.cache_dir,
        )
    rag = RAGSystem(
        index=index,
        llm=llm,
        top_k=cfg.query.top_k,
        reranker=reranker,
        rerank_expansion=cfg.reranker.top_n_expansion,
    )
    return rag, index, cfg


RAG_SYSTEM, RAG_INDEX, APP_CONFIG = _init_rag()

app = FastAPI(title="RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_INGEST_LOCK = threading.Lock()
_INGEST_STATE: Dict[str, Optional[object]] = {
    "status": "idle",
    "message": None,
    "started_at": None,
    "finished_at": None,
    "directory": None,
    "total_files": None,
    "changed_files": None,
    "last_duration_ms": None,
    "dry_run": None,
}

_STORAGE_CACHE: Dict[str, Optional[object]] = {
    "updated_at": None,
    "db_bytes": 0,
    "cache_bytes": 0,
}
_STORAGE_CACHE_TTL_SECONDS = 30.0

_EVAL_LOCK = threading.Lock()
_EVAL_STATE: Dict[str, Optional[object]] = {
    "status": "idle",
    "message": None,
    "started_at": None,
    "finished_at": None,
    "last_duration_ms": None,
    "exit_code": None,
    "latest_results_path": None,
    "options": None,
}


def _build_prompt(chunks: List[Chunk], question: str) -> str:
    context = "\n\n".join(
        f"[Source: {chunk.source}]\n{chunk.text}" for chunk in chunks
    )
    return (
        "You are a strict smart assistant that answers questions ONLY using the provided Context below.\n"
        "Do not hallucinate.\n"
        "Rules:\n"
        "1. Use the Context to answer the Question.\n"
        "2. Cite the Source ID (e.g. [Source: path/to/file]) for every claim.\n"
        "3. IF THE ANSWER IS NOT IN THE CONTEXT,  Do not guess.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _parse_extensions(extensions: Optional[Iterable[str]]) -> Optional[List[str]]:
    if extensions is None:
        return None
    cleaned = [ext.strip() for ext in extensions if ext.strip()]
    return cleaned or None


def _default_extensions() -> set:
    return {
        ".txt", ".md", ".markdown", ".rst", ".py", ".json",
        ".yaml", ".yml", ".toml", ".csv", ".ts", ".js",
        ".html", ".css", ".cpp", ".cc", ".c", ".h", ".hpp",
        ".java", ".go", ".rs", ".sh",
    }


def _list_files(directory: Path, extensions: Optional[Iterable[str]]) -> List[str]:
    if extensions is None:
        valid_exts = _default_extensions()
    else:
        valid_exts = {ext if ext.startswith(".") else f".{ext}" for ext in extensions}
    files: List[str] = []
    for root, _, names in os.walk(directory):
        for name in names:
            if Path(name).suffix.lower() in valid_exts:
                files.append(os.path.join(root, name))
    return files


def _build_overrides(request: IngestRequest) -> Dict:
    overrides: Dict = {}

    def _set(path: List[str], value) -> None:
        node = overrides
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value

    if request.chunk_size is not None:
        _set(["ingestion", "chunk_size"], request.chunk_size)
    if request.overlap is not None:
        _set(["ingestion", "overlap"], request.overlap)
    if request.extensions is not None:
        _set(["ingestion", "extensions"], _parse_extensions(request.extensions))
    if request.files_per_batch is not None:
        _set(["ingestion", "files_per_batch"], request.files_per_batch)
    if request.adaptive_batching is not None:
        _set(["ingestion", "adaptive_batching"], request.adaptive_batching)
    if request.min_files_per_batch is not None:
        _set(["ingestion", "min_files_per_batch"], request.min_files_per_batch)
    if request.max_files_per_batch is not None:
        _set(["ingestion", "max_files_per_batch"], request.max_files_per_batch)
    if request.target_batch_seconds is not None:
        _set(["ingestion", "target_batch_seconds"], request.target_batch_seconds)
    if request.db_path is not None:
        _set(["vector_store", "db_path"], request.db_path)
    if request.table_name is not None:
        _set(["vector_store", "table_name"], request.table_name)
    if request.prefer_gpu is not None:
        _set(["embedding", "prefer_gpu"], request.prefer_gpu)
    if request.max_retries is not None:
        _set(["embedding", "max_retries"], request.max_retries)
    if request.gpu_batch_size is not None:
        _set(["embedding", "gpu_batch_size"], request.gpu_batch_size)
    if request.cpu_batch_size is not None:
        _set(["embedding", "cpu_batch_size"], request.cpu_batch_size)
    if request.log_level is not None:
        _set(["logging", "level"], request.log_level)

    return overrides


def _iso_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def _fastembed_cache_path() -> Path:
    env_override = os.environ.get("FASTEMBED_CACHE_PATH")
    if env_override:
        return Path(env_override).expanduser()
    local_app = os.environ.get("LOCALAPPDATA")
    if local_app:
        return Path(local_app) / "Temp" / "fastembed_cache"
    return Path("fastembed_cache")


def _resolve_evaluation_path() -> Path:
    state_path = _EVAL_STATE.get("latest_results_path")
    if isinstance(state_path, str) and state_path:
        return Path(state_path).expanduser().resolve()
    path = os.environ.get("RAG_EVAL_RESULTS_PATH")
    if path:
        return Path(path).expanduser().resolve()
    return Path("eval_results") / "latest.json"


def _default_dataset_path() -> Optional[Path]:
    candidates = [Path("datasets") / "new_dataset.json", Path("datasets") / "dataset.json"]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _build_eval_options(request: EvaluationRunRequest) -> Dict[str, Any]:
    return {
        "dataset": request.dataset,
        "index_dir": request.index_dir,
        "output_dir": request.output_dir,
        "compare_baseline": request.compare_baseline,
        "set_baseline": request.set_baseline,
        "config": request.config,
        "chunk_size": request.chunk_size,
        "overlap": request.overlap,
        "extensions": request.extensions,
        "files_per_batch": request.files_per_batch,
        "adaptive_batching": request.adaptive_batching,
        "min_files_per_batch": request.min_files_per_batch,
        "max_files_per_batch": request.max_files_per_batch,
        "target_batch_seconds": request.target_batch_seconds,
        "top_k": request.top_k,
        "model": request.model,
        "db_path": request.db_path,
        "table_name": request.table_name,
        "prefer_gpu": request.prefer_gpu,
        "embed_retries": request.embed_retries,
        "embed_gpu_batch": request.embed_gpu_batch,
        "embed_cpu_batch": request.embed_cpu_batch,
        "log_level": request.log_level,
    }


def _get_storage_sizes() -> tuple[int, int]:
    now = datetime.utcnow().timestamp()
    updated_at = _STORAGE_CACHE.get("updated_at")
    if updated_at and (now - updated_at) < _STORAGE_CACHE_TTL_SECONDS:
        return int(_STORAGE_CACHE.get("db_bytes") or 0), int(
            _STORAGE_CACHE.get("cache_bytes") or 0
        )

    db_path = Path(APP_CONFIG.vector_store.db_path)
    cache_path = _fastembed_cache_path()
    db_bytes = _directory_size(db_path)
    cache_bytes = _directory_size(cache_path)
    _STORAGE_CACHE.update(
        {
            "updated_at": now,
            "db_bytes": db_bytes,
            "cache_bytes": cache_bytes,
        }
    )
    return db_bytes, cache_bytes


def _snapshot_ingest_state() -> IngestStatus:
    return IngestStatus(**_INGEST_STATE)


def _update_ingest_state(**updates) -> None:
    _INGEST_STATE.update(updates)


def _snapshot_eval_state() -> EvaluationRunStatus:
    return EvaluationRunStatus(**_EVAL_STATE)


def _update_eval_state(**updates) -> None:
    _EVAL_STATE.update(updates)


def _eval_worker(request: EvaluationRunRequest) -> None:
    start = perf_counter()
    try:
        dataset = request.dataset
        if not dataset:
            default_dataset = _default_dataset_path()
            if not default_dataset:
                raise ValueError(
                    "Dataset path is required (no datasets/new_dataset.json or datasets/dataset.json found)."
                )
            dataset = str(default_dataset)

        index_dir = request.index_dir or str(_resolve_directory())
        output_dir = request.output_dir or "eval_results"

        overrides = build_eval_overrides(
            request.chunk_size,
            request.overlap,
            request.extensions,
            request.files_per_batch,
            request.adaptive_batching,
            request.min_files_per_batch,
            request.max_files_per_batch,
            request.target_batch_seconds,
            request.top_k,
            request.model,
            request.db_path,
            request.table_name,
            request.prefer_gpu,
            request.embed_retries,
            request.embed_gpu_batch,
            request.embed_cpu_batch,
            request.log_level,
        )
        exit_code = run_evaluate(
            dataset_path=Path(dataset).expanduser().resolve(),
            index_dir=Path(index_dir).expanduser().resolve(),
            output_dir=Path(output_dir).expanduser().resolve(),
            compare_baseline=request.compare_baseline,
            set_baseline_flag=request.set_baseline,
            config=request.config,
            overrides=overrides,
        )
        latest_results_path = str((Path(output_dir).expanduser().resolve() / "latest.json"))
        _update_eval_state(
            status="complete" if exit_code == 0 else "error",
            message="Evaluation finished." if exit_code == 0 else f"Evaluation exited with code {exit_code}.",
            finished_at=_iso_now(),
            last_duration_ms=(perf_counter() - start) * 1000.0,
            exit_code=exit_code,
            latest_results_path=latest_results_path,
        )
    except Exception as exc:
        _update_eval_state(
            status="error",
            message=f"Evaluation failed: {exc}",
            finished_at=_iso_now(),
            last_duration_ms=(perf_counter() - start) * 1000.0,
            exit_code=2,
        )


def _ingest_worker(request: IngestRequest, directory: Path) -> None:
    start = perf_counter()
    try:
        overrides = _build_overrides(request)
        cfg = build_config(os.environ.get("RAG_CONFIG"), overrides)

        files = _list_files(directory, cfg.ingestion.extensions)
        state_path = Path(cfg.vector_store.db_path) / "ingestion_state.json"
        state = load_state(state_path)
        to_process, _ = plan_files(files, state)
        _update_ingest_state(
            total_files=len(files),
            changed_files=len(to_process),
        )

        if request.dry_run:
            _update_ingest_state(
                status="complete",
                message=f"Dry run complete. {len(to_process)} files need ingestion.",
                finished_at=_iso_now(),
                last_duration_ms=(perf_counter() - start) * 1000.0,
            )
            return

        embedder = FastEmbedEmbedder(
            model_name=cfg.embedding.model_name,
            prefer_gpu=cfg.embedding.prefer_gpu,
            max_retries=cfg.embedding.max_retries,
            gpu_batch_size=cfg.embedding.gpu_batch_size,
            cpu_batch_size=cfg.embedding.cpu_batch_size,
        )
        vector_store = LanceDBVectorStore(
            db_path=cfg.vector_store.db_path,
            table_name=cfg.vector_store.table_name,
        )
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

        _update_ingest_state(
            status="complete",
            message="Ingestion finished.",
            finished_at=_iso_now(),
            last_duration_ms=(perf_counter() - start) * 1000.0,
        )
    except Exception as exc:
        _update_ingest_state(
            status="error",
            message=f"Ingestion failed: {exc}",
            finished_at=_iso_now(),
            last_duration_ms=(perf_counter() - start) * 1000.0,
        )


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    start = perf_counter()
    try:
        top_k = request.top_k or APP_CONFIG.query.top_k
        answer, ranked, metrics = RAG_SYSTEM.answer_with_metrics(request.question, top_k=top_k)
        latency_ms = (perf_counter() - start) * 1000.0
        return ChatResponse(
            answer=answer,
            sources=[
                ChunkResponse(text=c.text, source=c.source, score=score)
                for c, score in ranked
            ],
            metrics=MetricsResponse(
                latency_ms=latency_ms,
                embed_ms=metrics.get("embed_ms", 0.0),
                retrieve_ms=metrics.get("retrieve_ms", 0.0),
                rerank_ms=metrics.get("rerank_ms", 0.0),
                generate_ms=metrics.get("generate_ms", 0.0),
                engine="cpp-accelerated",
            ),
        )
    except QueryError as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat request failed: {exc}") from exc


@app.get("/api/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    try:
        storage_db_bytes, storage_cache_bytes = _get_storage_sizes()
        last_provider = getattr(RAG_INDEX._embedder, "last_provider", None)
        embedding_device = (
            last_provider if last_provider in {"gpu", "cpu"} else "unknown"
        )
        embedding_model = APP_CONFIG.embedding.model_name or "fastembed-default"
        return StatsResponse(
            ingestion_rate=128.5,
            gpu_active=APP_CONFIG.embedding.prefer_gpu,
            vector_count=120_000,
            embedding_model=embedding_model,
            llm_model=APP_CONFIG.llm.model,
            embedding_device=embedding_device,
            storage_db_bytes=storage_db_bytes,
            storage_cache_bytes=storage_cache_bytes,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stats request failed: {exc}") from exc


@app.get("/api/visualize", response_model=VisualizeResponse)
def visualize(query: VisualizeQuery = Depends()) -> VisualizeResponse:
    try:
        file_path = Path(query.filepath).expanduser().resolve()
        text = file_path.read_text(encoding="utf-8", errors="replace")
        chunker = RagCoreChunker()
        chunks = chunker.chunk_files(
            [str(file_path)],
            chunk_size=APP_CONFIG.ingestion.chunk_size,
            overlap=APP_CONFIG.ingestion.overlap,
        )
        step = max(1, APP_CONFIG.ingestion.chunk_size - APP_CONFIG.ingestion.overlap)
        spans: List[ChunkSpan] = []
        for idx, chunk in enumerate(chunks):
            token_count = len(chunk.text.split())
            start = idx * step
            end = start + token_count - 1 if token_count else start
            spans.append(ChunkSpan(start=start, end=end))
        return VisualizeResponse(text=text, chunks=spans)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Visualize request failed: {exc}") from exc


@app.post("/api/ingest", response_model=IngestStatus)
def ingest(request: IngestRequest) -> IngestStatus:
    if _INGEST_STATE["status"] == "running":
        raise HTTPException(status_code=409, detail="Ingestion already running.")

    directory = (
        Path(request.directory).expanduser().resolve()
        if request.directory
        else _resolve_directory()
    )
    if not directory.exists() or not directory.is_dir():
        raise HTTPException(status_code=400, detail="Invalid directory path.")

    with _INGEST_LOCK:
        _update_ingest_state(
            status="running",
            message="Ingestion started.",
            started_at=_iso_now(),
            finished_at=None,
            directory=str(directory),
            total_files=None,
            changed_files=None,
            last_duration_ms=None,
            dry_run=request.dry_run,
        )

    thread = threading.Thread(target=_ingest_worker, args=(request, directory), daemon=True)
    thread.start()

    return _snapshot_ingest_state()


@app.get("/api/ingest/status", response_model=IngestStatus)
def ingest_status() -> IngestStatus:
    return _snapshot_ingest_state()


@app.get("/api/evaluation", response_model=EvaluationResponse)
def evaluation() -> EvaluationResponse:
    eval_path = _resolve_evaluation_path()
    if not eval_path.exists() or not eval_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation results not found at {eval_path}",
        )
    try:
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        return EvaluationResponse(**payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load evaluation results: {exc}",
        ) from exc


@app.post("/api/evaluation/run", response_model=EvaluationRunStatus)
def run_evaluation(request: EvaluationRunRequest) -> EvaluationRunStatus:
    if _EVAL_STATE["status"] == "running":
        raise HTTPException(status_code=409, detail="Evaluation already running.")

    options = _build_eval_options(request)
    with _EVAL_LOCK:
        _update_eval_state(
            status="running",
            message="Evaluation started.",
            started_at=_iso_now(),
            finished_at=None,
            last_duration_ms=None,
            exit_code=None,
            options=options,
        )

    thread = threading.Thread(target=_eval_worker, args=(request,), daemon=True)
    thread.start()
    return _snapshot_eval_state()


@app.get("/api/evaluation/status", response_model=EvaluationRunStatus)
def evaluation_status() -> EvaluationRunStatus:
    return _snapshot_eval_state()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=APP_CONFIG.server.host, port=APP_CONFIG.server.port)

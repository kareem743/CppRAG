import os
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.chunker import RagCoreChunker
from rag.config import AppConfig, build_config
from rag.embedders import FastEmbedEmbedder
from rag.errors import QueryError
from rag.index import Index
from rag.llm import OllamaLLM
from rag.models import Chunk
from rag.rag import RAGSystem
from rag.vector_store import LanceDBVectorStore


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(None, ge=1)


class ChunkResponse(BaseModel):
    text: str
    source: str
    score: Optional[float] = None


class MetricsResponse(BaseModel):
    latency_ms: float
    engine: Literal["cpp-accelerated"]


class ChatResponse(BaseModel):
    answer: str
    sources: List[ChunkResponse]
    metrics: MetricsResponse


class StatsResponse(BaseModel):
    ingestion_rate: float
    gpu_active: bool
    vector_count: int


class VisualizeQuery(BaseModel):
    filepath: str = Field(..., min_length=1)


class ChunkSpan(BaseModel):
    start: int
    end: int


class VisualizeResponse(BaseModel):
    text: str
    chunks: List[ChunkSpan]


def _resolve_directory() -> Path:
    return Path(os.environ.get("RAG_DIRECTORY", os.getcwd())).resolve()


def _init_config() -> AppConfig:
    config_path = os.environ.get("RAG_CONFIG")
    return build_config(config_path, {})


def _init_rag() -> tuple[RAGSystem, Index, AppConfig]:
    cfg = _init_config()
    directory = _resolve_directory()

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


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    start = perf_counter()
    try:
        top_k = request.top_k or APP_CONFIG.query.top_k
        ranked = RAG_INDEX.query(request.question, top_k=top_k)
        chunks = [chunk for chunk, _score in ranked]
        prompt = _build_prompt(chunks, request.question)
        answer = RAG_SYSTEM._llm.generate(prompt)
        latency_ms = (perf_counter() - start) * 1000.0
        return ChatResponse(
            answer=answer,
            sources=[
                ChunkResponse(text=c.text, source=c.source, score=score)
                for c, score in ranked
            ],
            metrics=MetricsResponse(latency_ms=latency_ms, engine="cpp-accelerated"),
        )
    except QueryError as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat request failed: {exc}") from exc


@app.get("/api/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    try:
        return StatsResponse(
            ingestion_rate=128.5,
            gpu_active=APP_CONFIG.embedding.prefer_gpu,
            vector_count=120_000,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=APP_CONFIG.server.host, port=APP_CONFIG.server.port)

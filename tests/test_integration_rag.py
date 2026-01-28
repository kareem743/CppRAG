from pathlib import Path

import pytest

from rag.embedders import FastEmbedEmbedder
from rag.index import Index
from rag.vector_store import LanceDBVectorStore


def _require_dependencies() -> None:
    pytest.importorskip("rag_core")
    pytest.importorskip("lancedb")
    pytest.importorskip("fastembed")


def test_ingest_then_query(tmp_path: Path) -> None:
    _require_dependencies()

    data_dir = tmp_path / "docs"
    data_dir.mkdir()
    (data_dir / "alpha.txt").write_text("Alpha beta gamma delta", encoding="utf-8")

    db_dir = tmp_path / "lancedb"
    vector_store = LanceDBVectorStore(db_path=str(db_dir), table_name="vectors")
    embedder = FastEmbedEmbedder(prefer_gpu=False, max_retries=0, cpu_batch_size=16, gpu_batch_size=16)

    index = Index(
        directory=data_dir,
        chunk_size=4,
        overlap=0,
        embedder=embedder,
        vector_store=vector_store,
        files_per_batch=10,
        adaptive_batching=False,
        state_path=tmp_path / "ingestion_state.json",
    )

    results = index.query("What is alpha?", top_k=1)
    assert results
    chunk, score = results[0]
    assert "alpha" in chunk.text.lower()

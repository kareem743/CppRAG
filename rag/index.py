import gc
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Iterable, List, Optional, Tuple

from .batch_sizer import AdaptiveBatchSizer
from .chunker import RagCoreChunker
from .embedders import FastEmbedEmbedder
from .errors import EmbeddingError, IngestionError, QueryError, VectorStoreError
from .ingestion_state import load_state, plan_files, save_state
from .models import Chunk
from .vector_store import LanceDBVectorStore

_DB_PATH = "./lancedb_data"
_TABLE_NAME = "vectors"
_STATE_FILE = "ingestion_state.json"
_LOGGER = logging.getLogger(__name__)


class Index:
    def __init__(
            self,
            directory: Path,
            chunk_size: int = 200,
            overlap: int = 50,
            extensions: Optional[Iterable[str]] = None,
            embedder: Optional[FastEmbedEmbedder] = None,
            vector_store: Optional[LanceDBVectorStore] = None,
            files_per_batch: int = 5000,
            adaptive_batching: bool = True,
            min_files_per_batch: int = 100,
            max_files_per_batch: int = 8000,
            target_batch_seconds: float = 20.0,
            state_path: Optional[Path] = None,
    ):
        self._embedder = embedder or FastEmbedEmbedder()
        self._vector_store = vector_store or LanceDBVectorStore(
            db_path=_DB_PATH,
            table_name=_TABLE_NAME,
        )
        self._files_per_batch = files_per_batch
        self._adaptive_batching = adaptive_batching
        self._min_files_per_batch = min_files_per_batch
        self._max_files_per_batch = max_files_per_batch
        self._target_batch_seconds = target_batch_seconds
        if state_path is None:
            db_path = (
                self._vector_store.db_path
                if hasattr(self._vector_store, "db_path")
                else _DB_PATH
            )
            state_path = Path(db_path) / _STATE_FILE
        self._state_path = state_path
        self._ensure_schema()
        self._ingest(directory, chunk_size, overlap, extensions)

    @classmethod
    def from_directory(
            cls,
            directory: Path,
            chunk_size: int = 200,
            overlap: int = 50,
            extensions: Optional[Iterable[str]] = None,
            **kwargs,
    ) -> "Index":
        return cls(
            directory=directory,
            chunk_size=chunk_size,
            overlap=overlap,
            extensions=extensions,
            **kwargs,
        )

    def _ensure_schema(self):
        try:
            sample = self._embedder.embed(["init"])
            if not sample:
                raise EmbeddingError("Failed to create schema sample embedding")
            self._vector_store.ensure_schema(sample[0])
        except (EmbeddingError, VectorStoreError) as exc:
            raise IngestionError("Failed to initialize schema") from exc

    def _ingest(
        self,
        directory: Path,
        chunk_size: int,
        overlap: int,
        extensions: Optional[Iterable[str]],
    ) -> None:
        start = perf_counter()

        print(f"Scanning {directory}...")
        all_files = []
        if extensions is None:
            valid_exts = {
                ".txt", ".md", ".markdown", ".rst", ".py", ".json",
                ".yaml", ".yml", ".toml", ".csv", ".ts", ".js",
                ".html", ".css", ".cpp", ".cc", ".c", ".h", ".hpp",
                ".java", ".go", ".rs", ".sh"
            }
        else:
            valid_exts = {ext if ext.startswith(".") else f".{ext}" for ext in extensions}

        for root, _, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in valid_exts:
                    all_files.append(os.path.join(root, file))

        print(f"Found {len(all_files)} files.")

        state = load_state(self._state_path)
        to_process, pending_meta = plan_files(all_files, state)
        to_process_set = set(to_process)
        for path, meta in pending_meta.items():
            if path not in to_process_set:
                state[path] = meta
        if not to_process:
            save_state(self._state_path, state)
            print("No new or changed files to ingest.")
            return
        print(f"Changed files: {len(to_process)}")

        total_chunks = 0
        chunker = RagCoreChunker()
        print(f"Starting ingestion (Parallel C++ Chunking + GPU Embedding)...")
        batch_size = max(self._min_files_per_batch, self._files_per_batch)
        max_batch = max(batch_size, self._max_files_per_batch)
        sizer = AdaptiveBatchSizer(
            current=batch_size,
            min_size=self._min_files_per_batch,
            max_size=max_batch,
            target_seconds=self._target_batch_seconds,
        )
        idx = 0
        total_batches = 0

        while idx < len(to_process):
            current_batch = to_process[idx: idx + sizer.current]
            total_batches += 1
            batch_start = perf_counter()
            had_error = False
            try:
                print(f"Batch {total_batches}: Processing {len(current_batch)} files...")
                chunks = chunker.chunk_files(current_batch, chunk_size, overlap)
                if not chunks:
                    chunks_count = 0
                else:
                    chunks_text = [c.text for c in chunks]
                    chunks_source = [c.source for c in chunks]
                    del chunks
                    gc.collect()

                    embeddings = self._embedder.embed(chunks_text)
                    chunks_count = self._vector_store.add(
                        embeddings, chunks_text, chunks_source
                    )

                    del embeddings
                    del chunks_text
                    del chunks_source
                    gc.collect()

                total_chunks += chunks_count
                for path in current_batch:
                    meta = pending_meta.get(path)
                    if meta is not None:
                        state[path] = meta
                print(f"   -> Completed. {chunks_count} chunks added.")
            except Exception as e:
                _LOGGER.error("Batch %s failed: %s", total_batches, e)
                had_error = True
                idx += len(current_batch)
                gc.collect()
                continue
            finally:
                duration = perf_counter() - batch_start
                if self._adaptive_batching:
                    sizer.record(duration, had_error=had_error)

            idx += len(current_batch)
            gc.collect()

        # Compaction
        try:
            self._vector_store.compact()
        except Exception as exc:
            _LOGGER.warning("Compaction skipped: %s", exc)

        save_state(self._state_path, state)

        print(f"Ingestion complete. Total chunks: {total_chunks}")
        print(f"Total Time: {perf_counter() - start:.3f}s")

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        if top_k <= 0:
            return []
        try:
            vector = self._embedder.embed([query_text])
            if not vector:
                return []
            return self._vector_store.search(vector[0], top_k)
        except (EmbeddingError, VectorStoreError) as exc:
            _LOGGER.exception("Query failed")
            raise QueryError("Query failed") from exc

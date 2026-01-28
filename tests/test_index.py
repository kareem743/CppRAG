from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import json
import logging
import gc
import multiprocessing
import os
from typing import Iterable, List, Optional, Tuple

import lancedb
import rag_core
from fastembed import TextEmbedding

_DB_PATH = "./lancedb_data"
_TABLE_NAME = "vectors"
_STATS_PATH = Path(_DB_PATH) / "ingestion_stats.json"
_LOGGER = logging.getLogger(__name__)


# --- ISOLATED WORKER PROCESS ---
def _worker_process_batch(file_paths: List[str], chunk_size: int, overlap: int, db_path: str, table_name: str) -> int:
    """
    1. Loads C++ Engine to chunk files.
    2. Loads Embedding Model on GPU.
    3. Writes to DB.
    4. Terminates to release 100% of resources.
    """
    import lancedb
    from fastembed import TextEmbedding
    import rag_core
    import gc

    # 1. Chunking (High Speed C++)
    engine = rag_core.IngestionEngine()
    chunks_text = []
    chunks_source = []

    def collector(chunk):
        chunks_text.append(chunk.text)
        chunks_source.append(chunk.source)

    # Process list of files
    engine.process_file_list(file_paths, chunk_size, overlap, collector)

    if not chunks_text:
        return 0

    # 2. Embedding (GPU ACCELERATED)
    # We try to use CUDA. If it fails (no drivers), we fall back to CPU.
    try:
        # RTX 5070 has 12GB VRAM. We can use a large batch size.
        # threads=None lets FastEmbed use all CPU cores for tokenization pre-processing.
        model = TextEmbedding(providers=["CUDAExecutionProvider"])
        embedding_batch_size = 1024
        # print("DEBUG: Using GPU (CUDA)")
    except Exception as e:
        print(f"GPU Init failed, using CPU: {e}")
        model = TextEmbedding()
        embedding_batch_size = 256

    # Convert generator to list immediately to free model memory sooner
    embeddings = [list(vec) for vec in model.embed(chunks_text, batch_size=embedding_batch_size)]

    # Clean up Model VRAM immediately
    del model
    gc.collect()

    # 3. Data Formatting
    data = []
    for i, vec in enumerate(embeddings):
        data.append({
            "vector": vec,
            "text": chunks_text[i],
            "source": chunks_source[i]
        })

    # Release large embedding lists before DB write
    del embeddings
    del chunks_text
    del chunks_source
    gc.collect()

    # 4. Writing (LanceDB)
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    table.add(data)

    return len(data)


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str


class Index:
    def __init__(
            self,
            directory: Path,
            chunk_size: int = 200,
            overlap: int = 50,
            extensions: Optional[Iterable[str]] = None,
    ):
        self._chunks: List[Chunk] = []

        # Ensure Schema exists
        self._ensure_schema()

        # Start Ingestion
        self._ingest(directory, chunk_size, overlap)

    def _ensure_schema(self):
        """Creates table schema if it doesn't exist."""
        db = lancedb.connect(_DB_PATH)
        if _TABLE_NAME not in db.table_names():
            # Initialize with dummy data to set schema
            dummy_model = TextEmbedding()
            dummy_vec = next(dummy_model.embed(["init"]))
            schema_data = [{"vector": list(dummy_vec), "text": "init", "source": "init"}]
            db.create_table(_TABLE_NAME, data=schema_data, mode="overwrite")
            del dummy_model
            gc.collect()

    def _ingest(self, directory: Path, chunk_size: int, overlap: int) -> None:
        start = perf_counter()

        # 1. SCANNING: Python handles file discovery (Low Memory)
        print(f"Scanning {directory} for text files...")
        all_files = []

        # Common text extensions
        valid_exts = {
            ".txt", ".md", ".markdown", ".rst", ".py", ".json",
            ".yaml", ".yml", ".toml", ".csv", ".ts", ".js",
            ".html", ".css", ".cpp", ".cc", ".c", ".h", ".hpp",
            ".java", ".go", ".rs", ".sh"
        }

        for root, _, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in valid_exts:
                    all_files.append(os.path.join(root, file))

        print(f"Found {len(all_files)} files.")

        # 2. BATCHING: Send files to worker in groups
        # 1000 files per batch keeps the C++ Heap size moderate (<1GB)
        # before the worker is killed and restarted.
        FILES_PER_BATCH = 1000
        total_chunks_processed = 0

        # Pool with maxtasksperchild=1 ensures strict memory cleanup
        pool = multiprocessing.Pool(processes=1, maxtasksperchild=1)

        print(f"Starting GPU ingestion in batches of {FILES_PER_BATCH} files...")

        for i in range(0, len(all_files), FILES_PER_BATCH):
            batch_files = all_files[i: i + FILES_PER_BATCH]

            try:
                # Main process waits here. RAM usage should be flat/low.
                chunks_count = pool.apply(
                    _worker_process_batch,
                    (batch_files, chunk_size, overlap, _DB_PATH, _TABLE_NAME)
                )
                total_chunks_processed += chunks_count

                # Progress Log
                batch_num = (i // FILES_PER_BATCH) + 1
                total_batches = (len(all_files) // FILES_PER_BATCH) + 1
                print(f"Batch {batch_num}/{total_batches}: Processed {chunks_count} chunks.")

            except Exception as e:
                _LOGGER.error(f"Batch {i} failed: {e}")

            # Keep Main Process clean
            gc.collect()

        pool.close()
        pool.join()

        # Final Cleanup
        self._compact_db()

        duration = perf_counter() - start
        print(f"Ingestion complete. Total chunks: {total_chunks_processed}")
        print(f"Total Time: {duration:.3f}s")

    def _compact_db(self):
        print("Optimizing database storage...")
        try:
            db = lancedb.connect(_DB_PATH)
            if _TABLE_NAME in db.table_names():
                tbl = db.open_table(_TABLE_NAME)
                tbl.compact_files()
                tbl.cleanup_old_versions()
        except Exception as e:
            print(f"Compaction note: {e}")

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        if top_k <= 0: return []

        # Use CPU for single query (faster latency than loading GPU for 1 item)
        model = TextEmbedding()
        vector = next(model.embed([query_text]), None)
        del model

        if vector is None: return []

        db = lancedb.connect(_DB_PATH)
        if _TABLE_NAME not in db.table_names(): return []

        table = db.open_table(_TABLE_NAME)
        results = table.search(list(vector)).limit(top_k).to_list()

        output = []
        for row in results:
            # Reconstruct Chunk object
            c = Chunk(text=row.get("text", ""), source=row.get("source", ""))
            score = float(row.get("_distance", 0.0))
            output.append((c, score))
        return output

    def _log_timing(self, label: str, seconds: float) -> None:
        print(f"{label}: {seconds:.3f}s")
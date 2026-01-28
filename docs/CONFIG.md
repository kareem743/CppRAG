# Configuration

The project reads configuration from:
1) YAML config file (optional)
2) Environment variables
3) CLI overrides (highest priority)

The same structure is used by `rag_cli.py` and `run_eval.py`.

## YAML Config

Pass via `--config <path>` or environment variable `RAG_CONFIG`.

Expected structure (all keys optional):
```
embedding:
  prefer_gpu: true
  max_retries: 1
  gpu_batch_size: 32
  cpu_batch_size: 256

vector_store:
  db_path: "./lancedb_data"
  table_name: "vectors"

ingestion:
  chunk_size: 200
  overlap: 50
  extensions: ["md", "py"]
  files_per_batch: 5000
  adaptive_batching: true
  min_files_per_batch: 100
  max_files_per_batch: 8000
  target_batch_seconds: 20.0

query:
  top_k: 3

llm:
  model: "llama3"
  timeout_seconds: 60

logging:
  level: "INFO"

server:
  host: "127.0.0.1"
  port: 8000
```

## Environment Variables

```
RAG_LOG_LEVEL
RAG_DB_PATH
RAG_TABLE_NAME
RAG_CHUNK_SIZE
RAG_OVERLAP
RAG_EXTENSIONS
RAG_FILES_PER_BATCH
RAG_ADAPTIVE_BATCHING
RAG_MIN_FILES_PER_BATCH
RAG_MAX_FILES_PER_BATCH
RAG_TARGET_BATCH_SECONDS
RAG_TOP_K
RAG_MODEL
RAG_TIMEOUT_SECONDS
RAG_PREFER_GPU
RAG_EMBED_MAX_RETRIES
RAG_EMBED_GPU_BATCH_SIZE
RAG_EMBED_CPU_BATCH_SIZE
RAG_SERVER_HOST
RAG_SERVER_PORT
```

## CLI Overrides

Supported by both `rag_cli.py` and `run_eval.py`:
```
--chunk-size
--overlap
--extensions
--files-per-batch
--adaptive-batching / --no-adaptive-batching
--min-files-per-batch
--max-files-per-batch
--target-batch-seconds
--top-k
--model
--db-path
--table-name
--prefer-gpu / --no-prefer-gpu
--embed-retries
--embed-gpu-batch
--embed-cpu-batch
--log-level
```

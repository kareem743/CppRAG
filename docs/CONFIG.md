# Configuration

Configuration is loaded in this order:

## `rag_cli.py`
1) YAML config file (optional)
2) Environment variables
3) CLI overrides (highest priority)

`rag_cli.py` supports `RAG_CONFIG` as a fallback when `--config` is not provided.

## `run_eval.py`
1) YAML config file (optional, via `--config`)
2) Environment variables
3) CLI overrides (highest priority)

`run_eval.py` does not read `RAG_CONFIG`; it only uses `--config`.

The same schema is shared by `rag_cli.py` and `run_eval.py`.

## YAML Config

Defaults shown below match the code:
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
  extensions: null
  files_per_batch: 1500
  adaptive_batching: true
  min_files_per_batch: 1
  max_files_per_batch: 8000
  target_batch_seconds: 20.0

query:
  top_k: 5

llm:
  model: "llama3"
  timeout_seconds: 60

logging:
  level: "INFO"

server:
  host: "127.0.0.1"
  port: 8000
```

Validation rules:
- `chunk_size` must be > 0
- `overlap` must be >= 0 and < `chunk_size`
- batch sizes must be >= 1
- `target_batch_seconds` must be > 0
- `server.port` must be 1..65535

## Environment Variables

```
RAG_CONFIG
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

Notes:
- `RAG_CONFIG` is only read by `rag_cli.py` (not `run_eval.py`).
- `RAG_EXTENSIONS` is comma-separated (e.g. `md,py,cpp`).
- Boolean env vars use `"true"` / `"false"`.
- `server.host` and `server.port` are defined in config but not used by the current CLI.

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

Note: `run_eval.py` is a print-based script, so `--log-level` only affects config
values (it does not change output verbosity).

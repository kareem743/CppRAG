# CLI Usage

The CLI lives in `rag_cli.py` and exposes three commands.

---

## Command Map

| Command | Purpose |
| --- | --- |
| `ingest` | Build or update the LanceDB index from a directory |
| `query` | Answer a single question |
| `serve` | Interactive question loop |

All commands support `--config` and `--verbose`.

## Ingest (Build or Update an Index)

```
python rag_cli.py ingest <DIRECTORY>
```

Key behaviors (from code):
- Incremental ingestion uses `lancedb_data/ingestion_state.json` by default.
- Only changed files are re-chunked and re-embedded.
- Batch size can auto-adjust when `--adaptive-batching` is enabled.

Common overrides:
```
python rag_cli.py ingest <DIRECTORY> \
  --chunk-size 200 \
  --overlap 50 \
  --extensions "md,py,cpp" \
  --files-per-batch 1500 \
  --adaptive-batching
```

Dry run:
```
python rag_cli.py ingest <DIRECTORY> --dry-run
```

## Query (Single Question)

```
python rag_cli.py query <DIRECTORY> "What does this project do?"
```

Notes:
- `query` builds the index before answering (uses incremental ingestion).

Overrides:
```
python rag_cli.py query <DIRECTORY> "Question" --top-k 5 --model llama3
```

## Serve (Interactive)

```
python rag_cli.py serve <DIRECTORY>
```

Type `exit` or `quit` to end the loop.

---

## CLI Flags (Summary)

### Shared
- `--config`, `-c`: YAML config file path
- `--verbose`, `-v`: enable debug logging

### Ingest-only
- `--chunk-size`, `--overlap`
- `--extensions` (comma-separated)
- `--files-per-batch`
- `--adaptive-batching` / `--no-adaptive-batching`
- `--min-files-per-batch`, `--max-files-per-batch`
- `--target-batch-seconds`
- `--db-path`, `--table-name`
- `--prefer-gpu` / `--no-prefer-gpu`
- `--embed-retries`
- `--embed-gpu-batch`, `--embed-cpu-batch`
- `--log-level`: override log level (INFO, DEBUG, etc.)
- `--dry-run`

### Query
- `--top-k`
- `--model`
- `--chunk-size`, `--overlap`, `--extensions` (rebuilds index as needed)
- `--files-per-batch`, `--adaptive-batching`, `--min-files-per-batch`,
  `--max-files-per-batch`, `--target-batch-seconds`
- `--db-path`, `--table-name`
- `--log-level`
- `--dry-run`

### Serve
- `--config`, `--dry-run`, `--verbose`

---

## Default File Extensions

If no `--extensions` are provided, ingestion includes:
```
.txt .md .markdown .rst .py .json .yaml .yml .toml .csv .ts .js .html .css
.cpp .cc .c .h .hpp .java .go .rs .sh
```

## Dependencies Used At Runtime

- Python 3.11+
- `rag_core.pyd` (C++ extension)
- `fastembed`, `lancedb`, `typer`
- Ollama installed and a local model pulled (default model: `llama3`)

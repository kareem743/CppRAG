# Usage

This project exposes a CLI in `rag_cli.py` with three commands:
- `ingest`: build/update the vector index from a directory.
- `query`: answer a single question.
- `serve`: interactive question loop.

All commands accept `--config` and the same overrides (see `docs/CONFIG.md`).

## Ingest

```
python rag_cli.py ingest <DIRECTORY>
```

Common overrides:
```
python rag_cli.py ingest <DIRECTORY> --chunk-size 300 --overlap 50 --extensions "md,py"
```

Notes:
- Ingestion scans files by extension (defaults include `.md`, `.py`, `.cpp`, etc).
- Chunks are created by the C++ chunking engine.
- Embeddings are computed with `fastembed`.
- Vectors are stored in LanceDB (`./lancedb_data` by default).

## Query (single question)

```
python rag_cli.py query <DIRECTORY> "What does this project do?"
```

Overrides:
```
python rag_cli.py query <DIRECTORY> "Question" --top-k 5 --model llama3
```

## Serve (interactive)

```
python rag_cli.py serve <DIRECTORY>
```

Type `exit` or `quit` to end the loop.

## Environment Dependencies

- Python 3.x
- `rag_core.pyd` (C++ extension present in repo)
- `fastembed`, `lancedb`
- Ollama installed and a local model pulled (default model: `llama3`)

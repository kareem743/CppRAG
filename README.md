# Local RAG 

Local retrieval-augmented question answering over your files. This project uses a
C++ chunker (`rag_core.pyd`), FastEmbed embeddings, LanceDB storage, and a local
Ollama model for answers.

---

## At a Glance

Fast, local RAG pipeline you can run on a directory of files to get answers with
source snippets.

## What You Get

- Incremental ingestion with a file-hash cache (`lancedb_data/ingestion_state.json`).
- C++ chunking via `rag_core.IngestionEngine` (uses the parallel API when available).
- GPU-first embeddings with automatic CPU fallback.
- LanceDB vector store with post-ingestion compaction.
- CLI commands with structured logging and timing.
- An evaluation script with retrieval, generation, and end-to-end metrics.

## Who It's For

- Teams that need fast local search over code or docs.
- Projects that want repeatable ingestion and evaluation.
- Anyone who prefers local LLMs over hosted APIs.

## Quick Start

1) (Optional) Activate the virtualenv:
```
.\.venv\Scripts\activate
```

2) Ingest a directory:
```
python rag_cli.py ingest <PATH_TO_DOCS>
```

3) Ask a question:
```
python rag_cli.py query <PATH_TO_DOCS> "Your question here"
```

4) Interactive mode:
```
python rag_cli.py serve <PATH_TO_DOCS>
```

## Common CLI Examples

```
python rag_cli.py ingest <DIR> --chunk-size 200 --overlap 50 --extensions "md,py"
python rag_cli.py query <DIR> "Question" --top-k 5 --model llama3
python rag_cli.py ingest <DIR> --dry-run --verbose
```

## How Answers Are Produced

1) Files are scanned and filtered by extension.
2) Changed files are detected via a hash cache.
3) The C++ chunker splits text into chunks.
4) FastEmbed creates embeddings (GPU-first, CPU fallback).
5) LanceDB stores vectors, text, and source paths.
6) A query embeds your question, retrieves top-k chunks, and calls `ollama run <model>`.

## Project Layout

```
README.md
rag_cli.py
run_eval.py
rag/
docs/
```

## Docs

- `docs/INDEX.md` — Documentation start page.
- `docs/USAGE.md` — CLI usage and flags.
- `docs/CONFIG.md` — Config file, env vars, and overrides.
- `docs/EVALUATION.md` — Evaluation workflow and metrics.
- `docs/DATASET.md` — Golden dataset schema and guidance.
- `docs/ARCHITECTURE.md` — Modules and data flow.
- `docs/TROUBLESHOOTING.md` — Common issues and fixes.

## Requirements

- Python 3.11+
- `rag_core.pyd` in the project root
- `fastembed`, `lancedb`, `typer`, `pydantic`, `pyyaml`
- Ollama installed with a local model pulled (default model: `llama3`)

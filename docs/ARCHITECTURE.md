# Architecture

This document describes what the code currently does.

---

## High-Level Flow

### Ingestion
1) `rag_cli.py ingest` builds an `Index`.
2) `Index` scans files and compares against `ingestion_state.json`.
3) `RagCoreChunker` uses the C++ engine (`rag_core.IngestionEngine`).
4) `FastEmbedEmbedder` embeds chunks (GPU-first, CPU fallback).
5) `LanceDBVectorStore` stores vectors, text, and source paths.
6) Vector store compaction is attempted after ingestion.

### Query
1) `rag_cli.py query` builds an `Index` (incremental ingest may run).
2) `Index.query()` embeds the question and searches LanceDB.
3) `RAGSystem` formats retrieved chunks into a prompt with `[Source: ...]` tags.
4) `OllamaLLM` calls `ollama run <model>` to generate the answer.

### Serve (Interactive)
- `rag_cli.py serve` builds an `Index` and then loops on a terminal prompt.

---

## Core Modules

| Module | Responsibility |
| --- | --- |
| `rag/index.py` | Orchestrates ingestion/query, batch sizing, and vector store ops |
| `rag/ingestion_state.py` | Tracks file hashes and metadata |
| `rag/batch_sizer.py` | Adaptive batch sizing logic |
| `rag/chunker.py` | C++ chunker wrapper |
| `rag/embedders.py` | FastEmbed adapter with GPU/CPU fallback |
| `rag/vector_store.py` | LanceDB schema, insert, search, compaction |
| `rag/rag.py` | Prompt builder and answer orchestration |
| `rag/llm.py` | `OllamaLLM` wrapper |
| `rag/logging_utils.py` | Structured logging + timing decorator |
| `rag/config.py` | Config loading and validation |
| `rag/interfaces.py` / `rag/errors.py` | Shared interfaces and errors |

---

## Indexing Files

Default extensions (override with `--extensions` or `RAG_EXTENSIONS`):
```
.txt .md .markdown .rst .py .json .yaml .yml .toml .csv .ts .js .html .css
.cpp .cc .c .h .hpp .java .go .rs .sh
```

---

## State and Storage

- Vectors live in `./lancedb_data` by default.
- Incremental ingestion uses `./lancedb_data/ingestion_state.json`.

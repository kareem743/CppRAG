# Architecture

This document describes what the code currently does.

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

## Core Modules

- `rag/index.py`  
  Orchestrates ingestion and query. Handles file scanning, incremental state,
  adaptive batch sizing, chunking, embedding, and LanceDB operations.

- `rag/ingestion_state.py`  
  Tracks file hashes and metadata to skip unchanged files.

- `rag/batch_sizer.py`  
  Adaptive batch sizing based on recent batch durations.

- `rag/chunker.py`  
  Wrapper for the C++ chunker (`rag_core.IngestionEngine`).

- `rag/embedders.py`  
  FastEmbed adapter with GPU/CPU fallback and retry logic.

- `rag/vector_store.py`  
  LanceDB adapter for schema creation, insertion, search, and compaction.

- `rag/rag.py`  
  Builds the prompt and calls the local LLM for a final answer.

- `rag/llm.py`  
  `OllamaLLM` wrapper that calls `ollama run <model>`.

- `rag/logging_utils.py`  
  Key-value structured logging and a timing decorator.

- `rag/config.py`  
  Configuration loading and validation (YAML, env, CLI).

- `rag/interfaces.py` / `rag/errors.py`  
  Shared interfaces and error types.

## Indexing Files

Default extensions (override with `--extensions` or `RAG_EXTENSIONS`):
```
.txt .md .markdown .rst .py .json .yaml .yml .toml .csv .ts .js .html .css
.cpp .cc .c .h .hpp .java .go .rs .sh
```

## State and Storage

- Vectors live in `./lancedb_data` by default.
- Incremental ingestion uses `./lancedb_data/ingestion_state.json`.

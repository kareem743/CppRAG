# Architecture

## High-Level Flow

Ingestion:
1) `rag_cli.py ingest` builds an `Index`.
2) `Index` scans files and uses the C++ chunker (`rag_core`) to chunk text.
3) `FastEmbedEmbedder` embeds each chunk.
4) `LanceDBVectorStore` stores vectors + text + source.

Query:
1) `rag_cli.py query` builds the `Index` and runs `Index.query(question)`.
2) `Index.query` embeds the question and searches LanceDB.
3) `RAGSystem` formats retrieved chunks into context.
4) `OllamaLLM` generates an answer with citations.

## Core Modules

- `rag/index.py`  
  Orchestrates ingestion and querying. Handles file scanning, batch sizing, chunking, embedding, and LanceDB storage.

- `rag/chunker.py`  
  Wraps the C++ chunker (`rag_core.IngestionEngine`).

- `rag/embedders.py`  
  Embedding adapter using `fastembed` with GPU/CPU fallback.

- `rag/vector_store.py`  
  LanceDB adapter for schema creation, vector insertion, and search.

- `rag/rag.py`  
  `RAGSystem` that formats context and calls the LLM.

- `rag/llm.py`  
  `OllamaLLM` wrapper using `ollama run <model>`.

- `rag/models.py`  
  `Chunk` dataclass storing `text` and `source`.

- `rag/config.py`  
  Configuration loading from YAML + environment + CLI overrides.

## Indexing Files

By default, ingestion includes these extensions:
```
.txt .md .markdown .rst .py .json .yaml .yml .toml .csv .ts .js .html .css
.cpp .cc .c .h .hpp .java .go .rs .sh
```

Override with `--extensions "md,py"` or `RAG_EXTENSIONS`.

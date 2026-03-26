<div align="center">

# 🔍 Local RAG

**Fast, fully local retrieval-augmented question answering over your files.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-server-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LanceDB](https://img.shields.io/badge/LanceDB-vector%20store-F5A623?style=flat-square)](https://lancedb.com/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?style=flat-square)](https://ollama.com/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

*C++ chunking · GPU-first embeddings · incremental ingestion · no data leaves your machine*

</div>

---

## ✨ Features

| | Feature | Details |
|---|---|---|
| ⚡ | **C++ Chunker** | `rag_core.IngestionEngine` with parallel API support |
| 🔁 | **Incremental Ingestion** | File-hash cache skips unchanged files automatically |
| 🎮 | **GPU-first Embeddings** | FastEmbed with automatic CPU fallback |
| 🗄️ | **Vector Store** | LanceDB with post-ingestion compaction |
| 🖥️ | **Three Interfaces** | CLI · FastAPI server · Vite frontend |
| 📊 | **Evaluation Suite** | Retrieval, generation, and end-to-end metrics |

---

## 🚀 Quick Start

### CLI

```bash
# 1. (Optional) activate the virtualenv
.\.venv\Scripts\activate

# 2. Ingest a directory
python rag_cli.py ingest <PATH_TO_DOCS>

# 3. Ask a question
python rag_cli.py query <PATH_TO_DOCS> "Your question here"

# 4. Interactive mode
python rag_cli.py serve <PATH_TO_DOCS>
```

### API + Frontend

```bash
# 1. Start the API server
python server.py

# 2. Start the frontend
cd frontend
npm install
npm run dev
```

Then open the URL printed by Vite — usually **http://localhost:5173**.  
The frontend expects the API at `http://127.0.0.1:8000`.

<details>
<summary>⚙️ Optional environment overrides</summary>

```bash
set RAG_DIRECTORY=C:\path\to\docs
set RAG_CONFIG=C:\path\to\config.yaml
```

</details>

---

## 🔄 How It Works

```
 Your Files
     │
     ▼
 ① Scan & filter by extension
     │
     ▼
 ② Detect changes via hash cache
     │
     ▼
 ③ C++ chunker splits text into chunks
     │
     ▼
 ④ FastEmbed creates embeddings  (GPU → CPU fallback)
     │
     ▼
 ⑤ LanceDB stores vectors, text & source paths
     │
     ▼
 ⑥ Query → embed → retrieve top-k → Ollama → Answer ✅
```

---

## 🛠️ Common CLI Options

```bash
# Custom chunk size and file types
python rag_cli.py ingest <DIR> --chunk-size 200 --overlap 50 --extensions "md,py"

# Use a specific model and retrieve more chunks
python rag_cli.py query <DIR> "Question" --top-k 5 --model llama3

# Dry run to preview what would be ingested
python rag_cli.py ingest <DIR> --dry-run --verbose
```

---

## 🌐 API Reference

Base URL: `http://127.0.0.1:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Ask a question — body: `{ "question": "...", "top_k": 5 }` |
| `POST` | `/api/ingest` | Start ingestion with optional overrides and `dry_run` |
| `GET` | `/api/ingest/status` | Poll current ingestion progress |
| `GET` | `/api/visualize?filepath=...` | View chunk spans for a specific file |
| `GET` | `/api/stats` | System stats |

---

## 📁 Project Layout

```
local-rag/
├── rag_cli.py          # CLI entrypoint
├── server.py           # FastAPI server
├── run_eval.py         # Evaluation script
├── rag_core.pyd        # C++ chunking engine
├── rag/                # Core pipeline modules
├── frontend/           # Vite + JS frontend
└── docs/
    ├── INDEX.md        # Documentation start page
    ├── USAGE.md        # CLI usage and flags
    ├── CONFIG.md       # Config file, env vars, overrides
    ├── EVALUATION.md   # Evaluation workflow and metrics
    ├── DATASET.md      # Golden dataset schema
    ├── ARCHITECTURE.md # Modules and data flow
    └── TROUBLESHOOTING.md
```

---

## 📦 Requirements

- **Python** 3.11+
- `rag_core.pyd` in the project root
- Python packages: `fastembed` `lancedb` `typer` `pydantic` `pyyaml`
- [Ollama](https://ollama.com/) installed with a model pulled (default: `llama3`)

```bash
pip install fastembed lancedb typer pydantic pyyaml
ollama pull llama3
```

---

<div align="center">

Built for teams that want **fast local search** over code and docs — no cloud required.

</div>
# RAG Project (PythonProject4)

This project provides a local RAG (Retrieval-Augmented Generation) pipeline with:
- A CLI for ingestion, querying, and interactive use.
- A C++ chunking engine (`rag_core.pyd`) integrated via `rag_core`.
- A LanceDB vector store backend.
- A standalone evaluation script (`run_eval.py`) to measure retrieval, generation, and end-to-end quality.

Quick links:
- `docs/USAGE.md` — How to run ingestion/query/serve.
- `docs/CONFIG.md` — Configuration and overrides.
- `docs/EVALUATION.md` — Evaluation workflow + metrics.
- `docs/DATASET.md` — Golden dataset schema and guidance.
- `docs/ARCHITECTURE.md` — Module overview and data flow.
- `docs/TROUBLESHOOTING.md` — Common errors and fixes.

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

## Evaluation (Summary)

Generate a dataset skeleton:
```
python run_eval.py generate-dataset --source-dir <PATH_TO_DOCS> --output golden_dataset.json --num-questions 50
```

Run evaluation:
```
python run_eval.py evaluate --dataset golden_dataset.json --index-dir <PATH_TO_DOCS> --output-dir eval_results
```

See `docs/EVALUATION.md` for details.

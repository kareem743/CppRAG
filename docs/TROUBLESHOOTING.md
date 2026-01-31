# Troubleshooting

## "Index not ready" or "table not found"

Cause:
- No ingestion has been run, or the LanceDB table is empty.

Fix:
```
python rag_cli.py ingest <DIRECTORY>
```

## "No new or changed files to ingest."

Cause:
- Incremental ingestion determined nothing changed.

Fix:
- Touch or edit files to force re-ingestion, or delete
  `lancedb_data/ingestion_state.json` to rebuild from scratch.

## Ollama errors ("ollama" not found / model missing)

Cause:
- Ollama is not installed or the model is not pulled.

Fix:
```
ollama pull llama3
```

## GPU embedding failures

Cause:
- GPU provider not available for `fastembed`.

Fix:
- Use CPU embeddings:
```
python rag_cli.py ingest <DIRECTORY> --no-prefer-gpu
```

## C++ chunker errors

Cause:
- `rag_core.pyd` missing or incompatible with your Python version.

Fix:
- Ensure `rag_core.pyd` exists in the project root.
- Confirm the Python version matches the compiled extension.

## Slow ingestion

Cause:
- Large files, slow disk, or aggressive batch sizes.

Fix:
- Lower `--files-per-batch` or set `--target-batch-seconds` to a smaller value.
- Disable adaptive batching to debug:
```
python rag_cli.py ingest <DIRECTORY> --no-adaptive-batching
```

## Dataset validation errors

Cause:
- Missing required fields in `golden_dataset.json`.

Fix:
- Ensure each entry has all required keys (see `docs/DATASET.md`).

## Slow evaluation

Cause:
- LLM judge prompts can be slow with large contexts.

Fix:
- Reduce dataset size for quick checks.
- Use smaller chunk sizes or fewer sources per question.

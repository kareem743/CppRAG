# Troubleshooting

## "Index not ready" or "table not found"

Cause:
- No ingestion has been run, or the LanceDB table is empty.

Fix:
```
python rag_cli.py ingest <DIRECTORY>
```

## Ollama errors ("ollama" not found / model missing)

Cause:
- Ollama is not installed or the model isn't pulled.

Fix:
- Install Ollama and pull the model:
```
ollama pull llama3
```

## GPU embedding failures

Cause:
- GPU provider not available.

Fix:
- Set `--prefer-gpu false` or `RAG_PREFER_GPU=false` to force CPU embeddings.

## C++ chunker errors

Cause:
- `rag_core.pyd` not found or incompatible.

Fix:
- Ensure `rag_core.pyd` exists in the project root.
- Confirm the Python version matches the compiled extension.

## Dataset validation errors

Cause:
- Missing required fields in `golden_dataset.json`.

Fix:
- Ensure each entry has all required keys (see `docs/DATASET.md`).

## Slow evaluation

Cause:
- LLM judge prompts can be slow, especially with large contexts.

Fix:
- Reduce dataset size for quick checks.
- Use smaller chunk sizes or fewer sources per question.

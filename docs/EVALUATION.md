# Evaluation

Evaluation is run via `run_eval.py`. It produces JSON results plus an optional
comparison report.

---

## Commands

Generate a dataset skeleton:
```
python run_eval.py generate-dataset \
  --source-dir <PATH_TO_DOCS> \
  --output golden_dataset.json \
  --num-questions 50
```

Run evaluation:
```
python run_eval.py evaluate \
  --dataset golden_dataset.json \
  --index-dir <PATH_TO_DOCS> \
  --output-dir eval_results
```

Compare to baseline:
```
python run_eval.py evaluate \
  --dataset golden_dataset.json \
  --index-dir <PATH_TO_DOCS> \
  --compare-baseline
```

Set a new baseline:
```
python run_eval.py evaluate \
  --dataset golden_dataset.json \
  --index-dir <PATH_TO_DOCS> \
  --set-baseline
```

Or set a baseline from an existing results file:
```
python run_eval.py set-baseline --results eval_results/latest.json
```

---

## What Happens During Evaluation

1) Index readiness is checked (LanceDB table exists and has data).
2) `Index` is created for `--index-dir` (incremental ingest may run).
3) Retrieval metrics are computed from vector search.
4) Generation metrics use perfect context built from ground-truth sources.
5) End-to-end metrics call `RAGSystem.answer()` for every question.

---

## Key Metrics

### Retrieval
- Hit Rate @ K
- MRR
- Precision @ K
- Context Recall (multi-source)
- Latency (P50/P95/P99 + avg)
- Embed vs search timing
- Failure list (per question)

### Generation (Perfect Context)
- Faithfulness score (1-5)
- Completeness score (1-5)
- Citation accuracy
- Latency (avg + P95)
- Failure list (LLM or judge errors)

### End-to-End
- Semantic similarity (cosine over embeddings)
- Correctness score (1-5)
- "I don't know" accuracy for negative questions
- Latency (avg + P95)
- Failure list

---

## Output Files

`eval_results/latest.json`:
- Full metrics + per-question data.

`eval_results/baseline.json`:
- Snapshot used for regression checks (created with `--set-baseline`).

`eval_results/comparison_report.txt`:
- Baseline vs latest deltas + regression warnings (created with `--compare-baseline`).

---

## Exit Codes

- `0`: Success, no regression detected.
- `1`: Regression detected vs baseline (only when `--compare-baseline` is used).
- `2`: Fatal error (missing dataset/index/etc).

Regression rules (from code):
- Hit Rate drop > 0.10 -> CRITICAL (exit 1)
- MRR drop > 0.10 -> WARNING (exit 1)
- Avg latency increase > 50% -> PERFORMANCE REGRESSION (exit 1)

---

## Useful Flags

```
--extensions "md,py,cpp"
--num-questions 50
--negative-count 5
--snippet-chars 240
--seed 7
--top-k 5
--model llama3
--prefer-gpu true
```

## Notes

- The evaluator calls Ollama for both answers and judge prompts.
- Judge-based metrics are skipped if those LLM calls fail.
- If the LanceDB table is missing or empty, evaluation exits with a clear message.

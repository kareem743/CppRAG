# Evaluation

The evaluation system is a standalone script: `run_eval.py`. It produces:
- `eval_results/latest.json`
- `eval_results/baseline.json` (when set)
- `eval_results/comparison_report.txt` (when comparing)

## Commands

Generate a dataset skeleton (manual verification required):
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

## Phases

1) Retrieval Quality  
Uses `Index.query()` only. No LLM calls.

2) Generation Quality  
Builds perfect context directly from `ground_truth_sources` and calls the LLM.

3) End-to-End  
Calls `rag_system.answer()` and compares with ground truth.

## Metrics

### Retrieval
- Hit Rate @ K
- MRR
- Precision @ K
- Context Recall (multi-source)
- Latency (P50/P95/P99 + avg)
- Embedding vs search timing

### Generation (Perfect Context)
- Faithfulness score (1–5)
- Completeness score (1–5)
- Citation accuracy
- Latency

### End-to-End
- Semantic similarity (cosine over embeddings)
- Correctness score (1–5)
- "I don't know" accuracy (negative questions)
- Latency

## Output Files

`eval_results/latest.json`:
- Full metrics + per-question data.

`eval_results/baseline.json`:
- Snapshot used for regressions.

`eval_results/comparison_report.txt`:
- Baseline vs latest deltas + regression warnings.

## Exit Codes

- `0`: Success, no regression detected.
- `1`: Regression detected vs baseline.
- `2`: Fatal error (missing dataset/index/etc).

Regression rules:
- Hit Rate drop > 0.10 → CRITICAL (exit 1)
- MRR drop > 0.10 → WARNING (exit 1)
- Avg latency increase > 50% → PERFORMANCE REGRESSION (exit 1)

## Notes

- The evaluator uses Ollama for answer generation and judge prompts.
- If LLM calls fail, judge-based metrics are skipped but other metrics continue.
- If the LanceDB table is missing or empty, evaluation will exit with a clear message.

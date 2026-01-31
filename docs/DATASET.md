# Golden Dataset

The golden dataset is the ground truth for evaluation. `run_eval.py` accepts:

- JSON array (recommended for editing)
- JSONL (one object per line)

## Schema

```
{
  "question": "What is the chunk size used in the C++ engine?",
  "ground_truth_answer": "The chunk_size parameter is passed to the chunking function",
  "ground_truth_sources": ["rag_core.cpp"],
  "chunk_text_snippet": "std::vector<std::string> chunk_text(const std::string& text, int chunk_size, int overlap)",
  "question_type": "factual|multi-hop|negative",
  "expected_chunk_count": 1
}
```

## Required Fields

- `question`: The evaluation question.
- `ground_truth_answer`: The correct answer, or "I don't know" for negative questions.
- `ground_truth_sources`: List of source files that contain the answer.
- `chunk_text_snippet`: Snippet from the exact chunk that should answer the question.
- `question_type`: One of `factual`, `multi-hop`, `negative`.
- `expected_chunk_count`: Number of chunks expected to contain the answer.

## Creation Guidance

Minimum requirements:
- 50 to 100 entries.
- At least 20 manually curated questions.
- At least 5 negative questions (not in the corpus).
- Rough mix: 60% factual, 25% multi-hop, 15% negative.

Validation rules (enforced by code):
- No duplicate questions (recommended).
- Each entry must include all required fields.
- If the answer is not in the corpus, use `question_type = "negative"` and `ground_truth_sources = []`.

## Generator

`run_eval.py generate-dataset` creates a skeleton with real snippets and TODO placeholders.
You must manually replace the TODO fields before evaluating.

# Golden Dataset

The golden dataset is the ground truth for evaluation. It can be stored as:
- JSON array (recommended for editing)
- JSONL (one object per line)

## Schema

```
{
  "question": "What is the chunk size used in the C++ engine?",
  "ground_truth_answer": "The chunk_size parameter is passed to the tokenize and chunk_text functions",
  "ground_truth_sources": ["rag_core.cpp"],
  "chunk_text_snippet": "std::vector<std::string> chunk_text(const std::string& text, int chunk_size, int overlap)",
  "question_type": "factual|multi-hop|negative",
  "expected_chunk_count": 1
}
```

## Required Fields

- `question`: The evaluation question.
- `ground_truth_answer`: The correct answer (or "I don't know" for negative questions).
- `ground_truth_sources`: List of source files that contain the answer.
- `chunk_text_snippet`: A snippet from the exact chunk that should answer the question.
- `question_type`: One of `factual`, `multi-hop`, `negative`.
- `expected_chunk_count`: Number of chunks expected in retrieval.

## Creation Guidance

Minimum requirements:
- 50–100 entries.
- At least 20 manually curated questions.
- At least 5 negative questions (not in the corpus).
- 60% factual, 25% multi-hop, 15% negative.

Validation rules:
- No duplicate questions.
- Each entry must be manually verified against the source.
- If the answer is not in the corpus, mark `question_type = "negative"` and use `ground_truth_sources = []`.

## Generator

`run_eval.py generate-dataset` creates a skeleton with real snippets and TODO placeholders.
You must manually replace the TODO fields.

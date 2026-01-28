import argparse
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Tuple

from rag.config import AppConfig, build_config
from rag.embedders import FastEmbedEmbedder
from rag.index import Index
from rag.llm import OllamaLLM
from rag.rag import RAGSystem
from rag.vector_store import LanceDBVectorStore
from rag.chunker import RagCoreChunker


DATASET_REQUIRED_FIELDS = {
    "question",
    "ground_truth_answer",
    "ground_truth_sources",
    "chunk_text_snippet",
    "question_type",
    "expected_chunk_count",
}


@dataclass
class RetrievalTiming:
    embed_ms: float
    search_ms: float
    total_ms: float


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset(path: Path) -> List[Dict]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.lstrip().startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("Dataset JSON must be a list of objects.")
        entries = data
    else:
        entries = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    for idx, entry in enumerate(entries, start=1):
        missing = DATASET_REQUIRED_FIELDS - set(entry.keys())
        if missing:
            raise ValueError(f"Dataset entry {idx} missing fields: {sorted(missing)}")
    return entries


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/").lower()


def source_matches(expected: str, observed: str) -> bool:
    if not expected or not observed:
        return False
    expected_norm = _normalize_path(expected)
    observed_norm = _normalize_path(observed)
    if observed_norm == expected_norm:
        return True
    if observed_norm.endswith("/" + expected_norm) or observed_norm.endswith(expected_norm):
        return True
    if os.path.basename(observed_norm) == os.path.basename(expected_norm):
        return True
    return False


def build_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}

    def _set(path: List[str], value) -> None:
        node = overrides
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value

    if args.chunk_size is not None:
        _set(["ingestion", "chunk_size"], args.chunk_size)
    if args.overlap is not None:
        _set(["ingestion", "overlap"], args.overlap)
    if args.extensions is not None:
        _set(["ingestion", "extensions"], _parse_extensions(args.extensions))
    if args.files_per_batch is not None:
        _set(["ingestion", "files_per_batch"], args.files_per_batch)
    if args.adaptive_batching is not None:
        _set(["ingestion", "adaptive_batching"], args.adaptive_batching)
    if args.min_files_per_batch is not None:
        _set(["ingestion", "min_files_per_batch"], args.min_files_per_batch)
    if args.max_files_per_batch is not None:
        _set(["ingestion", "max_files_per_batch"], args.max_files_per_batch)
    if args.target_batch_seconds is not None:
        _set(["ingestion", "target_batch_seconds"], args.target_batch_seconds)
    if args.top_k is not None:
        _set(["query", "top_k"], args.top_k)
    if args.model is not None:
        _set(["llm", "model"], args.model)
    if args.db_path is not None:
        _set(["vector_store", "db_path"], args.db_path)
    if args.table_name is not None:
        _set(["vector_store", "table_name"], args.table_name)
    if args.prefer_gpu is not None:
        _set(["embedding", "prefer_gpu"], args.prefer_gpu)
    if args.embed_retries is not None:
        _set(["embedding", "max_retries"], args.embed_retries)
    if args.embed_gpu_batch is not None:
        _set(["embedding", "gpu_batch_size"], args.embed_gpu_batch)
    if args.embed_cpu_batch is not None:
        _set(["embedding", "cpu_batch_size"], args.embed_cpu_batch)
    if args.log_level is not None:
        _set(["logging", "level"], args.log_level)
    return overrides


def _parse_extensions(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return parts or None


def resolve_config(config_path: Optional[str], overrides: dict) -> AppConfig:
    return build_config(config_path, overrides)


def _index_ready(db_path: str, table_name: str) -> Tuple[bool, str]:
    try:
        import lancedb
    except Exception as exc:
        return False, f"Failed to import lancedb: {exc}"
    if not os.path.exists(db_path):
        return False, f"Vector store path not found: {db_path}"
    try:
        db = lancedb.connect(db_path)
        if table_name not in db.table_names():
            return False, f"Table '{table_name}' not found in {db_path}"
        table = db.open_table(table_name)
        try:
            count = table.count_rows()
        except Exception:
            count = len(table.to_list())
        if count <= 1:
            return False, "Vector table has no ingested data. Run ingestion first."
    except Exception as exc:
        return False, f"Failed to inspect LanceDB: {exc}"
    return True, ""


def _timed_query(
    embedder: FastEmbedEmbedder,
    vector_store: LanceDBVectorStore,
    query_text: str,
    top_k: int,
) -> Tuple[List[Tuple[object, float]], RetrievalTiming]:
    start = perf_counter()
    embed_start = perf_counter()
    vector = embedder.embed([query_text])
    embed_ms = (perf_counter() - embed_start) * 1000
    if not vector:
        return [], RetrievalTiming(embed_ms=embed_ms, search_ms=0.0, total_ms=embed_ms)
    search_start = perf_counter()
    results = vector_store.search(vector[0], top_k)
    search_ms = (perf_counter() - search_start) * 1000
    total_ms = (perf_counter() - start) * 1000
    return results, RetrievalTiming(embed_ms=embed_ms, search_ms=search_ms, total_ms=total_ms)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(ordered[int(k)])
    d0 = ordered[f] * (c - k)
    d1 = ordered[c] * (k - f)
    return float(d0 + d1)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _make_answer_prompt(context: str, question: str) -> str:
    return (
        "Use the context to answer the question."
        "Think step by step. Cite the Source ID for every claim you make."
        " If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _judge_prompt_faithfulness(context: str, question: str, answer: str) -> str:
    return (
        "You are evaluating if an AI assistant's answer uses ONLY information from the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Generated Answer:\n{answer}\n\n"
        "Task: Rate faithfulness on scale 1-5:\n"
        "5 = Every claim is directly supported by context, no additions\n"
        "4 = Mostly faithful, minor inference needed\n"
        "3 = Some unsupported claims but generally accurate\n"
        "2 = Multiple unsupported or speculative claims\n"
        "1 = Significant hallucination or contradiction\n\n"
        "Think step-by-step:\n"
        "1. List each claim in the answer\n"
        "2. For each claim, find supporting evidence in context or mark as UNSUPPORTED\n"
        "3. Assign score\n\n"
        "Output format:\n"
        "REASONING: [your analysis]\n"
        "SCORE: [1-5]"
    )


def _judge_prompt_completeness(context: str, question: str, answer: str) -> str:
    return (
        "Does the generated answer fully address the question using available context?\n\n"
        "5 = Complete answer, all aspects covered\n"
        "3 = Partial answer, missing key details\n"
        "1 = Does not address the question\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Generated Answer:\n{answer}\n\n"
        "REASONING: [analysis]\n"
        "SCORE: [1-5]"
    )


def _judge_prompt_correctness(question: str, ground_truth: str, answer: str) -> str:
    return (
        "You are grading an answer against ground truth.\n\n"
        f"Question: {question}\n"
        f"Ground Truth: {ground_truth}\n"
        f"Generated: {answer}\n\n"
        "Rubric:\n"
        "5 = All key facts correct, well-structured\n"
        "4 = Minor factual error or missing secondary detail\n"
        "3 = Major fact wrong OR significant omission\n"
        "2 = Multiple errors, partially correct\n"
        "1 = Completely wrong or irrelevant\n\n"
        "Compare key facts:\n"
        "[List facts from ground truth]\n"
        "[Check each in generated answer]\n\n"
        "REASONING: [analysis]\n"
        "SCORE: [1-5]"
    )


def _parse_score(text: str) -> Optional[int]:
    match = re.search(r"SCORE:\s*([1-5])", text)
    if not match:
        return None
    return int(match.group(1))


def _extract_citations(answer: str) -> List[str]:
    return re.findall(r"\[Source:\s*([^\]]+)\]", answer)


def _answer_has_abstain(answer: str) -> bool:
    lowered = answer.lower()
    signals = [
        "not in the context",
        "not in context",
        "i don't know",
        "cannot answer",
        "not provided",
        "no information",
        "not mentioned",
        "unknown",
    ]
    return any(signal in lowered for signal in signals)


def _build_context_for_sources(
    chunker: RagCoreChunker,
    sources: Iterable[str],
    chunk_size: int,
    overlap: int,
    snippet: str,
    base_dir: Optional[Path],
) -> str:
    source_list = []
    for source in sources:
        candidate = Path(str(source))
        if not candidate.is_absolute() and base_dir is not None:
            candidate = base_dir / candidate
        source_list.append(str(candidate))
    chunks = chunker.chunk_files(source_list, chunk_size, overlap)
    filtered = []
    if snippet:
        for chunk in chunks:
            if snippet in chunk.text:
                filtered.append(chunk)
    if not filtered:
        filtered = chunks
    context_parts = []
    for chunk in filtered:
        context_parts.append(f"[Source: {chunk.source}]\n{chunk.text}")
    return "\n\n".join(context_parts)


def generate_dataset(args: argparse.Namespace) -> int:
    overrides = build_overrides(args)
    cfg = resolve_config(args.config, overrides)
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        return 2

    chunker = RagCoreChunker()
    file_paths = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if args.extensions:
                valid_exts = {ext if ext.startswith(".") else f".{ext}" for ext in _parse_extensions(args.extensions)}
                if Path(file).suffix.lower() not in valid_exts:
                    continue
            file_paths.append(os.path.join(root, file))

    if not file_paths:
        print("No files found for dataset generation.")
        return 2

    chunks = chunker.chunk_files(file_paths, cfg.ingestion.chunk_size, cfg.ingestion.overlap)
    if not chunks:
        print("Chunker returned no chunks.")
        return 2

    if args.num_questions > len(chunks):
        print(f"Requested {args.num_questions} questions, but only {len(chunks)} chunks available.")
        return 2

    rng = __import__("random")
    rng.seed(args.seed)
    selected = rng.sample(chunks, args.num_questions)

    entries = []
    for chunk in selected:
        snippet = chunk.text.strip().replace("\n", " ")
        snippet = snippet[: args.snippet_chars]
        entries.append(
            {
                "question": "TODO",
                "ground_truth_answer": "TODO",
                "ground_truth_sources": [chunk.source],
                "chunk_text_snippet": snippet,
                "question_type": "factual",
                "expected_chunk_count": 1,
            }
        )

    for _ in range(args.negative_count):
        entries.append(
            {
                "question": "TODO (negative)",
                "ground_truth_answer": "I don't know",
                "ground_truth_sources": [],
                "chunk_text_snippet": "",
                "question_type": "negative",
                "expected_chunk_count": 0,
            }
        )

    output = Path(args.output)
    save_json(output, entries)
    print(f"Wrote dataset skeleton to {output}")
    print("Manual verification required: replace TODO entries with real questions/answers.")
    return 0


def format_report(results: Dict) -> str:
    cfg = results["config"]
    retrieval = results["retrieval_metrics"]
    generation = results["generation_metrics"]
    end_to_end = results["end_to_end_metrics"]
    lines = []
    lines.append("=== RAG EVALUATION REPORT ===")
    lines.append(f"Run: {results['run_timestamp']}")
    lines.append(
        f"Config: chunk_size={cfg['chunk_size']}, overlap={cfg['overlap']}, "
        f"top_k={cfg['top_k']}, model={cfg['model']}"
    )
    lines.append("")
    lines.append("--- RETRIEVAL METRICS ---")
    lines.append(f"Hit Rate@{cfg['top_k']}:       {retrieval['hit_rate']:.2f}")
    lines.append(f"MRR:              {retrieval['mrr']:.2f}")
    lines.append(f"Precision@{cfg['top_k']}:      {retrieval['precision']:.2f}")
    lines.append(f"Avg Latency:      {retrieval['latency_ms']['avg']:.1f}ms")
    lines.append("")
    lines.append("--- GENERATION METRICS (Perfect Context) ---")
    lines.append(f"Faithfulness:     {generation['faithfulness']:.2f}/5.0")
    lines.append(f"Completeness:     {generation['completeness']:.2f}/5.0")
    lines.append(f"Citation Accuracy: {generation['citation_accuracy']:.2f}")
    lines.append("")
    lines.append("--- END-TO-END METRICS ---")
    lines.append(f"Semantic Similarity: {end_to_end['semantic_similarity']:.2f}")
    lines.append(f"Correctness Score:   {end_to_end['correctness']:.2f}/5.0")
    lines.append(f"\"I Don't Know\" Acc:  {end_to_end['abstain_accuracy']:.2f}")
    lines.append(f"Total Latency:       {end_to_end['latency_ms']['avg']:.1f}ms average")
    return "\n".join(lines)


def compare_results(baseline: Dict, latest: Dict) -> Dict[str, object]:
    base_retr = baseline.get("retrieval_metrics", {})
    new_retr = latest.get("retrieval_metrics", {})
    base_end = baseline.get("end_to_end_metrics", {})
    new_end = latest.get("end_to_end_metrics", {})
    report_lines = []
    report_lines.append("=== RAG COMPARISON REPORT ===")
    report_lines.append(f"Baseline: {baseline.get('run_timestamp', 'unknown')}")
    report_lines.append(f"Current:  {latest.get('run_timestamp', 'unknown')}")
    report_lines.append("")
    report_lines.append("Metric              Baseline    Current    Delta")

    def _line(name: str, base_val: float, new_val: float) -> None:
        delta = new_val - base_val
        report_lines.append(f"{name:<20} {base_val:>8.2f}    {new_val:>7.2f}    {delta:>+6.2f}")

    _line("Hit Rate", base_retr.get("hit_rate", 0.0), new_retr.get("hit_rate", 0.0))
    _line("MRR", base_retr.get("mrr", 0.0), new_retr.get("mrr", 0.0))
    _line("Semantic Sim", base_end.get("semantic_similarity", 0.0), new_end.get("semantic_similarity", 0.0))
    _line(
        "Avg Latency",
        base_end.get("latency_ms", {}).get("avg", 0.0),
        new_end.get("latency_ms", {}).get("avg", 0.0),
    )

    exit_code = 0
    hit_drop = base_retr.get("hit_rate", 0.0) - new_retr.get("hit_rate", 0.0)
    if hit_drop > 0.10:
        report_lines.append("CRITICAL: Hit Rate dropped more than 10%.")
        exit_code = 1
    mrr_drop = base_retr.get("mrr", 0.0) - new_retr.get("mrr", 0.0)
    if mrr_drop > 0.10:
        report_lines.append("WARNING: MRR dropped more than 0.1.")
        exit_code = max(exit_code, 1)
    base_latency = base_end.get("latency_ms", {}).get("avg", 0.0)
    new_latency = new_end.get("latency_ms", {}).get("avg", 0.0)
    if base_latency:
        latency_increase = (new_latency - base_latency) / base_latency
        if latency_increase > 0.50:
            report_lines.append("PERFORMANCE REGRESSION: Avg latency increased >50%.")
            exit_code = max(exit_code, 1)

    return {"report": "\n".join(report_lines), "exit_code": exit_code}


def set_baseline(args: argparse.Namespace) -> int:
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results not found: {results_path}")
        return 2
    data = _load_json(results_path)
    output_dir = Path(args.output_dir)
    baseline_path = output_dir / "baseline.json"
    save_json(baseline_path, data)
    print(f"Baseline saved to {baseline_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG evaluation system")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="Path to YAML config file")
    common.add_argument("--chunk-size", type=int, help="Tokens per chunk")
    common.add_argument("--overlap", type=int, help="Token overlap per chunk")
    common.add_argument("--extensions", help="Comma-separated extensions")
    common.add_argument("--files-per-batch", type=int, help="Initial batch size")
    common.add_argument("--adaptive-batching", type=lambda v: v.lower() == "true", help="Enable adaptive batching")
    common.add_argument("--min-files-per-batch", type=int, help="Minimum batch size")
    common.add_argument("--max-files-per-batch", type=int, help="Maximum batch size")
    common.add_argument("--target-batch-seconds", type=float, help="Target batch duration")
    common.add_argument("--top-k", type=int, help="Top K chunks")
    common.add_argument("--model", help="Ollama model name")
    common.add_argument("--db-path", help="LanceDB storage path")
    common.add_argument("--table-name", help="LanceDB table name")
    common.add_argument("--prefer-gpu", type=lambda v: v.lower() == "true", help="Prefer GPU embeddings")
    common.add_argument("--embed-retries", type=int, help="Embedding retry count")
    common.add_argument("--embed-gpu-batch", type=int, help="GPU batch size")
    common.add_argument("--embed-cpu-batch", type=int, help="CPU batch size")
    common.add_argument("--log-level", help="Logging level")

    gen = sub.add_parser("generate-dataset", parents=[common], help="Generate dataset skeleton")
    gen.add_argument("--source-dir", required=True, help="Directory to scan")
    gen.add_argument("--output", required=True, help="Output JSON file")
    gen.add_argument("--num-questions", type=int, default=50, help="Number of questions to generate")
    gen.add_argument("--negative-count", type=int, default=5, help="Number of negative placeholders")
    gen.add_argument("--snippet-chars", type=int, default=240, help="Snippet length")
    gen.add_argument("--seed", type=int, default=7, help="Random seed")

    eval_cmd = sub.add_parser("evaluate", parents=[common], help="Run evaluation")
    eval_cmd.add_argument("--dataset", required=True, help="Path to dataset JSON/JSONL")
    eval_cmd.add_argument("--index-dir", required=True, help="Directory to evaluate against")
    eval_cmd.add_argument("--output-dir", default="eval_results", help="Output directory")
    eval_cmd.add_argument("--compare-baseline", action="store_true", help="Compare against baseline")
    eval_cmd.add_argument("--set-baseline", action="store_true", help="Set baseline from this run")

    base = sub.add_parser("set-baseline", help="Set baseline from existing results")
    base.add_argument("--results", required=True, help="Path to results JSON")
    base.add_argument("--output-dir", default="eval_results", help="Output directory")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "generate-dataset":
        return generate_dataset(args)
    if args.command == "evaluate":
        return evaluate(args)
    if args.command == "set-baseline":
        return set_baseline(args)
    print("Unknown command")
    return 2


def evaluate(args: argparse.Namespace) -> int:
    overrides = build_overrides(args)
    cfg = resolve_config(args.config, overrides)
    dataset = load_dataset(Path(args.dataset))
    if not dataset:
        print("Dataset is empty.")
        return 2

    ok, message = _index_ready(cfg.vector_store.db_path, cfg.vector_store.table_name)
    if not ok:
        print("Index not ready:", message)
        print("Run ingestion first: python rag_cli.py ingest <DIRECTORY>")
        return 2

    embedder = FastEmbedEmbedder(
        prefer_gpu=cfg.embedding.prefer_gpu,
        max_retries=cfg.embedding.max_retries,
        gpu_batch_size=cfg.embedding.gpu_batch_size,
        cpu_batch_size=cfg.embedding.cpu_batch_size,
    )
    vector_store = LanceDBVectorStore(
        db_path=cfg.vector_store.db_path,
        table_name=cfg.vector_store.table_name,
    )
    index = Index(
        directory=Path(args.index_dir),
        chunk_size=cfg.ingestion.chunk_size,
        overlap=cfg.ingestion.overlap,
        extensions=cfg.ingestion.extensions,
        embedder=embedder,
        vector_store=vector_store,
        files_per_batch=cfg.ingestion.files_per_batch,
        adaptive_batching=cfg.ingestion.adaptive_batching,
        min_files_per_batch=cfg.ingestion.min_files_per_batch,
        max_files_per_batch=cfg.ingestion.max_files_per_batch,
        target_batch_seconds=cfg.ingestion.target_batch_seconds,
    )
    llm = OllamaLLM(model=cfg.llm.model, timeout_seconds=cfg.llm.timeout_seconds)
    rag = RAGSystem(index=index, llm=llm, top_k=cfg.query.top_k)

    retrieval_hits = 0
    retrieval_rrs = []
    retrieval_precisions = []
    retrieval_latencies = []
    retrieval_embed_latencies = []
    retrieval_search_latencies = []
    retrieval_failures = []
    multi_source_recalls = []

    per_question = []

    for idx, entry in enumerate(dataset, start=1):
        question = entry["question"]
        gt_sources = entry["ground_truth_sources"]
        expected_chunk_count = entry.get("expected_chunk_count", 1)
        question_type = entry.get("question_type", "factual")
        record = {
            "id": f"Q{idx}",
            "question": question,
            "question_type": question_type,
            "expected_chunk_count": expected_chunk_count,
        }
        try:
            results, timing = _timed_query(embedder, vector_store, question, cfg.query.top_k)
            retrieved_sources = [chunk.source for chunk, _score in results]
            record["retrieved_sources"] = retrieved_sources
            record["retrieval_timing_ms"] = {
                "embed": timing.embed_ms,
                "search": timing.search_ms,
                "total": timing.total_ms,
            }
            retrieval_latencies.append(timing.total_ms)
            retrieval_embed_latencies.append(timing.embed_ms)
            retrieval_search_latencies.append(timing.search_ms)

            hit = any(
                source_matches(gt, observed)
                for gt in gt_sources
                for observed in retrieved_sources
            ) if gt_sources else False
            retrieval_hits += int(hit)
            record["retrieval_hit"] = hit

            rr = 0.0
            for pos, observed in enumerate(retrieved_sources, start=1):
                if any(source_matches(gt, observed) for gt in gt_sources):
                    rr = 1.0 / pos
                    break
            retrieval_rrs.append(rr)
            record["retrieval_rr"] = rr

            relevant_count = 0
            for observed in retrieved_sources:
                if any(source_matches(gt, observed) for gt in gt_sources):
                    relevant_count += 1
            precision = relevant_count / cfg.query.top_k if cfg.query.top_k else 0.0
            retrieval_precisions.append(precision)
            record["retrieval_precision"] = precision

            if question_type == "multi-hop" or len(gt_sources) > 1:
                if gt_sources:
                    recall = len(
                        {gt for gt in gt_sources if any(source_matches(gt, obs) for obs in retrieved_sources)}
                    ) / len(gt_sources)
                    multi_source_recalls.append(recall)
                    record["multi_source_recall"] = recall
        except Exception as exc:
            retrieval_failures.append((idx, str(exc)))
            record["retrieval_error"] = str(exc)
        per_question.append(record)

    retrieval_metrics = {
        "hit_rate": retrieval_hits / len(dataset),
        "mrr": _mean(retrieval_rrs),
        "precision": _mean(retrieval_precisions),
        "context_recall": _mean(multi_source_recalls),
        "latency_ms": {
            "p50": _percentile(retrieval_latencies, 0.50),
            "p95": _percentile(retrieval_latencies, 0.95),
            "p99": _percentile(retrieval_latencies, 0.99),
            "avg": _mean(retrieval_latencies),
        },
        "embed_latency_ms": _mean(retrieval_embed_latencies),
        "search_latency_ms": _mean(retrieval_search_latencies),
        "failures": retrieval_failures,
    }

    chunker = RagCoreChunker()
    generation_scores = []
    completeness_scores = []
    citation_scores = []
    generation_latencies = []
    generation_failures = []

    for idx, entry in enumerate(dataset, start=1):
        if entry.get("question_type") == "negative":
            continue
        try:
            context = _build_context_for_sources(
                chunker,
                entry["ground_truth_sources"],
                cfg.ingestion.chunk_size,
                cfg.ingestion.overlap,
                entry.get("chunk_text_snippet", ""),
                Path(args.index_dir),
            )
            prompt = _make_answer_prompt(context, entry["question"])
            start = perf_counter()
            answer = llm.generate(prompt)
            duration = (perf_counter() - start) * 1000
            generation_latencies.append(duration)
            per_question[idx - 1]["generation_answer"] = answer
            try:
                faith_prompt = _judge_prompt_faithfulness(context, entry["question"], answer)
                comp_prompt = _judge_prompt_completeness(context, entry["question"], answer)
                faith_raw = llm.generate(faith_prompt)
                comp_raw = llm.generate(comp_prompt)
                faith_score = _parse_score(faith_raw)
                comp_score = _parse_score(comp_raw)
                if faith_score is not None:
                    generation_scores.append(faith_score)
                if comp_score is not None:
                    completeness_scores.append(comp_score)

                citations = _extract_citations(answer)
                if citations:
                    correct = 0
                    for citation in citations:
                        if any(source_matches(gt, citation) for gt in entry["ground_truth_sources"]):
                            correct += 1
                    citation_scores.append(correct / len(citations))
            except Exception as exc:
                generation_failures.append((idx, f"judge_error: {exc}"))
                per_question[idx - 1]["generation_judge_error"] = str(exc)
        except Exception as exc:
            generation_failures.append((idx, str(exc)))
            per_question[idx - 1]["generation_error"] = str(exc)

    generation_metrics = {
        "faithfulness": _mean(generation_scores),
        "completeness": _mean(completeness_scores),
        "citation_accuracy": _mean(citation_scores),
        "latency_ms": {
            "avg": _mean(generation_latencies),
            "p95": _percentile(generation_latencies, 0.95),
        },
        "failures": generation_failures,
    }

    similarity_scores = []
    correctness_scores = []
    abstain_hits = 0
    end_to_end_latencies = []
    end_to_end_failures = []

    for idx, entry in enumerate(dataset, start=1):
        try:
            start = perf_counter()
            answer = rag.answer(entry["question"])
            duration = (perf_counter() - start) * 1000
            end_to_end_latencies.append(duration)
            per_question[idx - 1]["end_to_end_answer"] = answer

            if entry.get("question_type") == "negative":
                if _answer_has_abstain(answer):
                    abstain_hits += 1
                continue

            gt_answer = entry["ground_truth_answer"]
            embeddings = embedder.embed([answer, gt_answer])
            if len(embeddings) == 2:
                similarity_scores.append(_cosine_similarity(embeddings[0], embeddings[1]))
            try:
                correctness_prompt = _judge_prompt_correctness(entry["question"], gt_answer, answer)
                correctness_raw = llm.generate(correctness_prompt)
                correctness_score = _parse_score(correctness_raw)
                if correctness_score is not None:
                    correctness_scores.append(correctness_score)
            except Exception as exc:
                end_to_end_failures.append((idx, f"judge_error: {exc}"))
                per_question[idx - 1]["end_to_end_judge_error"] = str(exc)
        except Exception as exc:
            end_to_end_failures.append((idx, str(exc)))
            per_question[idx - 1]["end_to_end_error"] = str(exc)

    negative_questions = [q for q in dataset if q.get("question_type") == "negative"]
    end_to_end_metrics = {
        "semantic_similarity": _mean(similarity_scores),
        "correctness": _mean(correctness_scores),
        "abstain_accuracy": (abstain_hits / len(negative_questions)) if negative_questions else 0.0,
        "latency_ms": {
            "avg": _mean(end_to_end_latencies),
            "p95": _percentile(end_to_end_latencies, 0.95),
        },
        "failures": end_to_end_failures,
    }

    results = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "chunk_size": cfg.ingestion.chunk_size,
            "overlap": cfg.ingestion.overlap,
            "top_k": cfg.query.top_k,
            "model": cfg.llm.model,
            "db_path": cfg.vector_store.db_path,
            "table_name": cfg.vector_store.table_name,
        },
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics,
        "end_to_end_metrics": end_to_end_metrics,
        "per_question": per_question,
    }

    output_dir = Path(args.output_dir)
    latest_path = output_dir / "latest.json"
    save_json(latest_path, results)

    if args.set_baseline:
        baseline_path = output_dir / "baseline.json"
        save_json(baseline_path, results)

    report = format_report(results)
    print(report)

    if args.compare_baseline:
        baseline_path = output_dir / "baseline.json"
        if not baseline_path.exists():
            print("Baseline not found. Run with --set-baseline first.")
            return 2
        baseline = _load_json(baseline_path)
        comparison = compare_results(baseline, results)
        report_path = output_dir / "comparison_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(comparison["report"], encoding="utf-8")
        if comparison["exit_code"] != 0:
            return comparison["exit_code"]
    return 0


if __name__ == "__main__":
    sys.exit(main())

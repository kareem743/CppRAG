import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Tuple

import typer

from rag.config import AppConfig, build_config
from rag.embedders import FastEmbedEmbedder
from rag.index import Index
from rag.llm import OllamaLLM
from rag.rag import RAGSystem
from rag.vector_store import LanceDBVectorStore
from rag.chunker import RagCoreChunker

app = typer.Typer(
    help="RAG evaluation system with dataset generation, evaluation, and baseline comparison.",
)

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


def _parse_extensions(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return parts or None


def _build_overrides(
    chunk_size: Optional[int],
    overlap: Optional[int],
    extensions: Optional[str],
    files_per_batch: Optional[int],
    adaptive_batching: Optional[bool],
    min_files_per_batch: Optional[int],
    max_files_per_batch: Optional[int],
    target_batch_seconds: Optional[float],
    top_k: Optional[int],
    model: Optional[str],
    db_path: Optional[str],
    table_name: Optional[str],
    prefer_gpu: Optional[bool],
    embed_retries: Optional[int],
    embed_gpu_batch: Optional[int],
    embed_cpu_batch: Optional[int],
    log_level: Optional[str],
) -> dict:
    overrides: dict = {}

    def _set(path: List[str], value) -> None:
        node = overrides
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value

    if chunk_size is not None:
        _set(["ingestion", "chunk_size"], chunk_size)
    if overlap is not None:
        _set(["ingestion", "overlap"], overlap)
    if extensions is not None:
        _set(["ingestion", "extensions"], _parse_extensions(extensions))
    if files_per_batch is not None:
        _set(["ingestion", "files_per_batch"], files_per_batch)
    if adaptive_batching is not None:
        _set(["ingestion", "adaptive_batching"], adaptive_batching)
    if min_files_per_batch is not None:
        _set(["ingestion", "min_files_per_batch"], min_files_per_batch)
    if max_files_per_batch is not None:
        _set(["ingestion", "max_files_per_batch"], max_files_per_batch)
    if target_batch_seconds is not None:
        _set(["ingestion", "target_batch_seconds"], target_batch_seconds)
    if top_k is not None:
        _set(["query", "top_k"], top_k)
    if model is not None:
        _set(["llm", "model"], model)
    if db_path is not None:
        _set(["vector_store", "db_path"], db_path)
    if table_name is not None:
        _set(["vector_store", "table_name"], table_name)
    if prefer_gpu is not None:
        _set(["embedding", "prefer_gpu"], prefer_gpu)
    if embed_retries is not None:
        _set(["embedding", "max_retries"], embed_retries)
    if embed_gpu_batch is not None:
        _set(["embedding", "gpu_batch_size"], embed_gpu_batch)
    if embed_cpu_batch is not None:
        _set(["embedding", "cpu_batch_size"], embed_cpu_batch)
    if log_level is not None:
        _set(["logging", "level"], log_level)
    return overrides


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


def generate_dataset(
    source_dir: Path,
    output: Path,
    num_questions: int,
    negative_count: int,
    snippet_chars: int,
    seed: int,
    extensions: Optional[str],
    config: Optional[str],
    overrides: dict,
) -> int:
    cfg = resolve_config(config, overrides)
    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        return 2

    chunker = RagCoreChunker()
    file_paths = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if extensions:
                valid_exts = {ext if ext.startswith(".") else f".{ext}" for ext in _parse_extensions(extensions)}
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

    if num_questions > len(chunks):
        print(f"Requested {num_questions} questions, but only {len(chunks)} chunks available.")
        return 2

    rng = __import__("random")
    rng.seed(seed)
    selected = rng.sample(chunks, num_questions)

    entries = []
    for chunk in selected:
        snippet = chunk.text.strip().replace("\n", " ")
        snippet = snippet[:snippet_chars]
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

    for _ in range(negative_count):
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


def set_baseline(results_path: Path, output_dir: Path) -> int:
    if not results_path.exists():
        print(f"Results not found: {results_path}")
        return 2
    data = _load_json(results_path)
    baseline_path = output_dir / "baseline.json"
    save_json(baseline_path, data)
    print(f"Baseline saved to {baseline_path}")
    return 0


def evaluate(
    dataset_path: Path,
    index_dir: Path,
    output_dir: Path,
    compare_baseline: bool,
    set_baseline_flag: bool,
    config: Optional[str],
    overrides: dict,
) -> int:
    cfg = resolve_config(config, overrides)
    dataset = load_dataset(dataset_path)
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
        directory=index_dir,
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
    total_questions = len(dataset)
    print(f"Evaluation started: {total_questions} questions")

    for idx, entry in enumerate(dataset, start=1):
        if idx == 1 or idx % 10 == 0 or idx == total_questions:
            print(f"[Retrieval] {idx}/{total_questions}")
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
    generation_total = sum(1 for entry in dataset if entry.get("question_type") != "negative")
    generation_done = 0

    for idx, entry in enumerate(dataset, start=1):
        if entry.get("question_type") == "negative":
            continue
        generation_done += 1
        if generation_done == 1 or generation_done % 10 == 0 or generation_done == generation_total:
            print(f"[Generation] {generation_done}/{generation_total}")
        try:
            context = _build_context_for_sources(
                chunker,
                entry["ground_truth_sources"],
                cfg.ingestion.chunk_size,
                cfg.ingestion.overlap,
                entry.get("chunk_text_snippet", ""),
                index_dir,
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
        if idx == 1 or idx % 10 == 0 or idx == total_questions:
            print(f"[End-to-End] {idx}/{total_questions}")
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

    latest_path = output_dir / "latest.json"
    save_json(latest_path, results)

    if set_baseline_flag:
        baseline_path = output_dir / "baseline.json"
        save_json(baseline_path, results)

    report = format_report(results)
    print(report)

    if compare_baseline:
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


@app.command("generate-dataset", help="Generate a dataset skeleton with TODO questions and answers.")
def generate_dataset_cmd(
    source_dir: Path = typer.Option(..., "--source-dir", help="Directory to scan"),
    output: Path = typer.Option(..., "--output", help="Output JSON file"),
    num_questions: int = typer.Option(50, "--num-questions", help="Number of questions to generate"),
    negative_count: int = typer.Option(5, "--negative-count", help="Number of negative placeholders"),
    snippet_chars: int = typer.Option(240, "--snippet-chars", help="Snippet length"),
    seed: int = typer.Option(7, "--seed", help="Random seed"),
    extensions: Optional[str] = typer.Option(None, "--extensions", help="Comma-separated extensions"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to YAML config file"),
    chunk_size: Optional[int] = typer.Option(None, "--chunk-size", help="Tokens per chunk"),
    overlap: Optional[int] = typer.Option(None, "--overlap", help="Token overlap per chunk"),
    files_per_batch: Optional[int] = typer.Option(None, "--files-per-batch", help="Initial batch size"),
    adaptive_batching: Optional[bool] = typer.Option(
        None,
        "--adaptive-batching/--no-adaptive-batching",
        help="Enable adaptive batch sizing",
    ),
    min_files_per_batch: Optional[int] = typer.Option(None, "--min-files-per-batch", help="Minimum batch size"),
    max_files_per_batch: Optional[int] = typer.Option(None, "--max-files-per-batch", help="Maximum batch size"),
    target_batch_seconds: Optional[float] = typer.Option(None, "--target-batch-seconds", help="Target batch duration"),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Top K chunks"),
    model: Optional[str] = typer.Option(None, "--model", help="Ollama model name"),
    db_path: Optional[str] = typer.Option(None, "--db-path", help="LanceDB storage path"),
    table_name: Optional[str] = typer.Option(None, "--table-name", help="LanceDB table name"),
    prefer_gpu: Optional[bool] = typer.Option(
        None,
        "--prefer-gpu/--no-prefer-gpu",
        help="Prefer GPU embeddings",
    ),
    embed_retries: Optional[int] = typer.Option(None, "--embed-retries", help="Embedding retry count"),
    embed_gpu_batch: Optional[int] = typer.Option(None, "--embed-gpu-batch", help="GPU batch size"),
    embed_cpu_batch: Optional[int] = typer.Option(None, "--embed-cpu-batch", help="CPU batch size"),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="Logging level"),
):
    overrides = _build_overrides(
        chunk_size,
        overlap,
        extensions,
        files_per_batch,
        adaptive_batching,
        min_files_per_batch,
        max_files_per_batch,
        target_batch_seconds,
        top_k,
        model,
        db_path,
        table_name,
        prefer_gpu,
        embed_retries,
        embed_gpu_batch,
        embed_cpu_batch,
        log_level,
    )
    code = generate_dataset(
        source_dir=source_dir,
        output=output,
        num_questions=num_questions,
        negative_count=negative_count,
        snippet_chars=snippet_chars,
        seed=seed,
        extensions=extensions,
        config=config,
        overrides=overrides,
    )
    if code != 0:
        raise typer.Exit(code)


@app.command("evaluate", help="Run evaluation and optionally compare to a baseline.")
def evaluate_cmd(
    dataset: Path = typer.Option(..., "--dataset", help="Path to dataset JSON/JSONL"),
    index_dir: Path = typer.Option(..., "--index-dir", help="Directory to evaluate against"),
    output_dir: Path = typer.Option("eval_results", "--output-dir", help="Output directory"),
    compare_baseline: bool = typer.Option(False, "--compare-baseline", help="Compare against baseline"),
    set_baseline_flag: bool = typer.Option(False, "--set-baseline", help="Set baseline from this run"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to YAML config file"),
    chunk_size: Optional[int] = typer.Option(None, "--chunk-size", help="Tokens per chunk"),
    overlap: Optional[int] = typer.Option(None, "--overlap", help="Token overlap per chunk"),
    extensions: Optional[str] = typer.Option(None, "--extensions", help="Comma-separated extensions"),
    files_per_batch: Optional[int] = typer.Option(None, "--files-per-batch", help="Initial batch size"),
    adaptive_batching: Optional[bool] = typer.Option(
        None,
        "--adaptive-batching/--no-adaptive-batching",
        help="Enable adaptive batch sizing",
    ),
    min_files_per_batch: Optional[int] = typer.Option(None, "--min-files-per-batch", help="Minimum batch size"),
    max_files_per_batch: Optional[int] = typer.Option(None, "--max-files-per-batch", help="Maximum batch size"),
    target_batch_seconds: Optional[float] = typer.Option(None, "--target-batch-seconds", help="Target batch duration"),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Top K chunks"),
    model: Optional[str] = typer.Option(None, "--model", help="Ollama model name"),
    db_path: Optional[str] = typer.Option(None, "--db-path", help="LanceDB storage path"),
    table_name: Optional[str] = typer.Option(None, "--table-name", help="LanceDB table name"),
    prefer_gpu: Optional[bool] = typer.Option(
        None,
        "--prefer-gpu/--no-prefer-gpu",
        help="Prefer GPU embeddings",
    ),
    embed_retries: Optional[int] = typer.Option(None, "--embed-retries", help="Embedding retry count"),
    embed_gpu_batch: Optional[int] = typer.Option(None, "--embed-gpu-batch", help="GPU batch size"),
    embed_cpu_batch: Optional[int] = typer.Option(None, "--embed-cpu-batch", help="CPU batch size"),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="Logging level"),
):
    overrides = _build_overrides(
        chunk_size,
        overlap,
        extensions,
        files_per_batch,
        adaptive_batching,
        min_files_per_batch,
        max_files_per_batch,
        target_batch_seconds,
        top_k,
        model,
        db_path,
        table_name,
        prefer_gpu,
        embed_retries,
        embed_gpu_batch,
        embed_cpu_batch,
        log_level,
    )
    code = evaluate(
        dataset_path=dataset,
        index_dir=index_dir,
        output_dir=output_dir,
        compare_baseline=compare_baseline,
        set_baseline_flag=set_baseline_flag,
        config=config,
        overrides=overrides,
    )
    if code != 0:
        raise typer.Exit(code)


@app.command("set-baseline", help="Set baseline from an existing results file.")
def set_baseline_cmd(
    results: Path = typer.Option(..., "--results", help="Path to results JSON"),
    output_dir: Path = typer.Option("eval_results", "--output-dir", help="Output directory"),
):
    code = set_baseline(results_path=results, output_dir=output_dir)
    if code != 0:
        raise typer.Exit(code)


if __name__ == "__main__":
    app()

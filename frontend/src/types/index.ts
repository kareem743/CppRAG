export interface ChunkSource {
  text: string;
  source: string;
  score?: number;
}

export interface ChatMetrics {
  latency_ms: number;
  embed_ms: number;
  retrieve_ms: number;
  generate_ms: number;
  engine: "cpp-accelerated";
}

export interface ChatResponse {
  answer: string;
  sources: ChunkSource[];
  metrics: ChatMetrics;
}

export interface SystemStats {
  ingestion_rate: number;
  gpu_active: boolean;
  vector_count: number;
  embedding_model: string;
  llm_model: string;
  embedding_device: "gpu" | "cpu" | "unknown";
  storage_db_bytes: number;
  storage_cache_bytes: number;
}

export type IngestStatus = {
  status: "idle" | "running" | "complete" | "error";
  message?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  directory?: string | null;
  total_files?: number | null;
  changed_files?: number | null;
  last_duration_ms?: number | null;
  dry_run?: boolean | null;
};

export interface IngestRequest {
  directory?: string;
  chunk_size?: number;
  overlap?: number;
  extensions?: string[];
  files_per_batch?: number;
  adaptive_batching?: boolean;
  min_files_per_batch?: number;
  max_files_per_batch?: number;
  target_batch_seconds?: number;
  db_path?: string;
  table_name?: string;
  prefer_gpu?: boolean;
  max_retries?: number;
  gpu_batch_size?: number;
  cpu_batch_size?: number;
  log_level?: string;
  dry_run?: boolean;
}

export interface ChunkSpan {
  start: number;
  end: number;
}

export interface VisualizeResponse {
  text: string;
  chunks: ChunkSpan[];
}

export interface Message {
  role: "user" | "ai";
  content: string;
  metrics?: Record<string, unknown>;
}

export interface EvalLatency {
  avg?: number;
  p50?: number;
  p95?: number;
  p99?: number;
}

export interface EvaluationPerQuestion {
  id: string;
  question: string;
  question_type?: string;
  retrieval_hit?: boolean;
  retrieval_precision?: number;
  retrieval_rr?: number;
  end_to_end_answer?: string;
}

export interface EvaluationResponse {
  run_timestamp: string;
  config: Record<string, unknown>;
  retrieval_metrics: {
    hit_rate?: number;
    mrr?: number;
    precision?: number;
    context_recall?: number;
    latency_ms?: EvalLatency;
  };
  generation_metrics: {
    faithfulness?: number;
    completeness?: number;
    citation_accuracy?: number;
    latency_ms?: EvalLatency;
  };
  end_to_end_metrics: {
    semantic_similarity?: number;
    correctness?: number;
    abstain_accuracy?: number;
    latency_ms?: EvalLatency;
  };
  per_question: EvaluationPerQuestion[];
}

export interface EvaluationRunRequest {
  dataset?: string;
  index_dir?: string;
  output_dir?: string;
  compare_baseline?: boolean;
  set_baseline?: boolean;
  config?: string;
  chunk_size?: number;
  overlap?: number;
  extensions?: string;
  files_per_batch?: number;
  adaptive_batching?: boolean;
  min_files_per_batch?: number;
  max_files_per_batch?: number;
  target_batch_seconds?: number;
  top_k?: number;
  model?: string;
  db_path?: string;
  table_name?: string;
  prefer_gpu?: boolean;
  embed_retries?: number;
  embed_gpu_batch?: number;
  embed_cpu_batch?: number;
  log_level?: string;
}

export interface EvaluationRunStatus {
  status: "idle" | "running" | "complete" | "error";
  message?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  last_duration_ms?: number | null;
  exit_code?: number | null;
  latest_results_path?: string | null;
  options?: Record<string, unknown> | null;
}

export interface DatasetGenerateRequest {
  source_dir?: string;
  output?: string;
  num_questions?: number;
  negative_count?: number;
  snippet_chars?: number;
  seed?: number;
  extensions?: string;
  config?: string;
  chunk_size?: number;
  overlap?: number;
  files_per_batch?: number;
  adaptive_batching?: boolean;
  min_files_per_batch?: number;
  max_files_per_batch?: number;
  target_batch_seconds?: number;
  top_k?: number;
  model?: string;
  db_path?: string;
  table_name?: string;
  prefer_gpu?: boolean;
  embed_retries?: number;
  embed_gpu_batch?: number;
  embed_cpu_batch?: number;
  log_level?: string;
}

export interface DatasetGenerateStatus {
  status: "idle" | "running" | "complete" | "error";
  message?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  last_duration_ms?: number | null;
  exit_code?: number | null;
  output_path?: string | null;
  options?: Record<string, unknown> | null;
}

export interface DatasetTodoItem {
  index: number;
  question: string;
  ground_truth_answer: string;
  question_type?: string | null;
  expected_chunk_count?: number | null;
}

export interface DatasetTodoListResponse {
  path: string;
  total_entries: number;
  todo_count: number;
  items: DatasetTodoItem[];
}

export interface DatasetTodoUpdateItem {
  index: number;
  question?: string;
  ground_truth_answer?: string;
  question_type?: string;
  expected_chunk_count?: number;
}

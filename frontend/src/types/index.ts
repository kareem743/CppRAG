export interface ChunkSource {
  text: string;
  source: string;
}

export interface ChatMetrics {
  latency_ms: number;
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

import os
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator


class EmbeddingConfig(BaseModel):
    prefer_gpu: bool = True
    max_retries: int = Field(1, ge=0)
    gpu_batch_size: int = Field(32, ge=1)
    cpu_batch_size: int = Field(256, ge=1)


class VectorStoreConfig(BaseModel):
    db_path: str = "./lancedb_data"
    table_name: str = "vectors"


class IngestionConfig(BaseModel):
    chunk_size: int = Field(200, ge=1)
    overlap: int = Field(50, ge=0)
    extensions: Optional[List[str]] = None
    files_per_batch: int = Field(1500, ge=1)
    adaptive_batching: bool = True
    min_files_per_batch: int = Field(1, ge=1)
    max_files_per_batch: int = Field(8000, ge=1)
    target_batch_seconds: float = Field(20.0, gt=0)

    @field_validator("overlap")
    @classmethod
    def overlap_lt_chunk(cls, value: int, info) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        return value


class QueryConfig(BaseModel):
    top_k: int = Field(5, ge=1)


class LLMConfig(BaseModel):
    model: str = "llama3"
    timeout_seconds: int = Field(60, ge=1)


class LoggingConfig(BaseModel):
    level: str = "INFO"


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)


class AppConfig(BaseModel):
    embedding: EmbeddingConfig = EmbeddingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    ingestion: IngestionConfig = IngestionConfig()
    query: QueryConfig = QueryConfig()
    llm: LLMConfig = LLMConfig()
    logging: LoggingConfig = LoggingConfig()
    server: ServerConfig = ServerConfig()


def load_yaml_config(path: Optional[str]) -> Dict:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def load_env_config() -> Dict:
    def _split_list(value: Optional[str]) -> Optional[List[str]]:
        if not value:
            return None
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return parts or None

    env = os.environ
    config: Dict = {}

    def _set(path: List[str], value) -> None:
        node = config
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value

    if "RAG_LOG_LEVEL" in env:
        _set(["logging", "level"], env["RAG_LOG_LEVEL"])
    if "RAG_DB_PATH" in env:
        _set(["vector_store", "db_path"], env["RAG_DB_PATH"])
    if "RAG_TABLE_NAME" in env:
        _set(["vector_store", "table_name"], env["RAG_TABLE_NAME"])
    if "RAG_CHUNK_SIZE" in env:
        _set(["ingestion", "chunk_size"], int(env["RAG_CHUNK_SIZE"]))
    if "RAG_OVERLAP" in env:
        _set(["ingestion", "overlap"], int(env["RAG_OVERLAP"]))
    if "RAG_EXTENSIONS" in env:
        _set(["ingestion", "extensions"], _split_list(env["RAG_EXTENSIONS"]))
    if "RAG_FILES_PER_BATCH" in env:
        _set(["ingestion", "files_per_batch"], int(env["RAG_FILES_PER_BATCH"]))
    if "RAG_ADAPTIVE_BATCHING" in env:
        _set(["ingestion", "adaptive_batching"], env["RAG_ADAPTIVE_BATCHING"].lower() == "true")
    if "RAG_MIN_FILES_PER_BATCH" in env:
        _set(["ingestion", "min_files_per_batch"], int(env["RAG_MIN_FILES_PER_BATCH"]))
    if "RAG_MAX_FILES_PER_BATCH" in env:
        _set(["ingestion", "max_files_per_batch"], int(env["RAG_MAX_FILES_PER_BATCH"]))
    if "RAG_TARGET_BATCH_SECONDS" in env:
        _set(["ingestion", "target_batch_seconds"], float(env["RAG_TARGET_BATCH_SECONDS"]))
    if "RAG_TOP_K" in env:
        _set(["query", "top_k"], int(env["RAG_TOP_K"]))
    if "RAG_MODEL" in env:
        _set(["llm", "model"], env["RAG_MODEL"])
    if "RAG_TIMEOUT_SECONDS" in env:
        _set(["llm", "timeout_seconds"], int(env["RAG_TIMEOUT_SECONDS"]))
    if "RAG_PREFER_GPU" in env:
        _set(["embedding", "prefer_gpu"], env["RAG_PREFER_GPU"].lower() == "true")
    if "RAG_EMBED_MAX_RETRIES" in env:
        _set(["embedding", "max_retries"], int(env["RAG_EMBED_MAX_RETRIES"]))
    if "RAG_EMBED_GPU_BATCH_SIZE" in env:
        _set(["embedding", "gpu_batch_size"], int(env["RAG_EMBED_GPU_BATCH_SIZE"]))
    if "RAG_EMBED_CPU_BATCH_SIZE" in env:
        _set(["embedding", "cpu_batch_size"], int(env["RAG_EMBED_CPU_BATCH_SIZE"]))
    if "RAG_SERVER_HOST" in env:
        _set(["server", "host"], env["RAG_SERVER_HOST"])
    if "RAG_SERVER_PORT" in env:
        _set(["server", "port"], int(env["RAG_SERVER_PORT"]))

    return config


def deep_merge(base: Dict, updates: Dict) -> Dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def build_config(
    yaml_path: Optional[str],
    cli_overrides: Optional[Dict] = None,
) -> AppConfig:
    data = {}
    data = deep_merge(data, load_yaml_config(yaml_path))
    data = deep_merge(data, load_env_config())
    if cli_overrides:
        data = deep_merge(data, cli_overrides)
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc

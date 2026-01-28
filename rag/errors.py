class RAGError(Exception):
    """Base class for RAG-related errors."""


class ChunkingError(RAGError):
    """Raised when chunking fails."""


class EmbeddingError(RAGError):
    """Raised when embedding fails."""


class VectorStoreError(RAGError):
    """Raised when vector store operations fail."""


class IngestionError(RAGError):
    """Raised when ingestion fails."""


class QueryError(RAGError):
    """Raised when query operations fail."""

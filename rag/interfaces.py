from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple

from .models import Chunk


class Chunker(ABC):
    @abstractmethod
    def chunk_files(self, file_paths: List[str], chunk_size: int, overlap: int) -> List[Chunk]:
        raise NotImplementedError


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        raise NotImplementedError


class VectorStore(ABC):
    @abstractmethod
    def ensure_schema(self, sample_vector: List[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def add(self, vectors: List[List[float]], texts: List[str], sources: List[str]) -> int:
        raise NotImplementedError

    @abstractmethod
    def search(self, vector: List[float], top_k: int) -> List[Tuple[Chunk, float]]:
        raise NotImplementedError

    @abstractmethod
    def compact(self) -> None:
        raise NotImplementedError

import gc
import logging
from typing import Iterable, List, Optional

from fastembed import TextEmbedding

from .errors import EmbeddingError
from .interfaces import Embedder

_LOGGER = logging.getLogger(__name__)


class FastEmbedEmbedder(Embedder):
    def __init__(
        self,
        prefer_gpu: bool = True,
        max_retries: int = 1,
        gpu_batch_size: int = 32,
        cpu_batch_size: int = 256,
    ):
        self._prefer_gpu = prefer_gpu
        self._max_retries = max_retries
        self._gpu_batch_size = gpu_batch_size
        self._cpu_batch_size = cpu_batch_size

    def config(self) -> dict:
        return {
            "prefer_gpu": self._prefer_gpu,
            "max_retries": self._max_retries,
            "gpu_batch_size": self._gpu_batch_size,
            "cpu_batch_size": self._cpu_batch_size,
        }

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        payload = list(texts)
        if not payload:
            return []

        errors: List[Exception] = []
        if self._prefer_gpu:
            for attempt in range(self._max_retries + 1):
                try:
                    return self._embed_with_providers(
                        payload,
                        providers=["CUDAExecutionProvider"],
                        batch_size=self._gpu_batch_size,
                    )
                except Exception as exc:
                    errors.append(exc)
                    _LOGGER.warning(
                        "GPU embedding failed (attempt %s/%s): %s",
                        attempt + 1,
                        self._max_retries + 1,
                        exc,
                    )

        for attempt in range(self._max_retries + 1):
            try:
                return self._embed_with_providers(
                    payload,
                    providers=None,
                    batch_size=self._cpu_batch_size,
                )
            except Exception as exc:
                errors.append(exc)
                _LOGGER.warning(
                    "CPU embedding failed (attempt %s/%s): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )

        message = "Embedding failed after GPU/CPU attempts."
        if errors:
            raise EmbeddingError(message) from errors[-1]
        raise EmbeddingError(message)

    def _embed_with_providers(
        self,
        texts: List[str],
        providers: Optional[List[str]],
        batch_size: int,
    ) -> List[List[float]]:
        model = TextEmbedding(providers=providers) if providers else TextEmbedding()
        try:
            return [list(vec) for vec in model.embed(texts, batch_size=batch_size)]
        finally:
            del model
            gc.collect()

import gc
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from .errors import EmbeddingError
from .interfaces import Embedder

_LOGGER = logging.getLogger(__name__)


class FastEmbedEmbedder(Embedder):
    def __init__(
        self,
        model_name: Optional[str] = None,
        prefer_gpu: bool = True,
        max_retries: int = 1,
        gpu_batch_size: int = 32,
        cpu_batch_size: int = 256,
    ):
        self._model_name = model_name
        self._prefer_gpu = prefer_gpu
        self._max_retries = max_retries
        self._gpu_batch_size = gpu_batch_size
        self._cpu_batch_size = cpu_batch_size
        self._last_provider: Optional[str] = None

    def config(self) -> dict:
        return {
            "model_name": self._model_name,
            "prefer_gpu": self._prefer_gpu,
            "max_retries": self._max_retries,
            "gpu_batch_size": self._gpu_batch_size,
            "cpu_batch_size": self._cpu_batch_size,
        }

    @property
    def last_provider(self) -> Optional[str]:
        return self._last_provider

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
                    # Retry loops cannot recover from a missing provider.
                    if self._is_cuda_provider_unavailable(exc):
                        break

        cache_recovered = False
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
                if not cache_recovered and self._is_missing_model_file_error(exc):
                    recovered = self._recover_corrupt_fastembed_cache(exc)
                    if recovered:
                        cache_recovered = True
                        try:
                            return self._embed_with_providers(
                                payload,
                                providers=None,
                                batch_size=self._cpu_batch_size,
                            )
                        except Exception as retry_exc:
                            errors.append(retry_exc)
                            _LOGGER.warning(
                                "CPU embedding failed after cache recovery: %s",
                                retry_exc,
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
        try:
            from fastembed import TextEmbedding
        except Exception as exc:
            raise EmbeddingError(
                "FastEmbed could not be imported. This is usually an onnxruntime "
                "binary mismatch. If you are using Python "
                f"{sys.version_info.major}.{sys.version_info.minor}, try a supported "
                "Python version for your installed onnxruntime/fastembed packages "
                "(for example Python 3.12), then reinstall dependencies."
            ) from exc

        if providers:
            model = (
                TextEmbedding(model_name=self._model_name, providers=providers)
                if self._model_name
                else TextEmbedding(providers=providers)
            )
        else:
            model = (
                TextEmbedding(model_name=self._model_name)
                if self._model_name
                else TextEmbedding()
            )
        try:
            vectors = [list(vec) for vec in model.embed(texts, batch_size=batch_size)]
            self._last_provider = "gpu" if providers else "cpu"
            return vectors
        finally:
            del model
            gc.collect()

    @staticmethod
    def _is_cuda_provider_unavailable(exc: Exception) -> bool:
        text = str(exc)
        return "CUDAExecutionProvider" in text and "not available" in text

    @staticmethod
    def _is_missing_model_file_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "no_suchfile" in text and "load model from" in text and ".onnx" in text

    @classmethod
    def _recover_corrupt_fastembed_cache(cls, exc: Exception) -> bool:
        model_root = cls._extract_fastembed_model_root(exc)
        if model_root is None:
            return False
        try:
            if model_root.exists():
                shutil.rmtree(model_root)
            _LOGGER.warning("Cleared corrupted FastEmbed cache directory: %s", model_root)
            return True
        except Exception as cleanup_exc:
            _LOGGER.warning(
                "Failed to clear corrupted FastEmbed cache directory '%s': %s",
                model_root,
                cleanup_exc,
            )
            return False

    @staticmethod
    def _extract_fastembed_model_root(exc: Exception) -> Optional[Path]:
        match = re.search(r"Load model from (.+?\.onnx)", str(exc), flags=re.IGNORECASE)
        if not match:
            return None
        model_path = Path(match.group(1))
        for parent in model_path.parents:
            if parent.name.startswith("models--"):
                has_fastembed_cache = any(
                    part.lower() == "fastembed_cache" for part in parent.parts
                )
                if has_fastembed_cache:
                    return parent
                return None
        return None

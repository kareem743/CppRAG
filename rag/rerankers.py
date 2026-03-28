import logging
from typing import List, Optional, Tuple

from .interfaces import Reranker
from .models import Chunk

_LOGGER = logging.getLogger(__name__)


class FlashRankReranker(Reranker):
    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        cache_dir: Optional[str] = None,
    ):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from flashrank import Ranker
        except Exception as exc:
            raise RuntimeError(
                "FlashRank is not installed. Install it with `pip install flashrank`."
            ) from exc

        _LOGGER.info("Loading reranker model '%s'", self._model_name)
        kwargs = {"model_name": self._model_name}
        if self._cache_dir:
            kwargs["cache_dir"] = self._cache_dir
        self._client = Ranker(**kwargs)
        return self._client

    def rerank(self, query: str, chunks: List[Chunk], top_k: int) -> List[Tuple[Chunk, float]]:
        if top_k <= 0 or not chunks:
            return []

        passages = [
            {"id": idx, "text": chunk.text, "meta": {"source": chunk.source}}
            for idx, chunk in enumerate(chunks)
        ]

        try:
            from flashrank import RerankRequest

            client = self._get_client()
            results = client.rerank(RerankRequest(query=query, passages=passages))
        except Exception as exc:
            _LOGGER.error("Reranking failed, using vector order fallback: %s", exc)
            return [(chunk, 0.0) for chunk in chunks[:top_k]]

        ranked_output: List[Tuple[Chunk, float]] = []
        for row in results:
            try:
                original_idx = int(row["id"])
                score = float(row.get("score", 0.0))
            except Exception:
                continue
            if 0 <= original_idx < len(chunks):
                ranked_output.append((chunks[original_idx], score))

        if not ranked_output:
            return [(chunk, 0.0) for chunk in chunks[:top_k]]
        return ranked_output[:top_k]

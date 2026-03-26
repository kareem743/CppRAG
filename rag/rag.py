import logging
from time import perf_counter
from typing import Dict, List, Optional, Tuple

from .index import Index
from .models import Chunk
from .llm import LocalLLM
from .interfaces import Reranker


_LOGGER = logging.getLogger(__name__)


class RAGSystem:
    def __init__(
        self,
        index: Index,
        llm: LocalLLM,
        top_k: int = 3,
        reranker: Optional[Reranker] = None,
        rerank_expansion: int = 10,
        rerank_candidate_cap: int = 50,
    ):
        self._index = index
        self._llm = llm
        self._top_k = top_k
        self._reranker = reranker
        self._rerank_expansion = max(1, rerank_expansion)
        self._rerank_candidate_cap = max(1, rerank_candidate_cap)

    def _build_prompt(self, ranked: List[Tuple[Chunk, float]], question: str) -> str:
        context = "\n\n".join(
            f"[Source: {chunk.source}]\n{chunk.text}" for chunk, _score in ranked
        )
        _LOGGER.debug("Context length: %s characters", len(context))
        prompt = (
            "You are a strict smart assistant that answers questions ONLY using the provided Context below.\n"
            "Do not hallucinate.\n"
            "Rules:\n"
            "1. Use the Context to answer the Question.\n"
            "2. Cite the Source ID (e.g. [Source: path/to/file]) for every claim.\n"
            "3. IF THE ANSWER IS NOT IN THE CONTEXT,  Do not guess.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        _LOGGER.debug("Prompt length: %s characters", len(prompt))
        return prompt

    def _index_query_with_metrics(
        self, question: str, top_k: int
    ) -> Tuple[List[Tuple[Chunk, float]], Dict[str, float]]:
        if hasattr(self._index, "query_with_metrics"):
            ranked, metrics = self._index.query_with_metrics(question, top_k=top_k)
            out = {
                "embed_ms": float(metrics.get("embed_ms", 0.0)),
                "retrieve_ms": float(metrics.get("retrieve_ms", 0.0)),
            }
            return ranked, out

        ranked = self._index.query(question, top_k=top_k)
        return ranked, {"embed_ms": 0.0, "retrieve_ms": 0.0}

    def retrieve_with_metrics(
        self, question: str, top_k: Optional[int] = None
    ) -> Tuple[List[Tuple[Chunk, float]], Dict[str, float]]:
        final_top_k = self._top_k if top_k is None else top_k
        if final_top_k <= 0:
            return [], {"embed_ms": 0.0, "retrieve_ms": 0.0, "rerank_ms": 0.0}

        retrieve_k = final_top_k
        if self._reranker is not None:
            retrieve_k = min(
                max(final_top_k, final_top_k * self._rerank_expansion),
                self._rerank_candidate_cap,
            )

        ranked, metrics = self._index_query_with_metrics(question, top_k=retrieve_k)
        metrics = dict(metrics)
        rerank_ms = 0.0

        if self._reranker is not None and ranked:
            rerank_start = perf_counter()
            chunks = [chunk for chunk, _score in ranked]
            try:
                ranked = self._reranker.rerank(question, chunks, final_top_k)
            except Exception as exc:
                _LOGGER.exception("Reranker failed, falling back to vector order: %s", exc)
                ranked = ranked[:final_top_k]
            rerank_ms = (perf_counter() - rerank_start) * 1000.0
            _LOGGER.info(
                "Reranked %s candidates to %s in %.2fms",
                len(chunks),
                len(ranked),
                rerank_ms,
            )
        else:
            ranked = ranked[:final_top_k]

        metrics["rerank_ms"] = rerank_ms
        return ranked, metrics

    def answer_with_metrics(
        self, question: str, top_k: Optional[int] = None
    ) -> Tuple[str, List[Tuple[Chunk, float]], Dict[str, float]]:
        _LOGGER.info("Answering question: %s", question)
        ranked, metrics = self.retrieve_with_metrics(question, top_k=top_k)
        prompt = self._build_prompt(ranked, question)
        generate_start = perf_counter()
        answer = self._llm.generate(prompt)
        metrics = dict(metrics)
        metrics["generate_ms"] = (perf_counter() - generate_start) * 1000.0
        return answer, ranked, metrics

    def answer(self, question: str) -> str:
        answer, _ranked, _metrics = self.answer_with_metrics(question)
        return answer

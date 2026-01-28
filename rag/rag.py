import logging

from .index import Index
from .models import Chunk
from .llm import LocalLLM


_LOGGER = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self, index: Index, llm: LocalLLM, top_k: int = 3):
        self._index = index
        self._llm = llm
        self._top_k = top_k

    def answer(self, question: str) -> str:
        _LOGGER.info("Answering question: %s", question)
        ranked = self._index.query(question, top_k=self._top_k)
        context = "\n\n".join(
            f"[Source: {chunk.source}]\n{chunk.text}" for chunk, _score in ranked
        )
        _LOGGER.debug("Context length: %s characters", len(context))
        prompt = (
            "Use the context to answer the question."
            "Think step by step. Cite the Source ID for every claim you make."
            " If the answer is not in the context, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        _LOGGER.debug("Prompt length: %s characters", len(prompt))
        return self._llm.generate(prompt)

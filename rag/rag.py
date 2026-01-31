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
        return self._llm.generate(prompt)

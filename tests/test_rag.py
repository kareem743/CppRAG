import tempfile
import unittest
from pathlib import Path

import pytest

from rag.embedders import FastEmbedEmbedder
from rag.index import Index
from rag.interfaces import Reranker
from rag.llm import LocalLLM
from rag.models import Chunk
from rag.rag import RAGSystem
from rag.vector_store import LanceDBVectorStore


class FakeLLM(LocalLLM):
    def __init__(self):
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "fake-answer"


class FakeIndex:
    def __init__(self, results):
        self._results = results
        self.last_query = None
        self.last_top_k = None

    def query(self, question: str, top_k: int = 3):
        self.last_query = question
        self.last_top_k = top_k
        return self._results

    def query_with_metrics(self, question: str, top_k: int = 3):
        self.last_query = question
        self.last_top_k = top_k
        return self._results[:top_k], {"embed_ms": 1.0, "retrieve_ms": 2.0}


class FakeReranker(Reranker):
    def __init__(self):
        self.last_query = None
        self.last_chunks = None
        self.last_top_k = None
        self.raise_error = False

    def rerank(self, query: str, chunks, top_k: int):
        self.last_query = query
        self.last_chunks = list(chunks)
        self.last_top_k = top_k
        if self.raise_error:
            raise RuntimeError("boom")
        ranked = list(reversed(chunks))
        return [(chunk, float(idx + 1)) for idx, chunk in enumerate(ranked[:top_k])]


class RagSystemTests(unittest.TestCase):
    def test_rag_includes_context_in_prompt(self):
        pytest.importorskip("rag_core")
        pytest.importorskip("lancedb")
        pytest.importorskip("fastembed")

        root = Path(__file__).resolve().parent / "fixtures" / "rag_prompt"
        with tempfile.TemporaryDirectory() as tmpdir:
            vector_store = LanceDBVectorStore(db_path=tmpdir, table_name="vectors")
            embedder = FastEmbedEmbedder(prefer_gpu=False, max_retries=0, cpu_batch_size=16, gpu_batch_size=16)
            try:
                index = Index.from_directory(
                    root,
                    chunk_size=4,
                    overlap=0,
                    embedder=embedder,
                    vector_store=vector_store,
                    files_per_batch=10,
                    adaptive_batching=False,
                    state_path=Path(tmpdir) / "state.json",
                )
            except Exception as exc:
                self.skipTest(f"Skipping due to local embedding initialization failure: {exc}")

            llm = FakeLLM()
            rag = RAGSystem(index=index, llm=llm, top_k=1)
            answer = rag.answer("What is alpha?")

            self.assertEqual(answer, "fake-answer")
            self.assertIn("Alpha beta gamma delta", llm.last_prompt)
            self.assertIn("Question: What is alpha?", llm.last_prompt)

    def test_rag_formats_context_with_sources(self):
        chunk = Chunk(text="Alpha content", source="doc.txt")
        index = FakeIndex([(chunk, 0.1)])
        llm = FakeLLM()
        rag = RAGSystem(index=index, llm=llm, top_k=5)

        answer = rag.answer("What is alpha?")
        self.assertEqual(answer, "fake-answer")
        self.assertIn("[Source: doc.txt]", llm.last_prompt)
        self.assertIn("Alpha content", llm.last_prompt)
        self.assertEqual(index.last_top_k, 5)

    def test_rag_handles_empty_results(self):
        index = FakeIndex([])
        llm = FakeLLM()
        rag = RAGSystem(index=index, llm=llm, top_k=2)

        rag.answer("Where is beta?")
        self.assertIn("Question: Where is beta?", llm.last_prompt)
        self.assertIn("Context:\n", llm.last_prompt)

    def test_rag_reranks_with_expanded_candidate_pool(self):
        chunks = [
            Chunk(text="one", source="1.txt"),
            Chunk(text="two", source="2.txt"),
            Chunk(text="three", source="3.txt"),
            Chunk(text="four", source="4.txt"),
        ]
        index = FakeIndex([(chunk, 0.1) for chunk in chunks])
        llm = FakeLLM()
        reranker = FakeReranker()
        rag = RAGSystem(index=index, llm=llm, top_k=2, reranker=reranker, rerank_expansion=2)

        answer, ranked, metrics = rag.answer_with_metrics("pick best")

        self.assertEqual(answer, "fake-answer")
        self.assertEqual(index.last_top_k, 4)
        self.assertEqual(reranker.last_top_k, 2)
        self.assertEqual([c.source for c, _ in ranked], ["4.txt", "3.txt"])
        self.assertIn("generate_ms", metrics)
        self.assertIn("rerank_ms", metrics)

    def test_rag_reranker_failure_falls_back_to_vector_order(self):
        chunks = [Chunk(text="one", source="1.txt"), Chunk(text="two", source="2.txt")]
        index = FakeIndex([(chunk, 0.1) for chunk in chunks])
        llm = FakeLLM()
        reranker = FakeReranker()
        reranker.raise_error = True
        rag = RAGSystem(index=index, llm=llm, top_k=1, reranker=reranker, rerank_expansion=3)

        answer = rag.answer("fallback")

        self.assertEqual(answer, "fake-answer")
        self.assertIn("one", llm.last_prompt)

    def test_rag_answer_with_metrics_respects_runtime_top_k_override(self):
        chunks = [
            Chunk(text="a", source="a.txt"),
            Chunk(text="b", source="b.txt"),
            Chunk(text="c", source="c.txt"),
        ]
        index = FakeIndex([(chunk, 0.1) for chunk in chunks])
        llm = FakeLLM()
        rag = RAGSystem(index=index, llm=llm, top_k=1)

        _answer, ranked, _metrics = rag.answer_with_metrics("override", top_k=2)

        self.assertEqual(index.last_top_k, 2)
        self.assertEqual(len(ranked), 2)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import pytest

from rag.embedders import FastEmbedEmbedder
from rag.index import Index
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


class RagSystemTests(unittest.TestCase):
    def test_rag_includes_context_in_prompt(self):
        pytest.importorskip("rag_core")
        pytest.importorskip("lancedb")
        pytest.importorskip("fastembed")

        root = Path(__file__).resolve().parent / "fixtures" / "rag_prompt"
        with tempfile.TemporaryDirectory() as tmpdir:
            vector_store = LanceDBVectorStore(db_path=tmpdir, table_name="vectors")
            embedder = FastEmbedEmbedder(prefer_gpu=False, max_retries=0, cpu_batch_size=16, gpu_batch_size=16)
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
        self.assertIn("Context:\n\nQuestion: Where is beta?", llm.last_prompt)


if __name__ == "__main__":
    unittest.main()

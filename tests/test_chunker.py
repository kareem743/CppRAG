import pytest

pytest.importorskip("rag_core")

from rag.chunker import RagCoreChunker
from rag.errors import ChunkingError
from rag.models import Chunk


class DummyChunk:
    def __init__(self, text: str, source: str):
        self.text = text
        self.source = source


def test_chunker_uses_parallel_api(monkeypatch) -> None:
    class Engine:
        def process_file_list_parallel(self, file_paths, chunk_size, overlap):
            return ([DummyChunk("alpha", "a.txt")],)

    import rag.chunker as chunker_module

    monkeypatch.setattr(chunker_module.rag_core, "IngestionEngine", Engine)
    chunker = RagCoreChunker()
    chunks = chunker.chunk_files(["a.txt"], chunk_size=5, overlap=1)
    assert chunks == [Chunk(text="alpha", source="a.txt")]


def test_chunker_uses_collector_api(monkeypatch) -> None:
    class Engine:
        def process_file_list(self, file_paths, chunk_size, overlap, collector):
            collector(DummyChunk("beta", "b.txt"))

    import rag.chunker as chunker_module

    monkeypatch.setattr(chunker_module.rag_core, "IngestionEngine", Engine)
    chunker = RagCoreChunker()
    chunks = chunker.chunk_files(["b.txt"], chunk_size=5, overlap=1)
    assert chunks == [Chunk(text="beta", source="b.txt")]


def test_chunker_raises_on_missing_api(monkeypatch) -> None:
    class Engine:
        pass

    import rag.chunker as chunker_module

    monkeypatch.setattr(chunker_module.rag_core, "IngestionEngine", Engine)
    chunker = RagCoreChunker()
    with pytest.raises(ChunkingError):
        chunker.chunk_files(["c.txt"], chunk_size=5, overlap=1)

import logging
from typing import List

import rag_core

from .errors import ChunkingError
from .interfaces import Chunker
from .models import Chunk

_LOGGER = logging.getLogger(__name__)


class RagCoreChunker(Chunker):
    def chunk_files(self, file_paths: List[str], chunk_size: int, overlap: int) -> List[Chunk]:
        try:
            engine = rag_core.IngestionEngine()
            if hasattr(engine, "process_file_list_parallel"):
                results = engine.process_file_list_parallel(file_paths, chunk_size, overlap)
                cpp_chunks = results[0]
            elif hasattr(engine, "process_file_list"):
                cpp_chunks = []

                def collector(chunk: object) -> None:
                    cpp_chunks.append(chunk)

                engine.process_file_list(file_paths, chunk_size, overlap, collector)
            else:
                raise AttributeError("rag_core.IngestionEngine has no file list processing method")
        except Exception as exc:
            _LOGGER.exception("Chunking failed")
            raise ChunkingError("Failed to chunk files") from exc

        chunks: List[Chunk] = []
        for chunk in cpp_chunks:
            chunks.append(Chunk(text=chunk.text, source=chunk.source))
        return chunks

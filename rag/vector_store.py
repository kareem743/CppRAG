import logging
from typing import List, Tuple

import lancedb

from .errors import VectorStoreError
from .interfaces import VectorStore
from .models import Chunk

_LOGGER = logging.getLogger(__name__)


class LanceDBVectorStore(VectorStore):
    def __init__(self, db_path: str, table_name: str):
        self._db_path = db_path
        self._table_name = table_name

    @property
    def db_path(self) -> str:
        return self._db_path

    @property
    def table_name(self) -> str:
        return self._table_name

    def ensure_schema(self, sample_vector: List[float]) -> None:
        try:
            db = lancedb.connect(self._db_path)
            table_names = db.list_tables() if hasattr(db, "list_tables") else db.table_names()
            if self._table_name not in table_names:
                schema_data = [
                    {"vector": list(sample_vector), "text": "init", "source": "init"}
                ]
                db.create_table(self._table_name, data=schema_data, mode="overwrite")
            table = db.open_table(self._table_name)
            if hasattr(table, "create_fts_index"):
                try:
                    table.create_fts_index("text")
                except Exception as exc:
                    _LOGGER.warning("Failed to create FTS index: %s", exc)
        except Exception as exc:
            _LOGGER.exception("Failed to ensure LanceDB schema")
            raise VectorStoreError("Failed to ensure schema") from exc

    def add(self, vectors: List[List[float]], texts: List[str], sources: List[str]) -> int:
        if not vectors:
            return 0
        if not (len(vectors) == len(texts) == len(sources)):
            raise VectorStoreError("Vector/text/source length mismatch")
        try:
            db = lancedb.connect(self._db_path)
            table = db.open_table(self._table_name)
            data = []
            for i, vec in enumerate(vectors):
                data.append(
                    {
                        "vector": vec,
                        "text": texts[i],
                        "source": sources[i],
                    }
                )
            table.add(data)
            return len(data)
        except Exception as exc:
            _LOGGER.exception("Failed to add vectors to LanceDB")
            raise VectorStoreError("Failed to add vectors") from exc

    def search(
        self,
        vector: List[float],
        query_text: str,
        top_k: int,
        candidate_k: int | None = None,
    ) -> List[Tuple[Chunk, float]]:
        try:
            db = lancedb.connect(self._db_path)
            table_names = db.list_tables() if hasattr(db, "list_tables") else db.table_names()
            if self._table_name not in table_names:
                return []
            table = db.open_table(self._table_name)
            limit = max(top_k, 50) if candidate_k is None else max(candidate_k, top_k)
            try:
                results = (
                    table.search(
                        query_text,
                        query_type="hybrid",
                        vector=list(vector),
                    )
                    .limit(limit)
                    .to_list()
                )
            except TypeError:
                _LOGGER.warning("Hybrid search not supported; falling back to vector search.")
                results = table.search(list(vector)).limit(limit).to_list()
        except Exception as exc:
            _LOGGER.exception("Failed to search LanceDB")
            raise VectorStoreError("Failed to search") from exc

        output: List[Tuple[Chunk, float]] = []
        for row in results:
            chunk = Chunk(text=row.get("text", ""), source=row.get("source", ""))
            score = float(row.get("_score", row.get("_distance", 0.0)))
            output.append((chunk, score))
        return output

    def compact(self) -> None:
        try:
            db = lancedb.connect(self._db_path)
            if self._table_name in db.table_names():
                table = db.open_table(self._table_name)
                table.compact_files()
                table.cleanup_old_versions()
        except Exception as exc:
            _LOGGER.warning("Compaction failed: %s", exc)

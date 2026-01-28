import pytest

pytest.importorskip("lancedb")

from rag.errors import VectorStoreError
from rag.vector_store import LanceDBVectorStore


class FakeTable:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.added = []

    def add(self, data):
        self.added.extend(data)

    def search(self, vector):
        self._search_vector = vector
        return self

    def limit(self, _count):
        return self

    def to_list(self):
        return self.rows


class FakeDB:
    def __init__(self, table_names=None, table=None):
        self._table_names = table_names or []
        self._table = table or FakeTable()
        self.created = None

    def table_names(self):
        return self._table_names

    def create_table(self, name, data, mode):
        self.created = {"name": name, "data": data, "mode": mode}
        self._table_names.append(name)

    def open_table(self, name):
        return self._table


def test_vector_store_ensure_schema_creates_table(monkeypatch) -> None:
    fake_db = FakeDB(table_names=[])
    monkeypatch.setattr("rag.vector_store.lancedb.connect", lambda _path: fake_db)
    store = LanceDBVectorStore(db_path="db", table_name="vectors")
    store.ensure_schema([0.1, 0.2])
    assert fake_db.created["name"] == "vectors"
    assert fake_db.created["mode"] == "overwrite"


def test_vector_store_add_handles_empty_and_mismatch(monkeypatch) -> None:
    fake_db = FakeDB(table_names=["vectors"])
    monkeypatch.setattr("rag.vector_store.lancedb.connect", lambda _path: fake_db)
    store = LanceDBVectorStore(db_path="db", table_name="vectors")
    assert store.add([], [], []) == 0
    with pytest.raises(VectorStoreError):
        store.add([[1.0]], ["a"], [])


def test_vector_store_add_inserts_rows(monkeypatch) -> None:
    table = FakeTable()
    fake_db = FakeDB(table_names=["vectors"], table=table)
    monkeypatch.setattr("rag.vector_store.lancedb.connect", lambda _path: fake_db)
    store = LanceDBVectorStore(db_path="db", table_name="vectors")
    count = store.add([[0.1, 0.2]], ["text"], ["src"])
    assert count == 1
    assert table.added[0]["text"] == "text"


def test_vector_store_search_returns_chunks(monkeypatch) -> None:
    rows = [
        {"text": "alpha", "source": "a.txt", "_distance": 0.1},
        {"text": "beta", "source": "b.txt", "_distance": 0.2},
    ]
    table = FakeTable(rows=rows)
    fake_db = FakeDB(table_names=["vectors"], table=table)
    monkeypatch.setattr("rag.vector_store.lancedb.connect", lambda _path: fake_db)
    store = LanceDBVectorStore(db_path="db", table_name="vectors")
    results = store.search([0.1, 0.2], top_k=2)
    assert len(results) == 2
    assert results[0][0].text == "alpha"
    assert results[1][0].source == "b.txt"


def test_vector_store_search_no_table(monkeypatch) -> None:
    fake_db = FakeDB(table_names=[])
    monkeypatch.setattr("rag.vector_store.lancedb.connect", lambda _path: fake_db)
    store = LanceDBVectorStore(db_path="db", table_name="vectors")
    assert store.search([0.1], top_k=1) == []


def test_vector_store_connect_error(monkeypatch) -> None:
    def boom(_path):
        raise RuntimeError("fail")

    monkeypatch.setattr("rag.vector_store.lancedb.connect", boom)
    store = LanceDBVectorStore(db_path="db", table_name="vectors")
    with pytest.raises(VectorStoreError):
        store.ensure_schema([0.1])

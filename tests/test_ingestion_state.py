import json
import os
import time
from pathlib import Path

import pytest

from rag.errors import IngestionError
from rag.ingestion_state import load_state, plan_files, save_state


def test_plan_files_detects_changes(tmp_path: Path) -> None:
    data_file = tmp_path / "doc.txt"
    data_file.write_text("alpha beta gamma", encoding="utf-8")

    state_path = tmp_path / "state.json"
    state = load_state(state_path)
    to_process, pending = plan_files([str(data_file)], state)
    assert str(data_file) in to_process
    assert str(data_file) in pending

    state.update(pending)
    save_state(state_path, state)

    loaded = load_state(state_path)
    to_process, _pending = plan_files([str(data_file)], loaded)
    assert to_process == []

    data_file.write_text("alpha beta gamma delta", encoding="utf-8")
    os_time = time.time() + 2
    os.utime(data_file, (os_time, os_time))

    to_process, pending = plan_files([str(data_file)], loaded)
    assert str(data_file) in to_process


def test_load_state_missing_file_returns_empty(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert load_state(missing) == {}


def test_load_state_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{invalid", encoding="utf-8")
    with pytest.raises(IngestionError):
        load_state(path)


def test_load_state_skips_invalid_entries(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    payload = {
        "version": 1,
        "files": {
            "ok.txt": {"hash": "abc", "mtime": 1.0, "size": 2},
            "bad.txt": {"hash": "def"},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    state = load_state(path)
    assert "ok.txt" in state
    assert "bad.txt" not in state


def test_plan_files_ignores_missing_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    to_process, pending = plan_files([str(missing)], {})
    assert to_process == []
    assert pending == {}

import hashlib
import json
import os
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .errors import IngestionError


@dataclass(frozen=True)
class FileMeta:
    hash: str
    mtime: float
    size: int


def _hash_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        # Read in 1MB chunks to be memory efficient
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_state(path: Path) -> Dict[str, FileMeta]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise IngestionError(f"Failed to read ingestion state: {path}") from exc
    files = raw.get("files", {})
    state: Dict[str, FileMeta] = {}
    for key, value in files.items():
        try:
            state[key] = FileMeta(
                hash=value["hash"],
                mtime=float(value["mtime"]),
                size=int(value["size"]),
            )
        except Exception:
            continue
    return state


def save_state(path: Path, state: Dict[str, FileMeta]) -> None:
    payload = {
        "version": 1,
        "files": {
            key: {"hash": meta.hash, "mtime": meta.mtime, "size": meta.size}
            for key, meta in state.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _process_file_task(args) -> Tuple[str, FileMeta, bool]:
    """Worker function for threading."""
    path, existing_meta = args
    try:
        stat = os.stat(path)
    except OSError:
        return None

    # 1. Fast Check: If size and date match, skip hashing
    if existing_meta and existing_meta.size == stat.st_size and existing_meta.mtime == stat.st_mtime:
        return path, existing_meta, False

    # 2. Slow Check: Hash the file (CPU Intensive)
    file_hash = _hash_file(path)
    meta = FileMeta(hash=file_hash, mtime=stat.st_mtime, size=stat.st_size)

    # 3. If hash is new/changed, mark for processing
    needs_ingest = (existing_meta is None or existing_meta.hash != file_hash)
    return path, meta, needs_ingest


def plan_files(
        file_paths: Iterable[str],
        state: Dict[str, FileMeta],
) -> Tuple[List[str], Dict[str, FileMeta]]:
    to_process: List[str] = []
    pending_meta: Dict[str, FileMeta] = {}

    paths = list(file_paths)
    total = len(paths)
    print(f"Checking state for {total} files (Multi-threaded)...")

    # Prepare arguments for workers
    tasks = [(p, state.get(p)) for p in paths]

    # Use max_workers=None (defaults to CPU count * 5)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(_process_file_task, tasks)

        for i, result in enumerate(results):
            if i % 1000 == 0:
                print(f"Scanned {i}/{total} files...", end="\r")

            if result is None:
                continue

            path, meta, needs_ingest = result
            pending_meta[path] = meta
            if needs_ingest:
                to_process.append(path)

    print(f"\nPlanning complete. {len(to_process)} files need ingestion.")
    return to_process, pending_meta
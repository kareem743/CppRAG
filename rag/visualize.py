import html
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import rag_core

_LOGGER = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class FileChunks:
    source: Path
    chunks: List[str]


def _chunk_directory(
        directory: Path,
        chunk_size: int,
        overlap: int,
        extensions: Optional[Iterable[str]],
) -> List[FileChunks]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    if extensions is not None:
        _LOGGER.warning(
            "extensions filtering is handled in C++ and cannot be customized yet; ignoring %s",
            list(extensions),
        )

    engine = rag_core.IngestionEngine()
    if hasattr(engine, "process_directory_streaming"):
        cpp_chunks = []

        def handle_chunk(cpp_chunk: object) -> None:
            cpp_chunks.append(cpp_chunk)

        engine.process_directory_streaming(
            str(directory), chunk_size, overlap, handle_chunk
        )
    elif hasattr(engine, "process_directory_with_stats"):
        cpp_chunks, _stats = engine.process_directory_with_stats(
            str(directory), chunk_size, overlap
        )
    elif hasattr(engine, "process_directory"):
        cpp_chunks = engine.process_directory(str(directory), chunk_size, overlap)
    else:
        raise AttributeError("rag_core.IngestionEngine has no processing methods")

    files: List[FileChunks] = []
    seen: Dict[str, FileChunks] = {}
    for chunk in cpp_chunks:
        source = chunk.source
        entry = seen.get(source)
        if entry is None:
            entry = FileChunks(source=Path(source), chunks=[])
            seen[source] = entry
            files.append(entry)
        entry.chunks.append(chunk.text)
    return files


def _render_chunk_html(text: str, overlap: int, has_overlap: bool) -> str:
    tokens = _TOKEN_RE.findall(text)
    rendered: List[str] = []
    for idx, token in enumerate(tokens):
        safe = html.escape(token)
        if has_overlap and overlap > 0 and idx < overlap:
            rendered.append(f'<span class="overlap" title="Overlap from previous chunk">{safe}</span>')
        else:
            rendered.append(f'<span class="token">{safe}</span>')
    return " ".join(rendered)


def export_chunks_html(
        directory: Path,
        output_path: Path,
        chunk_size: int = 200,
        overlap: int = 50,
        extensions: Optional[Iterable[str]] = None,
) -> None:
    files = _chunk_directory(directory, chunk_size, overlap, extensions)
    total_chunks = sum(len(file.chunks) for file in files)
    total_files = len(files)

    # Calculate statistics
    all_tokens = []
    for file in files:
        for chunk in file.chunks:
            tokens = _TOKEN_RE.findall(chunk)
            all_tokens.append(len(tokens))

    avg_tokens = sum(all_tokens) / len(all_tokens) if all_tokens else 0
    min_tokens = min(all_tokens) if all_tokens else 0
    max_tokens = max(all_tokens) if all_tokens else 0

    palette = [
        "#fef3e2",
        "#e8f4f8",
        "#f0e8f8",
        "#e8f8f0",
        "#ffe8f0",
        "#f8f0e8",
    ]
    css_palette = "\n".join(
        f".chunk-{idx} {{ --chunk-color: {color}; }}" for idx, color in enumerate(palette)
    )

    parts: List[str] = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Chunk Visualization</title>",
        "<style>",
        ":root {",
        "  --primary: #6366f1;",
        "  --primary-light: #818cf8;",
        "  --primary-dark: #4f46e5;",
        "  --accent: #ec4899;",
        "  --border: #e0e7ff;",
        "  --text: #1e1b4b;",
        "  --muted: #64748b;",
        "  --overlap-bg: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);",
        "  --overlap-border: #f59e0b;",
        "  --bg: #ffffff;",
        "  --surface: linear-gradient(to bottom, #faf5ff 0%, #f5f3ff 100%);",
        "  --shadow: rgba(99, 102, 241, 0.08);",
        "  --shadow-lg: rgba(99, 102, 241, 0.15);",
        "}",
        "* { box-sizing: border-box; }",
        "body {",
        "  margin: 0;",
        "  background: var(--surface);",
        "  color: var(--text);",
        "  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;",
        "  line-height: 1.6;",
        "  min-height: 100vh;",
        "}",
        "body::before {",
        "  content: '';",
        "  position: fixed;",
        "  top: 0;",
        "  left: 0;",
        "  right: 0;",
        "  height: 400px;",
        "  background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);",
        "  opacity: 0.03;",
        "  z-index: 0;",
        "  pointer-events: none;",
        "}",
        ".page {",
        "  max-width: 1200px;",
        "  margin: 0 auto;",
        "  padding: 40px 24px;",
        "  position: relative;",
        "  z-index: 1;",
        "}",
        "header {",
        "  background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);",
        "  backdrop-filter: blur(10px);",
        "  border-radius: 20px;",
        "  padding: 40px;",
        "  margin-bottom: 32px;",
        "  box-shadow: 0 8px 32px var(--shadow-lg), 0 0 0 1px rgba(99, 102, 241, 0.1);",
        "  border: 1px solid rgba(255, 255, 255, 0.8);",
        "}",
        "h1 {",
        "  margin: 0 0 12px;",
        "  font-size: 36px;",
        "  font-weight: 800;",
        "  background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%);",
        "  -webkit-background-clip: text;",
        "  -webkit-text-fill-color: transparent;",
        "  background-clip: text;",
        "  letter-spacing: -0.5px;",
        "}",
        ".stats-grid {",
        "  display: grid;",
        "  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));",
        "  gap: 16px;",
        "  margin-top: 20px;",
        "}",
        ".stat-card {",
        "  background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);",
        "  backdrop-filter: blur(10px);",
        "  border: 1px solid rgba(99, 102, 241, 0.2);",
        "  border-radius: 12px;",
        "  padding: 20px;",
        "  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);",
        "  position: relative;",
        "  overflow: hidden;",
        "}",
        ".stat-card::before {",
        "  content: '';",
        "  position: absolute;",
        "  top: 0;",
        "  left: 0;",
        "  right: 0;",
        "  height: 3px;",
        "  background: linear-gradient(90deg, #6366f1 0%, #ec4899 100%);",
        "  opacity: 0;",
        "  transition: opacity 0.3s;",
        "}",
        ".stat-card:hover {",
        "  transform: translateY(-4px);",
        "  box-shadow: 0 12px 24px var(--shadow-lg);",
        "  border-color: var(--primary);",
        "}",
        ".stat-card:hover::before {",
        "  opacity: 1;",
        "}",
        ".stat-label {",
        "  font-size: 12px;",
        "  text-transform: uppercase;",
        "  color: var(--muted);",
        "  font-weight: 600;",
        "  letter-spacing: 0.5px;",
        "}",
        ".stat-value {",
        "  font-size: 24px;",
        "  font-weight: 700;",
        "  color: var(--primary);",
        "  margin-top: 4px;",
        "}",
        ".controls {",
        "  background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);",
        "  backdrop-filter: blur(10px);",
        "  border-radius: 16px;",
        "  padding: 24px;",
        "  margin-bottom: 32px;",
        "  box-shadow: 0 4px 16px var(--shadow);",
        "  border: 1px solid rgba(255, 255, 255, 0.8);",
        "  display: flex;",
        "  gap: 12px;",
        "  flex-wrap: wrap;",
        "  align-items: center;",
        "}",
        ".search-box {",
        "  flex: 1;",
        "  min-width: 250px;",
        "  padding: 12px 18px;",
        "  border: 2px solid var(--border);",
        "  border-radius: 12px;",
        "  font-size: 14px;",
        "  outline: none;",
        "  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);",
        "  background: white;",
        "}",
        ".search-box:focus {",
        "  border-color: var(--primary);",
        "  box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);",
        "  transform: translateY(-1px);",
        "}",
        ".btn {",
        "  padding: 12px 24px;",
        "  border: 2px solid var(--border);",
        "  border-radius: 12px;",
        "  background: white;",
        "  color: var(--text);",
        "  font-size: 14px;",
        "  cursor: pointer;",
        "  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);",
        "  font-weight: 600;",
        "  position: relative;",
        "  overflow: hidden;",
        "}",
        ".btn::before {",
        "  content: '';",
        "  position: absolute;",
        "  top: 50%;",
        "  left: 50%;",
        "  width: 0;",
        "  height: 0;",
        "  border-radius: 50%;",
        "  background: rgba(99, 102, 241, 0.1);",
        "  transform: translate(-50%, -50%);",
        "  transition: width 0.6s, height 0.6s;",
        "}",
        ".btn:hover::before {",
        "  width: 300px;",
        "  height: 300px;",
        "}",
        ".btn:hover {",
        "  border-color: var(--primary);",
        "  transform: translateY(-2px);",
        "  box-shadow: 0 4px 12px var(--shadow);",
        "}",
        ".btn.active {",
        "  background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);",
        "  color: white;",
        "  border-color: var(--primary);",
        "  box-shadow: 0 4px 16px var(--shadow-lg);",
        "}",
        ".file {",
        "  background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);",
        "  backdrop-filter: blur(10px);",
        "  border-radius: 16px;",
        "  margin-bottom: 24px;",
        "  box-shadow: 0 4px 16px var(--shadow);",
        "  border: 1px solid rgba(255, 255, 255, 0.8);",
        "  overflow: hidden;",
        "  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);",
        "}",
        ".file:hover {",
        "  box-shadow: 0 8px 24px var(--shadow-lg);",
        "}",
        ".file-header {",
        "  padding: 20px 24px;",
        "  border-bottom: 1px solid var(--border);",
        "  cursor: pointer;",
        "  display: flex;",
        "  justify-content: space-between;",
        "  align-items: center;",
        "  transition: background 0.2s;",
        "}",
        ".file-header:hover {",
        "  background: var(--surface);",
        "}",
        ".file-title {",
        "  margin: 0;",
        "  font-size: 16px;",
        "  font-weight: 600;",
        "  color: var(--text);",
        "}",
        ".file-meta {",
        "  font-size: 13px;",
        "  color: var(--muted);",
        "  margin-top: 4px;",
        "}",
        ".toggle-icon {",
        "  font-size: 20px;",
        "  color: var(--muted);",
        "  transition: transform 0.3s;",
        "}",
        ".file.collapsed .toggle-icon {",
        "  transform: rotate(-90deg);",
        "}",
        ".file-content {",
        "  padding: 24px;",
        "  max-height: 100000px;",
        "  overflow: hidden;",
        "  transition: max-height 0.3s ease-out, padding 0.3s;",
        "}",
        ".file.collapsed .file-content {",
        "  max-height: 0;",
        "  padding: 0 24px;",
        "}",
        ".chunk {",
        "  background: var(--chunk-color);",
        "  border: 2px solid rgba(99, 102, 241, 0.15);",
        "  border-radius: 14px;",
        "  padding: 20px 24px;",
        "  margin-bottom: 16px;",
        "  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);",
        "  position: relative;",
        "  overflow: hidden;",
        "}",
        ".chunk::before {",
        "  content: '';",
        "  position: absolute;",
        "  top: 0;",
        "  left: 0;",
        "  width: 4px;",
        "  height: 100%;",
        "  background: linear-gradient(180deg, #6366f1 0%, #ec4899 100%);",
        "  opacity: 0;",
        "  transition: opacity 0.3s;",
        "}",
        ".chunk:hover {",
        "  transform: translateX(4px);",
        "  box-shadow: 0 8px 24px var(--shadow-lg);",
        "  border-color: var(--primary);",
        "}",
        ".chunk:hover::before {",
        "  opacity: 1;",
        "}",
        ".chunk.hidden {",
        "  display: none;",
        "}",
        ".chunk-meta {",
        "  font-size: 11px;",
        "  color: var(--muted);",
        "  margin-bottom: 12px;",
        "  display: flex;",
        "  gap: 16px;",
        "  flex-wrap: wrap;",
        "  font-weight: 500;",
        "}",
        ".chunk-text {",
        "  line-height: 1.7;",
        "  font-size: 14px;",
        "  font-family: 'Menlo', 'Monaco', 'Courier New', monospace;",
        "}",
        ".token {",
        "  display: inline;",
        "  padding: 1px 0;",
        "}",
        ".overlap {",
        "  background: var(--overlap-bg);",
        "  border-bottom: 2px solid var(--overlap-border);",
        "  padding: 3px 6px;",
        "  border-radius: 4px;",
        "  font-weight: 600;",
        "  box-shadow: 0 2px 4px rgba(245, 158, 11, 0.2);",
        "}",
        ".highlight {",
        "  background: linear-gradient(135deg, #fef08a 0%, #fde047 100%);",
        "  padding: 3px 6px;",
        "  border-radius: 4px;",
        "  box-shadow: 0 2px 4px rgba(250, 204, 21, 0.3);",
        "  font-weight: 600;",
        "}",
        ".empty-state {",
        "  text-align: center;",
        "  padding: 60px 20px;",
        "  color: var(--muted);",
        "}",
        "@media (max-width: 768px) {",
        "  .page { padding: 16px; }",
        "  header { padding: 24px 20px; }",
        "  h1 { font-size: 24px; }",
        "  .controls { flex-direction: column; align-items: stretch; }",
        "  .search-box { min-width: 100%; }",
        "  .stats-grid { grid-template-columns: 1fr; }",
        "}",
        css_palette,
        "</style>",
        "</head>",
        "<body>",
        '<div class="page">',
        "<header>",
        "<h1>📊 Chunk Visualization</h1>",
        f'<div class="file-meta">Directory: {html.escape(str(directory))}</div>',
        '<div class="stats-grid">',
        '<div class="stat-card">',
        '<div class="stat-label">Total Files</div>',
        f'<div class="stat-value">{total_files}</div>',
        '</div>',
        '<div class="stat-card">',
        '<div class="stat-label">Total Chunks</div>',
        f'<div class="stat-value">{total_chunks}</div>',
        '</div>',
        '<div class="stat-card">',
        '<div class="stat-label">Chunk Size</div>',
        f'<div class="stat-value">{chunk_size}</div>',
        '</div>',
        '<div class="stat-card">',
        '<div class="stat-label">Overlap</div>',
        f'<div class="stat-value">{overlap}</div>',
        '</div>',
        '<div class="stat-card">',
        '<div class="stat-label">Avg Tokens</div>',
        f'<div class="stat-value">{avg_tokens:.0f}</div>',
        '</div>',
        '<div class="stat-card">',
        '<div class="stat-label">Token Range</div>',
        f'<div class="stat-value">{min_tokens}-{max_tokens}</div>',
        '</div>',
        '</div>',
        "</header>",
        '<div class="controls">',
        '<input type="text" class="search-box" placeholder="Search chunks..." id="searchBox">',
        '<button class="btn" id="toggleOverlap">Show Overlap Only</button>',
        '<button class="btn" id="expandAll">Expand All</button>',
        '<button class="btn" id="collapseAll">Collapse All</button>',
        '</div>',
    ]

    for file_idx, file in enumerate(files):
        parts.append(f'<section class="file" id="file-{file_idx}">')
        parts.append('<div class="file-header" onclick="toggleFile(this)">')
        parts.append('<div>')
        parts.append(f'<h2 class="file-title">{html.escape(str(file.source))}</h2>')
        parts.append(f'<div class="file-meta">{len(file.chunks)} chunks</div>')
        parts.append('</div>')
        parts.append('<span class="toggle-icon">▼</span>')
        parts.append('</div>')
        parts.append('<div class="file-content">')

        step = chunk_size - overlap
        for idx, chunk in enumerate(file.chunks):
            tokens = _TOKEN_RE.findall(chunk)
            start = idx * step
            end = start + len(tokens) - 1 if tokens else start
            color_class = f"chunk-{idx % len(palette)}"
            has_overlap = idx > 0
            overlap_attr = 'data-has-overlap="true"' if has_overlap else ''

            parts.append(f'<div class="chunk {color_class}" {overlap_attr}>')
            parts.append('<div class="chunk-meta">')
            parts.append(f'<span>Chunk {idx + 1}/{len(file.chunks)}</span>')
            parts.append(f'<span>Tokens {start}-{end}</span>')
            parts.append(f'<span>Size: {len(tokens)}</span>')
            if has_overlap:
                parts.append(f'<span>Overlap: {overlap} tokens</span>')
            parts.append('</div>')
            rendered = _render_chunk_html(chunk, overlap, has_overlap)
            parts.append(f'<div class="chunk-text">{rendered}</div>')
            parts.append('</div>')

        parts.append('</div>')
        parts.append('</section>')

    if not files:
        parts.append('<div class="empty-state">')
        parts.append('<div style="font-size: 48px; margin-bottom: 16px;">📭</div>')
        parts.append('<div>No chunks found for the selected files.</div>')
        parts.append('</div>')

    parts.append("<script>")
    parts.append("""
function toggleFile(header) {
    header.parentElement.classList.toggle('collapsed');
}

let showOverlapOnly = false;
document.getElementById('toggleOverlap').addEventListener('click', function() {
    showOverlapOnly = !showOverlapOnly;
    this.classList.toggle('active');
    this.textContent = showOverlapOnly ? 'Show All Chunks' : 'Show Overlap Only';
    filterChunks();
});

document.getElementById('expandAll').addEventListener('click', function() {
    document.querySelectorAll('.file').forEach(f => f.classList.remove('collapsed'));
});

document.getElementById('collapseAll').addEventListener('click', function() {
    document.querySelectorAll('.file').forEach(f => f.classList.add('collapsed'));
});

document.getElementById('searchBox').addEventListener('input', function(e) {
    filterChunks();
});

function filterChunks() {
    const query = document.getElementById('searchBox').value.toLowerCase();
    const chunks = document.querySelectorAll('.chunk');

    chunks.forEach(chunk => {
        const text = chunk.querySelector('.chunk-text').textContent.toLowerCase();
        const hasOverlap = chunk.hasAttribute('data-has-overlap');
        const matchesSearch = !query || text.includes(query);
        const matchesOverlap = !showOverlapOnly || hasOverlap;

        if (matchesSearch && matchesOverlap) {
            chunk.classList.remove('hidden');
            if (query) {
                highlightText(chunk.querySelector('.chunk-text'), query);
            } else {
                removeHighlights(chunk.querySelector('.chunk-text'));
            }
        } else {
            chunk.classList.add('hidden');
        }
    });
}

function highlightText(element, query) {
    const html = element.innerHTML;
    const cleanHtml = html.replace(/<mark class="highlight">|<\\/mark>/g, '');
    element.innerHTML = cleanHtml;

    if (!query) return;

    const regex = new RegExp(`(${escapeRegex(query)})`, 'gi');
    element.innerHTML = element.innerHTML.replace(regex, '<mark class="highlight">$1</mark>');
}

function removeHighlights(element) {
    element.innerHTML = element.innerHTML.replace(/<mark class="highlight">|<\\/mark>/g, '');
}

function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
}
""")
    parts.append("</script>")
    parts.append("</div>")
    parts.append("</body>")
    parts.append("</html>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    _LOGGER.info("Wrote enhanced chunk visualization to %s", output_path)

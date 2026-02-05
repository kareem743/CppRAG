import { useMemo, useRef } from "react";

type ChunkSpan = { start: number; end: number };

interface ChunkVisualizerProps {
  fileContent: string;
  chunks: ChunkSpan[];
}

type Token = {
  text: string;
  start: number;
  end: number;
  chunkIndexes: number[];
};

const tokenRegex = /\S+/g;

const tokenize = (content: string): Token[] => {
  const tokens: Token[] = [];
  let match = tokenRegex.exec(content);
  while (match) {
    const text = match[0];
    const start = match.index;
    const end = start + text.length;
    tokens.push({ text, start, end, chunkIndexes: [] });
    match = tokenRegex.exec(content);
  }
  return tokens;
};

const mapTokensToChunks = (tokens: Token[], chunks: ChunkSpan[]): Token[] => {
  return tokens.map((token) => {
    const indexes: number[] = [];
    chunks.forEach((chunk, idx) => {
      if (token.start >= chunk.start && token.end <= chunk.end) {
        indexes.push(idx);
      }
    });
    return { ...token, chunkIndexes: indexes };
  });
};

export default function ChunkVisualizer({
  fileContent,
  chunks,
}: ChunkVisualizerProps) {
  const tokenRefs = useRef<Record<number, HTMLSpanElement | null>>({});

  const tokens = useMemo(() => {
    if (!fileContent) {
      return [];
    }
    const baseTokens = tokenize(fileContent);
    return mapTokensToChunks(baseTokens, chunks);
  }, [fileContent, chunks]);

  const handleScrollToChunk = (chunkIndex: number) => {
    const target = tokenRefs.current[chunkIndex];
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex items-center justify-between text-xs uppercase tracking-widest text-slate-400">
        <span>Chunk Visualizer</span>
        <span className="text-[10px] text-slate-500">
          {chunks.length} chunks
        </span>
      </div>
      <div className="rounded-lg border border-border bg-bg/70 p-4 font-mono text-xs leading-relaxed text-slate-200">
        <div className="max-h-[70vh] overflow-y-auto">
          {tokens.length === 0 ? (
            <div className="text-slate-500">No file loaded.</div>
          ) : (
            <div className="flex flex-wrap gap-x-1 gap-y-2">
              {tokens.map((token, index) => {
                const overlaps = token.chunkIndexes.length > 1;
                const inChunk = token.chunkIndexes.length === 1;
                const chunkIndex = token.chunkIndexes[0] ?? -1;
                return (
                  <span
                    key={`${token.start}-${index}`}
                    ref={(node) => {
                      if (chunkIndex >= 0) {
                        tokenRefs.current[chunkIndex] = node;
                      }
                    }}
                    onClick={() => {
                      if (chunkIndex >= 0) {
                        handleScrollToChunk(chunkIndex);
                      }
                    }}
                    className={`cursor-pointer rounded px-1 transition ${
                      overlaps
                        ? "bg-amber-400/30 text-amber-100"
                        : inChunk
                        ? "bg-indigo-500/20 text-indigo-100"
                        : "text-slate-300"
                    }`}
                    title={
                      overlaps
                        ? "Overlap between chunks"
                        : inChunk
                        ? `Chunk ${chunkIndex + 1}`
                        : "Unchunked token"
                    }
                  >
                    {token.text}
                  </span>
                );
              })}
            </div>
          )}
        </div>
      </div>
      <div className="text-xs text-slate-500">
        Click a highlighted token to jump to its chunk.
      </div>
    </div>
  );
}

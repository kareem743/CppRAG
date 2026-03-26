import { Fragment, useMemo, useRef } from "react";

type ChunkSpan = { start: number; end: number };

interface ChunkVisualizerProps {
  fileContent: string;
  chunks: ChunkSpan[];
}

type Token = {
  text: string;
  index: number;
  chunkIndexes: number[];
};

const tokenRegex = /\S+/g;

const tokenize = (content: string): Token[] => {
  const tokens: Token[] = [];
  let match = tokenRegex.exec(content);
  let index = 0;
  while (match) {
    const text = match[0];
    tokens.push({ text, index, chunkIndexes: [] });
    index += 1;
    match = tokenRegex.exec(content);
  }
  return tokens;
};

const mapTokensToChunks = (tokens: Token[], chunks: ChunkSpan[]): Token[] => {
  return tokens.map((token) => {
    const indexes: number[] = [];
    chunks.forEach((chunk, idx) => {
      if (token.index >= chunk.start && token.index <= chunk.end) {
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
  const chunkRefs = useRef<Record<number, HTMLDivElement | null>>({});

  const tokens = useMemo(() => {
    if (!fileContent) {
      return [];
    }
    const baseTokens = tokenize(fileContent);
    return mapTokensToChunks(baseTokens, chunks);
  }, [fileContent, chunks]);

  const handleScrollToChunk = (chunkIndex: number) => {
    const target = chunkRefs.current[chunkIndex];
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  const unchunkedTokens = useMemo(() => {
    return tokens.filter((token) => token.chunkIndexes.length === 0);
  }, [tokens]);

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.3em] text-muted">
        <span>Chunk Visualizer</span>
        <span className="text-[10px] text-muted">{chunks.length} chunks</span>
      </div>
      <div className="flex-1 rounded-xl border border-slate-500/40 bg-slate-950/40 p-4 font-mono text-xs leading-relaxed text-primary">
        <div className="h-full overflow-y-auto">
          {tokens.length === 0 ? (
            <div className="text-muted">No file loaded.</div>
          ) : (
            <div className="flex flex-col gap-4">
              {chunks.map((chunk, idx) => {
                const chunkTokens = tokens.filter(
                  (token) => token.index >= chunk.start && token.index <= chunk.end
                );
                return (
                  <Fragment key={`chunk-${idx}`}>
                    <div
                      ref={(node) => {
                        if (node) {
                          chunkRefs.current[idx] = node;
                        }
                      }}
                      className="flex items-center justify-between rounded-lg border border-slate-700/60 bg-slate-900/60 px-3 py-2 text-[11px] uppercase tracking-[0.25em] text-muted"
                    >
                      <span>Chunk {idx + 1}</span>
                      <span className="text-[10px] text-muted">
                        tokens {chunk.start}–{chunk.end}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-x-1 gap-y-2">
                      {chunkTokens.map((token, index) => {
                        const overlaps = token.chunkIndexes.length > 1;
                        return (
                          <span
                            key={`${token.index}-${index}-chunk-${idx}`}
                            onClick={() => handleScrollToChunk(idx)}
                            className={`cursor-pointer rounded px-1 transition ${
                              overlaps
                                ? "bg-border text-primary underline decoration-2 underline-offset-2 decoration-amber-400/80"
                                : "bg-border text-primary"
                            }`}
                            title={
                              overlaps
                                ? "Overlap between chunks"
                                : `Chunk ${idx + 1}`
                            }
                          >
                            {token.text}
                          </span>
                        );
                      })}
                    </div>
                  </Fragment>
                );
              })}
              {unchunkedTokens.length > 0 ? (
                <Fragment>
                  <div className="flex items-center justify-between rounded-lg border border-dashed border-slate-700/60 bg-slate-900/30 px-3 py-2 text-[11px] uppercase tracking-[0.25em] text-muted">
                    <span>Unchunked</span>
                    <span className="text-[10px] text-muted">
                      {unchunkedTokens.length} tokens
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-x-1 gap-y-2 text-muted">
                    {unchunkedTokens.map((token, index) => (
                      <span key={`${token.index}-${index}-unchunked`}>
                        {token.text}
                      </span>
                    ))}
                  </div>
                </Fragment>
              ) : null}
            </div>
          )}
        </div>
      </div>
      <div className="text-xs text-ink/50">
        Click a highlighted token to jump to its chunk.
      </div>
    </div>
  );
}

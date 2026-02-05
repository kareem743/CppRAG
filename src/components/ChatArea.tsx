import type { Message } from "../types";
import { useAppStore } from "../store";

const shimmerStyles = `
@keyframes shimmer {
  0% { background-position: -400px 0; }
  100% { background-position: 400px 0; }
}
`;

const aiLatency = (message: Message): string | null => {
  if (!message.metrics) {
    return null;
  }
  const latency = message.metrics["latency_ms"];
  return typeof latency === "number" ? `${latency.toFixed(0)}ms` : null;
};

export default function ChatArea() {
  const chatHistory = useAppStore((state) => state.chatHistory);
  const isGenerating = useAppStore((state) => state.isGenerating);

  return (
    <div className="flex h-full flex-col gap-4">
      <style>{shimmerStyles}</style>
      <div className="flex-1 space-y-4 overflow-y-auto pr-2">
        {chatHistory.length === 0 && !isGenerating && (
          <div className="rounded-lg border border-border bg-surface/60 p-6 text-sm text-slate-400">
            Ask a question to begin the session.
          </div>
        )}

        {chatHistory.map((message, idx) => {
          const isUser = message.role === "user";
          return (
            <div
              key={`${message.role}-${idx}`}
              className={`flex ${isUser ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[75%] rounded-lg px-4 py-3 text-sm shadow-sm ${
                  isUser
                    ? "bg-indigo-600/40 text-slate-100 border border-indigo-400/30"
                    : "bg-slate-900/80 text-emerald-100 border border-border font-mono"
                }`}
              >
                <div className="whitespace-pre-wrap">{message.content}</div>
                {!isUser && (
                  <div className="mt-3 flex items-center justify-between text-[11px] text-slate-400">
                    <span className="uppercase tracking-widest">
                      Terminal Log
                    </span>
                    {aiLatency(message) && (
                      <span className="rounded-full bg-indigo-500/20 px-2 py-0.5 text-indigo-200">
                        ⚡ {aiLatency(message)}
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {isGenerating && (
          <div className="flex justify-start">
            <div className="w-full max-w-[75%] rounded-lg border border-border bg-slate-900/80 p-4">
              <div className="h-4 w-2/3 rounded bg-gradient-to-r from-slate-800 via-slate-700 to-slate-800"
                style={{
                  backgroundSize: "400px 100%",
                  animation: "shimmer 1.4s infinite linear",
                }}
              />
              <div className="mt-3 h-3 w-1/2 rounded bg-gradient-to-r from-slate-800 via-slate-700 to-slate-800"
                style={{
                  backgroundSize: "400px 100%",
                  animation: "shimmer 1.4s infinite linear",
                }}
              />
              <div className="mt-3 h-3 w-1/3 rounded bg-gradient-to-r from-slate-800 via-slate-700 to-slate-800"
                style={{
                  backgroundSize: "400px 100%",
                  animation: "shimmer 1.4s infinite linear",
                }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

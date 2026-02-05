import { Line, LineChart, ResponsiveContainer, Tooltip } from "recharts";

import { useAppStore } from "../store";

type SparkPoint = { t: number; value: number };

const buildSparkData = (base: number): SparkPoint[] => {
  const normalized = Number.isFinite(base) ? base : 0;
  return Array.from({ length: 14 }, (_, idx) => {
    const wave = Math.sin(idx / 2.5) * 6;
    const drift = (idx - 7) * 0.6;
    return { t: idx, value: Math.max(0, normalized + wave + drift) };
  });
};

export default function SystemDashboard() {
  const systemStats = useAppStore((state) => state.systemStats);
  const ingestionRate = systemStats?.ingestion_rate ?? 0;
  const gpuActive = systemStats?.gpu_active ?? false;
  const vectorCount = systemStats?.vector_count ?? 0;
  const sparkData = buildSparkData(ingestionRate);

  return (
    <div className="flex h-full flex-col gap-6">
      <div>
        <div className="flex items-center justify-between text-xs uppercase tracking-widest text-slate-400">
          <span>Live Ingestion Speed</span>
          <span
            className="text-[10px] text-slate-500"
            title="Hover: Batch size adapts dynamically"
          >
            ?
          </span>
        </div>
        <div className="mt-3 h-24 rounded-lg border border-border bg-bg/60 p-3">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sparkData}>
              <Tooltip
                cursor={{ stroke: "rgba(148, 163, 184, 0.2)" }}
                contentStyle={{
                  background: "rgba(15, 23, 42, 0.9)",
                  border: "1px solid rgba(30, 41, 59, 1)",
                  borderRadius: "8px",
                  fontSize: "12px",
                  color: "#e2e8f0",
                }}
                labelFormatter={() => "Ingestion"}
                formatter={(value: number) => [`${value.toFixed(1)} docs/min`, "Speed"]}
              />
              <Line
                type="monotone"
                dataKey="value"
                stroke="rgb(99 102 241)"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-2 flex items-center justify-between text-xs text-slate-400">
          <span>{ingestionRate.toFixed(1)} docs/min</span>
          <span title="Hover: Batch size adapts dynamically">
            Adaptive batching on
          </span>
        </div>
      </div>

      <div className="rounded-lg border border-border bg-bg/60 p-4">
        <div className="flex items-center justify-between">
          <div className="text-xs uppercase tracking-widest text-slate-400">
            GPU Status
          </div>
          <span
            className={`rounded-full px-2 py-1 text-[10px] uppercase tracking-widest ${
              gpuActive
                ? "bg-emerald-400/20 text-emerald-300"
                : "bg-slate-700 text-slate-400"
            }`}
            title="GPU acceleration toggles embedding throughput"
          >
            {gpuActive ? "Active" : "Idle"}
          </span>
        </div>
        <div className="mt-3 text-sm text-slate-300">
          Vector Count:{" "}
          <span className="font-semibold text-slate-100">
            {vectorCount.toLocaleString()}
          </span>
        </div>
      </div>
    </div>
  );
}

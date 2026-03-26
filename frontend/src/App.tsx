import { useEffect, useState } from "react";

import { api } from "./api";
import { useAppStore } from "./store";
import type { IngestStatus, SystemStats } from "./types";

import HeaderBar from "./components/HeaderBar";
import QueryComposer from "./components/QueryComposer";
import ChatTranscript from "./components/ChatTranscript";
import IngestionPanel from "./components/IngestionPanel";
import SystemStatus from "./components/SystemStatus";
import SourcePanel from "./components/SourcePanel";
import FileInspector from "./components/FileInspector";
import Tabs from "./components/Tabs";
import EvaluationPanel from "./components/EvaluationPanel";

export default function App() {
  const setSystemStats = useAppStore((state) => state.setSystemStats);
  const setIngestStatus = useAppStore((state) => state.setIngestStatus);
  const [activeTab, setActiveTab] = useState<
    "query" | "ingestion" | "evidence" | "inspector" | "system" | "evaluation"
  >("query");

  useEffect(() => {
    const loadStats = async () => {
      try {
        const response = await api.get<SystemStats>("/api/stats");
        setSystemStats(response.data);
      } catch {
        setSystemStats(null);
      }
    };

    const loadIngestStatus = async () => {
      try {
        const response = await api.get<IngestStatus>("/api/ingest/status");
        setIngestStatus(response.data);
      } catch {
        setIngestStatus(null);
      }
    };

    void loadStats();
    void loadIngestStatus();
    const interval = window.setInterval(() => {
      void loadStats();
      void loadIngestStatus();
    }, 8000);

    return () => window.clearInterval(interval);
  }, [setSystemStats, setIngestStatus]);

  return (
    <div className="h-screen overflow-hidden bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-ink">
      <div className="h-full px-4 py-6">
        <div className="mx-auto flex h-full max-w-[1400px] flex-col gap-6">
          <HeaderBar />
          <Tabs active={activeTab} onChange={setActiveTab} />

          <div className="min-h-0 flex-1 overflow-y-auto pr-1">
            {activeTab === "query" && (
              <div className="grid min-h-0 grid-cols-1 gap-6 lg:grid-cols-[1.4fr,1fr]">
                <div className="flex min-h-0 flex-col gap-6">
                  <QueryComposer />
                  <ChatTranscript />
                </div>
                <div className="flex min-h-0 flex-col gap-6">
                  <SourcePanel />
                  <FileInspector />
                </div>
              </div>
            )}

            {activeTab === "ingestion" && (
              <div className="grid min-h-0 grid-cols-1 gap-6 lg:grid-cols-[1.1fr,0.9fr]">
                <IngestionPanel />
                <SystemStatus />
              </div>
            )}

            {activeTab === "evidence" && (
              <div className="grid min-h-0 grid-cols-1 gap-6 lg:grid-cols-[1fr,1fr]">
                <SourcePanel />
                <FileInspector />
              </div>
            )}

            {activeTab === "inspector" && (
              <div className="min-h-0">
                <FileInspector />
              </div>
            )}

            {activeTab === "system" && (
              <div className="min-h-0">
                <SystemStatus />
              </div>
            )}

            {activeTab === "evaluation" && (
              <div className="min-h-0">
                <EvaluationPanel />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

import { create } from "zustand";

import type { ChunkSpan, IngestStatus, Message, SystemStats, ChunkSource } from "./types";

interface AppState {
  chatHistory: Message[];
  isGenerating: boolean;
  systemStats: SystemStats | null;
  ingestStatus: IngestStatus | null;
  sources: ChunkSource[];
  activeFile: string | null;
  fileContent: string;
  fileChunks: ChunkSpan[];
  setChatHistory: (messages: Message[]) => void;
  appendMessage: (message: Message) => void;
  setIsGenerating: (value: boolean) => void;
  setSystemStats: (stats: SystemStats | null) => void;
  setIngestStatus: (status: IngestStatus | null) => void;
  setSources: (sources: ChunkSource[]) => void;
  setActiveFile: (filepath: string | null) => void;
  setFileContent: (content: string) => void;
  setFileChunks: (chunks: ChunkSpan[]) => void;
  resetChat: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  chatHistory: [],
  isGenerating: false,
  systemStats: null,
  ingestStatus: null,
  sources: [],
  activeFile: null,
  fileContent: "",
  fileChunks: [],
  setChatHistory: (messages) => set({ chatHistory: messages }),
  appendMessage: (message) =>
    set((state) => ({ chatHistory: [...state.chatHistory, message] })),
  setIsGenerating: (value) => set({ isGenerating: value }),
  setSystemStats: (stats) => set({ systemStats: stats }),
  setIngestStatus: (status) => set({ ingestStatus: status }),
  setSources: (sources) => set({ sources }),
  setActiveFile: (filepath) => set({ activeFile: filepath }),
  setFileContent: (content) => set({ fileContent: content }),
  setFileChunks: (chunks) => set({ fileChunks: chunks }),
  resetChat: () => set({ chatHistory: [], sources: [] }),
}));

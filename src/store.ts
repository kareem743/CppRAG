import { create } from "zustand";

import type { Message, SystemStats } from "./types";

interface AppState {
  chatHistory: Message[];
  isGenerating: boolean;
  systemStats: SystemStats | null;
  activeFile: string | null;
  setChatHistory: (messages: Message[]) => void;
  appendMessage: (message: Message) => void;
  setIsGenerating: (value: boolean) => void;
  setSystemStats: (stats: SystemStats | null) => void;
  setActiveFile: (filepath: string | null) => void;
  resetChat: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  chatHistory: [],
  isGenerating: false,
  systemStats: null,
  activeFile: null,
  setChatHistory: (messages) => set({ chatHistory: messages }),
  appendMessage: (message) =>
    set((state) => ({ chatHistory: [...state.chatHistory, message] })),
  setIsGenerating: (value) => set({ isGenerating: value }),
  setSystemStats: (stats) => set({ systemStats: stats }),
  setActiveFile: (filepath) => set({ activeFile: filepath }),
  resetChat: () => set({ chatHistory: [] }),
}));

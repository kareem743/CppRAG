import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const slideLeft = {
  hidden: { x: -40, opacity: 0 },
  visible: { x: 0, opacity: 1, transition: { duration: 0.4, ease: "easeOut" } },
};

const slideRight = {
  hidden: { x: 40, opacity: 0 },
  visible: { x: 0, opacity: 1, transition: { duration: 0.4, ease: "easeOut" } },
};

const panelBase =
  "bg-surface text-slate-100 border-border border-r md:border-r-0";

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);

  return (
    <div className="h-screen overflow-hidden bg-bg text-slate-100">
      <div className="grid h-full grid-cols-1 md:grid-cols-[260px,1fr,320px]">
        <motion.aside
          variants={slideLeft}
          initial="hidden"
          animate="visible"
          className={`${panelBase} hidden md:flex md:flex-col md:border-r`}
        >
          <div className="border-border flex items-center gap-2 border-b px-4 py-3">
            <span className="relative flex h-3 w-3">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400/70" />
              <span className="relative inline-flex h-3 w-3 rounded-full bg-emerald-400" />
            </span>
            <span className="text-xs uppercase tracking-widest text-emerald-300">
              Connection Status
            </span>
          </div>
          <div className="flex-1 overflow-y-auto p-4">
            <div className="text-sm text-slate-300">Sidebar content</div>
          </div>
        </motion.aside>

        <main className="relative flex h-full flex-col border-border md:border-x">
          <div className="border-border flex items-center justify-between border-b px-4 py-3">
            <div className="text-sm uppercase tracking-widest text-slate-400">
              Mission Control
            </div>
            <div className="flex items-center gap-2 md:hidden">
              <button
                className="rounded border border-border px-3 py-1 text-xs text-slate-300 hover:text-white"
                onClick={() => setSidebarOpen(true)}
                aria-label="Open sidebar"
              >
                Menu
              </button>
              <button
                className="rounded border border-border px-3 py-1 text-xs text-slate-300 hover:text-white"
                onClick={() => setInspectorOpen(true)}
                aria-label="Open inspector"
              >
                Inspector
              </button>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto p-6">
            <div className="text-slate-300">
              Main chat pane. Messages will render here.
            </div>
          </div>
        </main>

        <motion.aside
          variants={slideRight}
          initial="hidden"
          animate="visible"
          className="hidden md:flex md:flex-col border-border border-l bg-surface"
        >
          <div className="border-border border-b px-4 py-3 text-xs uppercase tracking-widest text-slate-400">
            Inspector
          </div>
          <div className="flex-1 overflow-y-auto p-4 text-sm text-slate-300">
            Inspector content
          </div>
        </motion.aside>
      </div>

      <AnimatePresence>
        {sidebarOpen && (
          <motion.aside
            initial={{ x: "-100%" }}
            animate={{ x: 0 }}
            exit={{ x: "-100%" }}
            transition={{ duration: 0.25, ease: "easeOut" }}
            className="fixed inset-y-0 left-0 z-40 w-72 bg-surface border-border border-r md:hidden"
          >
            <div className="border-border flex items-center justify-between border-b px-4 py-3">
              <div className="flex items-center gap-2">
                <span className="relative flex h-3 w-3">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400/70" />
                  <span className="relative inline-flex h-3 w-3 rounded-full bg-emerald-400" />
                </span>
                <span className="text-xs uppercase tracking-widest text-emerald-300">
                  Connection Status
                </span>
              </div>
              <button
                className="text-xs text-slate-400 hover:text-white"
                onClick={() => setSidebarOpen(false)}
              >
                Close
              </button>
            </div>
            <div className="h-full overflow-y-auto p-4 text-sm text-slate-300">
              Sidebar content
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {inspectorOpen && (
          <motion.aside
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ duration: 0.25, ease: "easeOut" }}
            className="fixed inset-y-0 right-0 z-40 w-72 bg-surface border-border border-l md:hidden"
          >
            <div className="border-border flex items-center justify-between border-b px-4 py-3 text-xs uppercase tracking-widest text-slate-400">
              Inspector
              <button
                className="text-xs text-slate-400 hover:text-white normal-case"
                onClick={() => setInspectorOpen(false)}
              >
                Close
              </button>
            </div>
            <div className="h-full overflow-y-auto p-4 text-sm text-slate-300">
              Inspector content
            </div>
          </motion.aside>
        )}
      </AnimatePresence>
    </div>
  );
}

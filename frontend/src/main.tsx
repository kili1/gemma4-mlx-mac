import React from "react";
import ReactDOM from "react-dom/client";
import {
  Activity,
  Bot,
  Cpu,
  Database,
  Monitor,
  Moon,
  ShieldCheck,
  SlidersHorizontal,
  Sun,
} from "lucide-react";

import { fetchJson } from "./api";
import { AdaptersPanel } from "./components/AdaptersPanel";
import { ChatPanel } from "./components/ChatPanel";
import { ModelsPanel } from "./components/ModelsPanel";
import { SystemPanel } from "./components/SystemPanel";
import { TunePanel } from "./components/TunePanel";
import { formatOptionalBytes } from "./format";
import { loadThemePreference, resolveTheme, saveThemePreference } from "./storage";
import "./styles.css";
import type { InferenceStatus, SystemInfo, ThemeMode, View } from "./types";

const views: Array<{ id: View; label: string; shortcut: string; icon: React.ReactNode }> = [
  { id: "chat", label: "Chat", shortcut: "⌘1", icon: <Bot size={17} /> },
  { id: "models", label: "Models", shortcut: "⌘2", icon: <Database size={17} /> },
  { id: "tune", label: "Fine-tune", shortcut: "⌘3", icon: <SlidersHorizontal size={17} /> },
  { id: "adapters", label: "Adapters", shortcut: "⌘4", icon: <Activity size={17} /> },
  { id: "system", label: "System", shortcut: "⌘5", icon: <Cpu size={17} /> },
];

function App() {
  const [activeView, setActiveView] = React.useState<View>("chat");
  const [chatModelHint, setChatModelHint] = React.useState<string | null>(null);
  const [themeMode, setThemeMode] = React.useState<ThemeMode>(() => loadThemePreference());
  const [systemInfo, setSystemInfo] = React.useState<SystemInfo | null>(null);
  const [inferenceStatus, setInferenceStatus] = React.useState<InferenceStatus | null>(null);

  React.useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => {
      document.documentElement.dataset.theme = resolveTheme(themeMode, media.matches);
    };
    apply();
    saveThemePreference(themeMode);
    media.addEventListener("change", apply);
    return () => media.removeEventListener("change", apply);
  }, [themeMode]);

  React.useEffect(() => {
    Promise.all([
      fetchJson<SystemInfo>("/api/system").catch(() => null),
      fetchJson<InferenceStatus>("/api/inference/status").catch(() => null),
    ]).then(([sys, inf]) => {
      if (sys) {
        setSystemInfo(sys);
      }
      if (inf) {
        setInferenceStatus(inf);
      }
    });
  }, []);

  React.useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (!(event.metaKey || event.ctrlKey) || event.altKey || event.shiftKey) {
        return;
      }
      const idx = Number.parseInt(event.key, 10) - 1;
      if (idx >= 0 && idx < views.length) {
        event.preventDefault();
        setActiveView(views[idx].id);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  function openChatWithModel(modelId?: string) {
    if (modelId) {
      setChatModelHint(modelId);
    }
    setActiveView("chat");
  }

  const totalBytes =
    systemInfo && systemInfo.total_memory_gb > 0 ? systemInfo.total_memory_gb * 1024 ** 3 : null;
  const loadedBytes = inferenceStatus?.model_memory.model_memory_bytes ?? 0;
  const memPercent =
    totalBytes && loadedBytes > 0 ? Math.min(100, Math.round((loadedBytes / totalBytes) * 100)) : 0;

  return (
    <main className="app-frame">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">G4</div>
          <div>
            <p className="eyebrow">Apple Silicon MLX</p>
            <h1>gemma4-mlx-mac</h1>
          </div>
        </div>

        <nav className="tabs" aria-label="Primary">
          {views.map((view) => (
            <button
              type="button"
              key={view.id}
              className={view.id === activeView ? "active" : ""}
              onClick={() => setActiveView(view.id)}
              title={`${view.label} (${view.shortcut})`}
            >
              {view.icon}
              <span>{view.label}</span>
              <kbd>{view.shortcut}</kbd>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <button
            type="button"
            className="sidebar-telemetry"
            onClick={() => setActiveView("system")}
            title="Inspect runtime & unified memory"
          >
            <div className="sidebar-telemetry-row">
              <span>Memory</span>
              <strong>
                {inferenceStatus?.model_memory.loaded
                  ? formatOptionalBytes(loadedBytes, "Loaded")
                  : "Idle"}
                {systemInfo?.total_memory_gb ? ` / ${systemInfo.total_memory_gb} GB` : ""}
              </strong>
            </div>
            <div className="mini-bar">
              <div className="mini-bar-fill" style={{ width: `${memPercent}%` }} />
            </div>
            <div className="sidebar-telemetry-row">
              <span>{inferenceStatus?.available ? "MLX ready" : "MLX setup needed"}</span>
              {inferenceStatus?.active_adapter && (
                <strong>LoRA: {inferenceStatus.active_adapter}</strong>
              )}
            </div>
          </button>

          <div className="sidebar-bottom-row">
            <div className="sidebar-status">
              <ShieldCheck size={15} />
              <span>Local only</span>
            </div>
            <div className="theme-toggle" role="group" aria-label="Color theme">
              <button
                type="button"
                className={themeMode === "light" ? "active" : ""}
                onClick={() => setThemeMode("light")}
                title="Light theme"
                aria-label="Light theme"
              >
                <Sun size={13} />
              </button>
              <button
                type="button"
                className={themeMode === "system" ? "active" : ""}
                onClick={() => setThemeMode("system")}
                title="Match system theme"
                aria-label="System theme"
              >
                <Monitor size={13} />
              </button>
              <button
                type="button"
                className={themeMode === "dark" ? "active" : ""}
                onClick={() => setThemeMode("dark")}
                title="Dark theme"
                aria-label="Dark theme"
              >
                <Moon size={13} />
              </button>
            </div>
          </div>
        </div>
      </aside>

      <section className="workspace">
        {/* Keep ChatPanel mounted so streaming & conversation state survive tab switches. */}
        <div className="view-panel" hidden={activeView !== "chat"}>
          <ChatPanel
            activeModelHint={chatModelHint}
            totalSystemGb={systemInfo?.total_memory_gb}
            onNavigate={setActiveView}
            onStatusChange={setInferenceStatus}
            isVisible={activeView === "chat"}
          />
        </div>
        {activeView === "models" && (
          <ModelsPanel
            totalMemoryGb={systemInfo?.total_memory_gb}
            onChatWithModel={openChatWithModel}
          />
        )}
        {activeView === "tune" && (
          <TunePanel onNavigate={setActiveView} onAdapterActivated={() => openChatWithModel()} />
        )}
        {activeView === "adapters" && (
          <AdaptersPanel onNavigate={setActiveView} onOpenChat={() => openChatWithModel()} />
        )}
        {activeView === "system" && (
          <SystemPanel onStatusChange={setInferenceStatus} onSystemLoaded={setSystemInfo} />
        )}
      </section>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

import React from "react";
import ReactDOM from "react-dom/client";
import { Activity, Bot, Cpu, Database, ShieldCheck, SlidersHorizontal } from "lucide-react";

import { AdaptersPanel } from "./components/AdaptersPanel";
import { ChatPanel } from "./components/ChatPanel";
import { ModelsPanel } from "./components/ModelsPanel";
import { SystemPanel } from "./components/SystemPanel";
import { TunePanel } from "./components/TunePanel";
import "./styles.css";
import type { View } from "./types";

const views: Array<{ id: View; label: string; icon: React.ReactNode }> = [
  { id: "chat", label: "Chat", icon: <Bot size={18} /> },
  { id: "models", label: "Models", icon: <Database size={18} /> },
  { id: "tune", label: "Fine-tune", icon: <SlidersHorizontal size={18} /> },
  { id: "adapters", label: "Adapters", icon: <Activity size={18} /> },
  { id: "system", label: "System", icon: <Cpu size={18} /> },
];

function App() {
  const [activeView, setActiveView] = React.useState<View>("chat");

  return (
    <main className="app-frame">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">G4</div>
          <div>
            <p className="eyebrow">Apple Silicon</p>
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
              title={view.label}
            >
              {view.icon}
              <span>{view.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-status">
          <ShieldCheck size={18} />
          <span>Local only</span>
        </div>
      </aside>

      <section className="workspace">{renderView(activeView)}</section>
    </main>
  );
}

function renderView(view: View) {
  switch (view) {
    case "chat":
      return <ChatPanel />;
    case "models":
      return <ModelsPanel />;
    case "tune":
      return <TunePanel />;
    case "adapters":
      return <AdaptersPanel />;
    case "system":
      return <SystemPanel />;
  }
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

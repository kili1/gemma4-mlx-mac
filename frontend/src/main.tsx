import React from "react";
import ReactDOM from "react-dom/client";
import { Activity, Bot, Cpu, Database, SlidersHorizontal } from "lucide-react";
import "./styles.css";

type View = "chat" | "models" | "tune" | "adapters" | "system";

const views: Array<{ id: View; label: string; icon: React.ReactNode }> = [
  { id: "chat", label: "Chat", icon: <Bot size={18} /> },
  { id: "models", label: "Models", icon: <Database size={18} /> },
  { id: "tune", label: "Fine-tune", icon: <SlidersHorizontal size={18} /> },
  { id: "adapters", label: "Adapters", icon: <Activity size={18} /> },
  { id: "system", label: "System", icon: <Cpu size={18} /> }
];

function App() {
  const [activeView, setActiveView] = React.useState<View>("chat");

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Apple Silicon local AI</p>
          <h1>gemma4-mlx-mac</h1>
        </div>
        <span className="status">Local only</span>
      </header>

      <nav className="tabs" aria-label="Primary">
        {views.map((view) => (
          <button
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

      <section className="workspace">{renderView(activeView)}</section>
    </main>
  );
}

function renderView(view: View) {
  switch (view) {
    case "chat":
      return (
        <>
          <h2>Chat</h2>
          <textarea placeholder="Ask Gemma 4 something..." />
          <button className="primary">Send</button>
        </>
      );
    case "models":
      return (
        <>
          <h2>Models</h2>
          <p>Default: mlx-community/gemma-4-e2b-it-4bit</p>
        </>
      );
    case "tune":
      return (
        <>
          <h2>Fine-tune</h2>
          <input placeholder="./examples/data" />
          <button className="primary">Validate Dataset</button>
        </>
      );
    case "adapters":
      return (
        <>
          <h2>Adapters</h2>
          <p>No adapters active.</p>
        </>
      );
    case "system":
      return (
        <>
          <h2>System</h2>
          <p>Run gemma4-mlx-mac doctor for readiness checks.</p>
        </>
      );
  }
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

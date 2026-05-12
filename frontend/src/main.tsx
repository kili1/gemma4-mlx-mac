import React from "react";
import ReactDOM from "react-dom/client";
import { Activity, Bot, Cpu, Database, Download, SlidersHorizontal } from "lucide-react";
import "./styles.css";

type View = "chat" | "models" | "tune" | "adapters" | "system";
type ModelProfile = {
  id: string;
  label: string;
  recommended_memory_gb: number;
  default: boolean;
  notes: string;
};

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
      return <ModelsPanel />;
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

function ModelsPanel() {
  const [models, setModels] = React.useState<ModelProfile[]>([]);
  const [loadingModel, setLoadingModel] = React.useState<string | null>(null);
  const [message, setMessage] = React.useState<string>("Loading model profiles...");

  React.useEffect(() => {
    fetch("/api/models")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Could not load models.");
        }
        return response.json();
      })
      .then((data: { models: ModelProfile[] }) => {
        setModels(data.models);
        setMessage("");
      })
      .catch((error: unknown) => {
        setMessage(error instanceof Error ? error.message : "Could not load models.");
      });
  }, []);

  async function downloadModel(model: string) {
    setLoadingModel(model);
    setMessage(`Downloading ${model}...`);
    try {
      const response = await fetch("/api/models/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model })
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail ?? "Download failed.");
      }
      setMessage(`Downloaded to ${data.path}`);
    } catch (error: unknown) {
      setMessage(error instanceof Error ? error.message : "Download failed.");
    } finally {
      setLoadingModel(null);
    }
  }

  return (
    <>
      <h2>Models</h2>
      {message && <p className="notice">{message}</p>}
      <div className="model-list">
        {models.map((model) => (
          <div className="model-row" key={model.id}>
            <div>
              <strong>{model.label}</strong>
              <p>{model.id}</p>
              <span>{model.recommended_memory_gb} GB recommended</span>
              {model.default && <span>Default</span>}
            </div>
            <button
              className="secondary"
              disabled={loadingModel !== null}
              onClick={() => downloadModel(model.id)}
              title={`Download ${model.label}`}
            >
              <Download size={18} />
              <span>{loadingModel === model.id ? "Downloading" : "Download"}</span>
            </button>
          </div>
        ))}
      </div>
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

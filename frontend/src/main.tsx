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
type DownloadJob = {
  id: string;
  model: string;
  status: "queued" | "running" | "succeeded" | "failed";
  bytes_downloaded: number;
  bytes_total: number | null;
  files_downloaded: number;
  files_total: number | null;
  percent: number | null;
  path: string | null;
  error: string | null;
  message: string;
};

const defaultModelId = "mlx-community/gemma-4-e2b-it-4bit";

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
      return <ChatPanel />;
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

function ChatPanel() {
  const [models, setModels] = React.useState<ModelProfile[]>([]);
  const [selectedModel, setSelectedModel] = React.useState(defaultModelId);
  const [prompt, setPrompt] = React.useState("");
  const [message, setMessage] = React.useState("");
  const [reply, setReply] = React.useState("");
  const [isSending, setIsSending] = React.useState(false);

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
        const defaultModel = data.models.find((model) => model.default) ?? data.models[0];
        if (defaultModel) {
          setSelectedModel(defaultModel.id);
        }
      })
      .catch((error: unknown) => {
        setMessage(error instanceof Error ? error.message : "Could not load models.");
      });
  }, []);

  async function sendPrompt() {
    const content = prompt.trim();
    if (!content) {
      setMessage("Enter a prompt first.");
      return;
    }

    setIsSending(true);
    setMessage("");
    setReply("");
    try {
      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: selectedModel,
          messages: [{ role: "user", content }]
        })
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error?.message ?? data.detail ?? "Chat request failed.");
      }
      setReply(data.choices?.[0]?.message?.content ?? JSON.stringify(data, null, 2));
    } catch (error: unknown) {
      setMessage(error instanceof Error ? error.message : "Chat request failed.");
    } finally {
      setIsSending(false);
    }
  }

  return (
    <>
      <div className="chat-header">
        <h2>Chat</h2>
        <label className="field">
          <span>Model</span>
          <select
            value={selectedModel}
            onChange={(event) => setSelectedModel(event.target.value)}
            disabled={models.length === 0 || isSending}
          >
            {models.length === 0 ? (
              <option value={selectedModel}>{selectedModel}</option>
            ) : (
              models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label}
                </option>
              ))
            )}
          </select>
        </label>
      </div>
      {message && <p className="notice">{message}</p>}
      <textarea
        value={prompt}
        onChange={(event) => setPrompt(event.target.value)}
        placeholder="Ask Gemma 4 something..."
      />
      <div className="chat-actions">
        <span>{selectedModel}</span>
        <button className="primary" disabled={isSending} onClick={sendPrompt}>
          {isSending ? "Sending" : "Send"}
        </button>
      </div>
      {reply && (
        <section className="response-panel" aria-label="Assistant response">
          <pre>{reply}</pre>
        </section>
      )}
    </>
  );
}

function ModelsPanel() {
  const [models, setModels] = React.useState<ModelProfile[]>([]);
  const [loadingModel, setLoadingModel] = React.useState<string | null>(null);
  const [downloads, setDownloads] = React.useState<Record<string, DownloadJob>>({});
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
    setMessage("");
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
      setDownloads((current) => ({ ...current, [model]: data }));
      pollDownload(model, data.id);
    } catch (error: unknown) {
      setMessage(error instanceof Error ? error.message : "Download failed.");
      setLoadingModel(null);
    }
  }

  async function pollDownload(model: string, jobId: string) {
    try {
      const response = await fetch(`/api/models/download/${jobId}`);
      const job: DownloadJob = await response.json();
      if (!response.ok) {
        throw new Error(job.error ?? "Could not read download progress.");
      }
      setDownloads((current) => ({ ...current, [model]: job }));
      if (job.status === "queued" || job.status === "running") {
        window.setTimeout(() => pollDownload(model, jobId), 800);
        return;
      }
      setLoadingModel(null);
      if (job.status === "failed") {
        setMessage(job.error ?? "Download failed.");
      }
    } catch (error: unknown) {
      setLoadingModel(null);
      setMessage(error instanceof Error ? error.message : "Could not read download progress.");
    }
  }

  return (
    <>
      <h2>Models</h2>
      {message && <p className="notice">{message}</p>}
      <div className="model-list">
        {models.map((model) => (
          <div className="model-card" key={model.id}>
            <div className="model-row">
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
                <span>{loadingModel === model.id ? "Starting" : "Download"}</span>
              </button>
            </div>
            {downloads[model.id] && <DownloadProgress job={downloads[model.id]} />}
          </div>
        ))}
      </div>
    </>
  );
}

function DownloadProgress({ job }: { job: DownloadJob }) {
  const percent = job.percent ?? 0;
  const progressLabel = job.percent === null ? "Preparing" : `${job.percent}%`;
  const terminalMessage = job.status === "succeeded" && job.path ? `Saved to ${job.path}` : job.error;

  return (
    <div className="download-progress">
      <div className="progress-meta">
        <span>{job.status}</span>
        <span>{progressLabel}</span>
      </div>
      <div className="progress-track" aria-label={`Download progress for ${job.model}`}>
        <div className="progress-fill" style={{ width: `${percent}%` }} />
      </div>
      <p>{terminalMessage || progressDetails(job)}</p>
    </div>
  );
}

function progressDetails(job: DownloadJob) {
  if (job.bytes_total && job.bytes_total >= job.bytes_downloaded) {
    return `${formatBytes(job.bytes_downloaded)} / ${formatBytes(job.bytes_total)}`;
  }
  if (job.files_total) {
    return `${job.files_downloaded} / ${job.files_total} files`;
  }
  return job.message;
}

function formatBytes(bytes: number) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 ? 1 : 2)} ${units[unitIndex]}`;
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

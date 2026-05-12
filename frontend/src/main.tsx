import React from "react";
import ReactDOM from "react-dom/client";
import {
  Activity,
  Bot,
  Cpu,
  Database,
  Download,
  SlidersHorizontal,
  Wrench
} from "lucide-react";
import "./styles.css";

type View = "chat" | "models" | "tune" | "adapters" | "system";
type ModelProfile = {
  id: string;
  label: string;
  recommended_memory_gb: number;
  default: boolean;
  downloaded: boolean;
  local_path: string | null;
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
type MlxInstallJob = {
  id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  command: string[];
  cwd: string | null;
  output: string[];
  returncode: number | null;
  error: string | null;
  message: string;
};
type InferenceStatus = {
  available: boolean;
  installing: boolean;
  install_job_id: string | null;
  command: string[];
  cwd: string | null;
  message: string;
  error: string | null;
  job: MlxInstallJob | null;
};
type ModelSource = "downloaded" | "path";

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
  const [selectedModel, setSelectedModel] = React.useState("");
  const [modelSource, setModelSource] = React.useState<ModelSource>("downloaded");
  const [modelPath, setModelPath] = React.useState("");
  const [prompt, setPrompt] = React.useState("");
  const [message, setMessage] = React.useState("");
  const [reply, setReply] = React.useState("");
  const [isSending, setIsSending] = React.useState(false);
  const [inferenceStatus, setInferenceStatus] = React.useState<InferenceStatus | null>(null);
  const [installJob, setInstallJob] = React.useState<MlxInstallJob | null>(null);
  const downloadedModels = models.filter((model) => model.downloaded);
  const selectedDownloadedModel =
    downloadedModels.find((model) => model.id === selectedModel) ?? null;
  const activeModel =
    modelSource === "path" ? modelPath.trim() : selectedDownloadedModel?.id ?? "";
  const activeModelLabel =
    modelSource === "path" ? modelPath.trim() || "No model path selected" : activeModel;
  const inferenceReady = inferenceStatus?.available === true;
  const isInstalling =
    installJob?.status === "queued" ||
    installJob?.status === "running" ||
    inferenceStatus?.installing;

  React.useEffect(() => {
    loadModels();
    loadInferenceStatus();
  }, []);

  async function loadModels() {
    try {
      const response = await fetch("/api/models");
      if (!response.ok) {
        throw new Error("Could not load models.");
      }
      const data: { models: ModelProfile[] } = await response.json();
      setModels(data.models);
      const availableModels = data.models.filter((model) => model.downloaded);
      const defaultModel = availableModels.find((model) => model.default) ?? availableModels[0];
      if (defaultModel) {
        setSelectedModel(defaultModel.id);
        if (defaultModel.local_path) {
          setModelPath(defaultModel.local_path);
        }
      } else {
        setSelectedModel("");
        setMessage("Download a model in the Models tab before chatting.");
      }
    } catch (error: unknown) {
      setMessage(error instanceof Error ? error.message : "Could not load models.");
    }
  }

  async function loadInferenceStatus() {
    try {
      const response = await fetch("/api/inference/status");
      if (!response.ok) {
        throw new Error("Could not read MLX status.");
      }
      const status: InferenceStatus = await response.json();
      setInferenceStatus(status);
      if (status.job) {
        setInstallJob(status.job);
      }
      const installJobId = status.install_job_id;
      if (status.installing && installJobId) {
        window.setTimeout(() => pollInstall(installJobId), 1000);
      }
    } catch (error: unknown) {
      setMessage(error instanceof Error ? error.message : "Could not read MLX status.");
    }
  }

  async function startInstall() {
    setMessage("");
    try {
      const response = await fetch("/api/inference/install", { method: "POST" });
      const job: MlxInstallJob = await response.json();
      if (!response.ok) {
        throw new Error(job.error ?? "Could not start MLX install.");
      }
      setInstallJob(job);
      setInferenceStatus((current) =>
        current ? { ...current, installing: job.status !== "succeeded", job } : current
      );
      if (job.status === "queued" || job.status === "running") {
        pollInstall(job.id);
      } else {
        loadInferenceStatus();
      }
    } catch (error: unknown) {
      setMessage(error instanceof Error ? error.message : "Could not start MLX install.");
    }
  }

  async function pollInstall(jobId: string) {
    try {
      const response = await fetch(`/api/inference/install/${jobId}`);
      const job: MlxInstallJob = await response.json();
      if (!response.ok) {
        throw new Error(job.error ?? "Could not read MLX install progress.");
      }
      setInstallJob(job);
      if (job.status === "queued" || job.status === "running") {
        window.setTimeout(() => pollInstall(jobId), 1000);
        return;
      }
      loadInferenceStatus();
      if (job.status === "failed") {
        setMessage(job.error ?? "MLX install failed.");
      }
    } catch (error: unknown) {
      setMessage(error instanceof Error ? error.message : "Could not read MLX install progress.");
    }
  }

  function chooseModelSource(source: ModelSource) {
    setModelSource(source);
    if (source === "path" && !modelPath && selectedDownloadedModel?.local_path) {
      setModelPath(selectedDownloadedModel.local_path);
    }
    setMessage("");
  }

  function chooseDownloadedModel(modelId: string) {
    setSelectedModel(modelId);
    const model = downloadedModels.find((profile) => profile.id === modelId);
    if (model?.local_path && modelSource === "path") {
      setModelPath(model.local_path);
    }
  }

  async function sendPrompt() {
    const content = prompt.trim();
    if (!inferenceReady) {
      setMessage("Install MLX inference before chatting.");
      return;
    }
    if (!activeModel) {
      if (modelSource === "path") {
        setMessage("Enter a local model path first.");
        return;
      }
      setMessage("Download a model in the Models tab before chatting.");
      return;
    }
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
          model: activeModel,
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
        <div className="model-picker">
          <div className="source-toggle" aria-label="Model source">
            <button
              className={modelSource === "downloaded" ? "active" : ""}
              onClick={() => chooseModelSource("downloaded")}
              type="button"
            >
              Downloaded
            </button>
            <button
              className={modelSource === "path" ? "active" : ""}
              onClick={() => chooseModelSource("path")}
              type="button"
            >
              Path
            </button>
          </div>
          {modelSource === "downloaded" ? (
            <label className="field">
              <span>Model</span>
              <select
                value={selectedModel}
                onChange={(event) => chooseDownloadedModel(event.target.value)}
                disabled={downloadedModels.length === 0 || isSending}
              >
                {downloadedModels.length === 0 ? (
                  <option value="">No downloaded models</option>
                ) : (
                  downloadedModels.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.label}
                    </option>
                  ))
                )}
              </select>
              {selectedDownloadedModel?.local_path && <small>{selectedDownloadedModel.local_path}</small>}
            </label>
          ) : (
            <label className="field">
              <span>Model path</span>
              <input
                value={modelPath}
                onChange={(event) => setModelPath(event.target.value)}
                disabled={isSending}
                list="downloaded-model-paths"
                placeholder="/Users/me/.cache/huggingface/.../snapshots/..."
              />
              <datalist id="downloaded-model-paths">
                {downloadedModels
                  .filter((model) => model.local_path)
                  .map((model) => (
                    <option key={model.id} value={model.local_path ?? ""}>
                      {model.label}
                    </option>
                  ))}
              </datalist>
            </label>
          )}
        </div>
      </div>
      {inferenceStatus && !inferenceReady && (
        <InferenceSetup
          status={inferenceStatus}
          job={installJob}
          isInstalling={Boolean(isInstalling)}
          onInstall={startInstall}
        />
      )}
      {message && <p className="notice">{message}</p>}
      <textarea
        value={prompt}
        onChange={(event) => setPrompt(event.target.value)}
        placeholder="Ask Gemma 4 something..."
      />
      <div className="chat-actions">
        <span>{activeModelLabel}</span>
        <button
          className="primary"
          disabled={isSending || !activeModel || !inferenceReady}
          onClick={sendPrompt}
        >
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

function InferenceSetup({
  status,
  job,
  isInstalling,
  onInstall
}: {
  status: InferenceStatus;
  job: MlxInstallJob | null;
  isInstalling: boolean;
  onInstall: () => void;
}) {
  const output = job?.output.slice(-8) ?? [];

  return (
    <section className="setup-panel" aria-label="MLX inference setup">
      <div>
        <strong>MLX inference is not installed</strong>
        <p>{status.message}</p>
        <code>{formatCommand(status.command)}</code>
      </div>
      <button className="secondary" disabled={isInstalling} onClick={onInstall}>
        <Wrench size={18} />
        <span>{isInstalling ? "Installing" : "Install MLX"}</span>
      </button>
      {job && (
        <div className="setup-progress">
          <div className="progress-meta">
            <span>{job.status}</span>
            <span>{job.returncode === null ? "Running locally" : `Exit ${job.returncode}`}</span>
          </div>
          <p>{job.error ?? job.message}</p>
          {output.length > 0 && <pre>{output.join("\n")}</pre>}
        </div>
      )}
    </section>
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
      if (job.status === "succeeded") {
        setModels((current) =>
          current.map((profile) =>
            profile.id === model
              ? { ...profile, downloaded: true, local_path: job.path }
              : profile
          )
        );
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
                {model.downloaded && <span>Downloaded</span>}
              </div>
              <button
                className="secondary"
                disabled={loadingModel !== null}
                onClick={() => downloadModel(model.id)}
                title={`Download ${model.label}`}
              >
                <Download size={18} />
                <span>
                  {loadingModel === model.id
                    ? "Starting"
                    : model.downloaded
                      ? "Download again"
                      : "Download"}
                </span>
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

function formatCommand(command: string[]) {
  return command.map((part) => (/\s/.test(part) ? `"${part}"` : part)).join(" ");
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

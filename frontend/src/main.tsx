import React from "react";
import ReactDOM from "react-dom/client";
import {
  Activity,
  AlertCircle,
  Bot,
  CheckCircle2,
  Circle,
  Cpu,
  Database,
  Download,
  FileText,
  FolderOpen,
  HardDrive,
  MessageSquareText,
  Send,
  ShieldCheck,
  SlidersHorizontal,
  Terminal,
  Wrench
} from "lucide-react";
import "./styles.css";

type View = "chat" | "models" | "tune" | "adapters" | "system";
type ModelProfile = {
  id: string;
  label: string;
  family: string;
  size: string;
  quantization: string;
  recommended_memory_gb: number;
  modality: string;
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
type SystemInfo = {
  os_name: string;
  os_version: string;
  machine: string;
  python_version: string;
  total_memory_gb: number;
  is_macos: boolean;
  is_apple_silicon: boolean;
  mlx_ready: boolean;
  recommendations: string[];
};
type TuneJob = {
  id: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  request: {
    model: string;
    data_path: string;
    adapter_name: string;
    iters: number;
    batch_size: number;
    learning_rate: number;
    fine_tune_type: "lora" | "dora" | "full";
  };
  dataset: {
    data_path: string;
    files: Array<{
      path: string;
      format: "chat" | "completion" | "text";
      examples: number;
    }>;
  };
  message: string;
};
type SyntheticFormat = "chat" | "completion" | "text";
type SyntheticDatasetResult = {
  data_path: string;
  train_path: string;
  format: SyntheticFormat;
  examples: number;
  report: TuneJob["dataset"];
};
type AdapterInfo = {
  id: string;
  path: string;
  active: boolean;
};
type GenerationStats = {
  completionTokens: number;
  tokensPerSecond: number;
  elapsedSeconds: number;
};
type ChatStreamPayload = {
  choices?: Array<{
    delta?: { content?: string };
    finish_reason?: string | null;
  }>;
  metrics?: {
    completion_tokens?: number;
    tokens_per_second?: number;
    elapsed_seconds?: number;
  } | null;
  error?: {
    type: string;
    message: string;
  };
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

function PageHeader({
  title,
  eyebrow,
  children
}: {
  title: string;
  eyebrow: string;
  children?: React.ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h2>{title}</h2>
      </div>
      {children && <div className="page-actions">{children}</div>}
    </header>
  );
}

function StatusBadge({
  tone = "neutral",
  children
}: {
  tone?: "neutral" | "good" | "warn" | "bad";
  children: React.ReactNode;
}) {
  const Icon =
    tone === "good" ? CheckCircle2 : tone === "bad" || tone === "warn" ? AlertCircle : Circle;
  return (
    <span className={`status-badge ${tone}`}>
      <Icon size={14} />
      {children}
    </span>
  );
}

function ChatPanel() {
  const [models, setModels] = React.useState<ModelProfile[]>([]);
  const [selectedModel, setSelectedModel] = React.useState("");
  const [modelSource, setModelSource] = React.useState<ModelSource>("downloaded");
  const [modelPath, setModelPath] = React.useState("");
  const [prompt, setPrompt] = React.useState("");
  const [lastPrompt, setLastPrompt] = React.useState("");
  const [message, setMessage] = React.useState("");
  const [reply, setReply] = React.useState("");
  const [generationStats, setGenerationStats] = React.useState<GenerationStats | null>(null);
  const [showThinking, setShowThinking] = React.useState(false);
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
    setLastPrompt(content);
    setGenerationStats({ completionTokens: 0, tokensPerSecond: 0, elapsedSeconds: 0 });
    try {
      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: activeModel,
          messages: [{ role: "user", content }],
          show_thinking: showThinking,
          stream: true
        })
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error?.message ?? data.detail ?? "Chat request failed.");
      }

      if (!response.body) {
        const data = await response.json();
        setReply(data.choices?.[0]?.message?.content ?? JSON.stringify(data, null, 2));
        return;
      }

      let generatedText = "";
      let fallbackTokens = 0;
      const startedAt = performance.now();
      await readChatStream(response, (payload) => {
        if (payload.error) {
          throw new Error(payload.error.message);
        }
        const delta = payload.choices?.[0]?.delta?.content ?? "";
        if (delta) {
          generatedText += delta;
          fallbackTokens += 1;
          setReply(generatedText);
        }
        const elapsedSeconds = Math.max((performance.now() - startedAt) / 1000, 0.001);
        const completionTokens = payload.metrics?.completion_tokens ?? fallbackTokens;
        setGenerationStats({
          completionTokens,
          tokensPerSecond: payload.metrics?.tokens_per_second ?? completionTokens / elapsedSeconds,
          elapsedSeconds: payload.metrics?.elapsed_seconds ?? elapsedSeconds
        });
      });
    } catch (error: unknown) {
      setMessage(error instanceof Error ? error.message : "Chat request failed.");
    } finally {
      setIsSending(false);
    }
  }

  return (
    <>
      <PageHeader title="Chat" eyebrow="Local inference">
        <StatusBadge tone={inferenceReady ? "good" : "warn"}>
          {inferenceReady ? "MLX ready" : "MLX setup"}
        </StatusBadge>
      </PageHeader>

      <div className="chat-layout">
        <section className="conversation-panel" aria-label="Conversation">
          {message && <p className="notice">{message}</p>}
          <div className="chat-thread">
            {!lastPrompt && !reply && (
              <div className="empty-state">
                <MessageSquareText size={28} />
                <strong>Ready</strong>
              </div>
            )}
            {lastPrompt && (
              <article className="message user-message">
                <span>User</span>
                <p>{lastPrompt}</p>
              </article>
            )}
            {(reply || isSending) && (
              <article className="message assistant-message">
                <div className="message-meta">
                  <span>Gemma</span>
                  {generationStats && (
                    <div className="generation-stats">
                      <span>{generationStats.completionTokens} tokens</span>
                      <span>{formatTokensPerSecond(generationStats.tokensPerSecond)} tok/s</span>
                    </div>
                  )}
                </div>
                {reply ? <pre>{reply}</pre> : <p>Waiting for first token</p>}
              </article>
            )}
          </div>

          <div className="composer">
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="Ask Gemma 4 something..."
            />
            <div className="chat-actions">
              <span>{activeModelLabel}</span>
              <button
                className="primary icon-button"
                disabled={isSending || !activeModel || !inferenceReady}
                onClick={sendPrompt}
                title="Send"
              >
                <Send size={18} />
                <span>{isSending ? "Sending" : "Send"}</span>
              </button>
            </div>
          </div>
        </section>

        <aside className="control-panel" aria-label="Chat controls">
          <div className="panel-block">
            <div className="block-heading">
              <HardDrive size={18} />
              <strong>Model</strong>
            </div>
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
                  <span>Profile</span>
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
                  {selectedDownloadedModel?.local_path && (
                    <small>{selectedDownloadedModel.local_path}</small>
                  )}
                </label>
              ) : (
                <label className="field">
                  <span>Path</span>
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
          <div className="panel-block">
            <div className="block-heading">
              <Activity size={18} />
              <strong>Thinking</strong>
            </div>
            <label className="switch-row">
              <span className="switch-copy">
                <strong>Show summary</strong>
                <small>Ask for a brief visible reasoning summary before the answer.</small>
              </span>
              <input
                aria-label="Show thinking summary"
                checked={showThinking}
                className="switch-control"
                disabled={isSending}
                onChange={(event) => setShowThinking(event.target.checked)}
                type="checkbox"
              />
            </label>
          </div>
          {inferenceStatus && !inferenceReady && (
            <InferenceSetup
              status={inferenceStatus}
              job={installJob}
              isInstalling={Boolean(isInstalling)}
              onInstall={startInstall}
            />
          )}
        </aside>
      </div>
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
  const [downloadFolder, setDownloadFolder] = React.useState("");
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
    const targetFolder = downloadFolder.trim();
    try {
      const response = await fetch("/api/models/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          local_dir: targetFolder || null
        })
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
      <PageHeader title="Models" eyebrow="Hugging Face snapshots">
        <StatusBadge tone={models.some((model) => model.downloaded) ? "good" : "neutral"}>
          {models.filter((model) => model.downloaded).length} downloaded
        </StatusBadge>
      </PageHeader>

      <section className="download-settings panel" aria-label="Model download location">
        <label className="field">
          <span>Download folder</span>
          <div className="path-input">
            <FolderOpen size={18} />
            <input
              value={downloadFolder}
              onChange={(event) => setDownloadFolder(event.target.value)}
              disabled={loadingModel !== null}
              placeholder="~/Models/gemma4"
            />
          </div>
          <small>{downloadFolder.trim() || "Default Hugging Face cache"}</small>
        </label>
      </section>
      {message && <p className="notice">{message}</p>}
      <div className="model-list">
        {models.map((model) => (
          <div className="model-card" key={model.id}>
            <div className="model-row">
              <div>
                <div className="model-title">
                  <strong>{model.label}</strong>
                  <div className="badges">
                    {model.default && <span>Default</span>}
                    {model.downloaded && <span className="good">Downloaded</span>}
                  </div>
                </div>
                <p>{model.id}</p>
                <div className="model-meta">
                  <span>{model.recommended_memory_gb} GB</span>
                  <span>{model.quantization}</span>
                  <span>{model.modality}</span>
                </div>
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

function TunePanel() {
  const [models, setModels] = React.useState<ModelProfile[]>([]);
  const [model, setModel] = React.useState("");
  const [dataPath, setDataPath] = React.useState("examples/data");
  const [adapterName, setAdapterName] = React.useState("demo-adapter");
  const [iters, setIters] = React.useState(100);
  const [batchSize, setBatchSize] = React.useState(1);
  const [syntheticTopic, setSyntheticTopic] = React.useState("local Apple Silicon AI");
  const [syntheticExamples, setSyntheticExamples] = React.useState(24);
  const [syntheticFormat, setSyntheticFormat] = React.useState<SyntheticFormat>("chat");
  const [syntheticOutput, setSyntheticOutput] = React.useState("examples/synthetic");
  const [message, setMessage] = React.useState("");
  const [job, setJob] = React.useState<TuneJob | null>(null);
  const [syntheticResult, setSyntheticResult] = React.useState<SyntheticDatasetResult | null>(null);
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [isGeneratingData, setIsGeneratingData] = React.useState(false);

  React.useEffect(() => {
    fetch("/api/models")
      .then((response) => response.json())
      .then((data: { models: ModelProfile[] }) => {
        setModels(data.models);
        setModel(data.models.find((profile) => profile.default)?.id ?? data.models[0]?.id ?? "");
      })
      .catch(() => setMessage("Could not load model profiles."));
  }, []);

  async function startTune() {
    setIsSubmitting(true);
    setMessage("");
    setJob(null);
    try {
      const response = await fetch("/api/tunes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          data_path: dataPath,
          adapter_name: adapterName,
          iters,
          batch_size: batchSize
        })
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail ?? "Fine-tune job failed.");
      }
      setJob(data);
    } catch (error: unknown) {
      setMessage(error instanceof Error ? error.message : "Fine-tune job failed.");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function generateSyntheticData() {
    setIsGeneratingData(true);
    setMessage("");
    setSyntheticResult(null);
    try {
      const response = await fetch("/api/datasets/synthetic", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: syntheticTopic,
          examples: syntheticExamples,
          output_dir: syntheticOutput,
          format: syntheticFormat,
          system_prompt: "You are a concise, helpful assistant."
        })
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail ?? "Synthetic dataset generation failed.");
      }
      setSyntheticResult(data);
      setDataPath(data.data_path);
      setMessage(`Synthetic dataset created at ${data.train_path}.`);
    } catch (error: unknown) {
      setMessage(
        error instanceof Error ? error.message : "Synthetic dataset generation failed.",
      );
    } finally {
      setIsGeneratingData(false);
    }
  }

  return (
    <>
      <PageHeader title="Fine-tune" eyebrow="LoRA jobs">
        <StatusBadge tone={job ? "good" : "neutral"}>{job?.status ?? "idle"}</StatusBadge>
      </PageHeader>
      {message && <p className="notice">{message}</p>}
      <div className="two-column">
        <div className="panel-stack">
          <section className="panel form-panel">
            <div className="block-heading">
              <FileText size={18} />
              <strong>Synthetic dataset</strong>
            </div>
            <label className="field">
              <span>Topic</span>
              <input
                value={syntheticTopic}
                onChange={(event) => setSyntheticTopic(event.target.value)}
              />
            </label>
            <label className="field">
              <span>Output folder</span>
              <input
                value={syntheticOutput}
                onChange={(event) => setSyntheticOutput(event.target.value)}
              />
            </label>
            <div className="form-grid">
              <label className="field">
                <span>Format</span>
                <select
                  value={syntheticFormat}
                  onChange={(event) => setSyntheticFormat(event.target.value as SyntheticFormat)}
                >
                  <option value="chat">Chat messages</option>
                  <option value="completion">Prompt completion</option>
                  <option value="text">Plain text</option>
                </select>
              </label>
              <label className="field">
                <span>Examples</span>
                <input
                  min={1}
                  max={500}
                  type="number"
                  value={syntheticExamples}
                  onChange={(event) => setSyntheticExamples(Number(event.target.value))}
                />
              </label>
            </div>
            <button
              className="secondary"
              disabled={isGeneratingData || !syntheticTopic.trim()}
              onClick={generateSyntheticData}
            >
              {isGeneratingData ? "Generating" : "Generate Dataset"}
            </button>
          </section>

          <section className="panel form-panel">
            <div className="block-heading">
              <SlidersHorizontal size={18} />
              <strong>Fine-tune job</strong>
            </div>
            <label className="field">
              <span>Model</span>
              <select value={model} onChange={(event) => setModel(event.target.value)}>
                {models.map((profile) => (
                  <option key={profile.id} value={profile.id}>
                    {profile.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Dataset</span>
              <input value={dataPath} onChange={(event) => setDataPath(event.target.value)} />
            </label>
            <label className="field">
              <span>Adapter</span>
              <input value={adapterName} onChange={(event) => setAdapterName(event.target.value)} />
            </label>
            <div className="form-grid">
              <label className="field">
                <span>Iterations</span>
                <input
                  min={1}
                  type="number"
                  value={iters}
                  onChange={(event) => setIters(Number(event.target.value))}
                />
              </label>
              <label className="field">
                <span>Batch</span>
                <input
                  min={1}
                  type="number"
                  value={batchSize}
                  onChange={(event) => setBatchSize(Number(event.target.value))}
                />
              </label>
            </div>
            <button className="primary" disabled={isSubmitting || !model} onClick={startTune}>
              {isSubmitting ? "Validating" : "Create Job"}
            </button>
          </section>
        </div>

        <section className="panel result-panel">
          {syntheticResult && (
            <div className="result-block">
              <div className="block-heading">
                <FileText size={18} />
                <strong>Synthetic data</strong>
              </div>
              <dl className="details-list">
                <div>
                  <dt>Examples</dt>
                  <dd>{syntheticResult.examples}</dd>
                </div>
                <div>
                  <dt>Format</dt>
                  <dd>{syntheticResult.format}</dd>
                </div>
                <div>
                  <dt>Path</dt>
                  <dd>{syntheticResult.train_path}</dd>
                </div>
              </dl>
            </div>
          )}
          {job ? (
            <>
              <div className="block-heading">
                <Terminal size={18} />
                <strong>{job.id}</strong>
              </div>
              <dl className="details-list">
                <div>
                  <dt>Examples</dt>
                  <dd>{countDatasetExamples(job.dataset)}</dd>
                </div>
                <div>
                  <dt>Status</dt>
                  <dd>{job.status}</dd>
                </div>
                <div>
                  <dt>Adapter</dt>
                  <dd>{job.request.adapter_name}</dd>
                </div>
              </dl>
              <p>{job.message}</p>
            </>
          ) : (
            <div className="empty-state">
              <SlidersHorizontal size={28} />
              <strong>No job queued</strong>
            </div>
          )}
        </section>
      </div>
    </>
  );
}

function AdaptersPanel() {
  const [adapters, setAdapters] = React.useState<AdapterInfo[]>([]);
  const [message, setMessage] = React.useState("");

  React.useEffect(() => {
    fetch("/api/adapters")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Could not load adapters.");
        }
        return response.json();
      })
      .then((data: { adapters: AdapterInfo[] }) => setAdapters(data.adapters))
      .catch((error: unknown) => {
        setMessage(error instanceof Error ? error.message : "Could not load adapters.");
      });
  }, []);

  return (
    <>
      <PageHeader title="Adapters" eyebrow="Local LoRA outputs">
        <StatusBadge tone={adapters.some((adapter) => adapter.active) ? "good" : "neutral"}>
          {adapters.length} found
        </StatusBadge>
      </PageHeader>
      {message && <p className="notice">{message}</p>}
      <div className="model-list">
        {adapters.length === 0 ? (
          <div className="panel empty-state">
            <Activity size={28} />
            <strong>No adapters</strong>
          </div>
        ) : (
          adapters.map((adapter) => (
            <div className="model-card" key={adapter.id}>
              <div className="model-row">
                <div>
                  <div className="model-title">
                    <strong>{adapter.id}</strong>
                    {adapter.active && <div className="badges"><span className="good">Active</span></div>}
                  </div>
                  <p>{adapter.path}</p>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </>
  );
}

function SystemPanel() {
  const [system, setSystem] = React.useState<SystemInfo | null>(null);
  const [inference, setInference] = React.useState<InferenceStatus | null>(null);
  const [message, setMessage] = React.useState("");

  React.useEffect(() => {
    Promise.all([fetch("/api/system"), fetch("/api/inference/status")])
      .then(async ([systemResponse, inferenceResponse]) => {
        if (!systemResponse.ok || !inferenceResponse.ok) {
          throw new Error("Could not load system status.");
        }
        setSystem(await systemResponse.json());
        setInference(await inferenceResponse.json());
      })
      .catch((error: unknown) => {
        setMessage(error instanceof Error ? error.message : "Could not load system status.");
      });
  }, []);

  return (
    <>
      <PageHeader title="System" eyebrow="Runtime status">
        <StatusBadge tone={inference?.available ? "good" : "warn"}>
          {inference?.available ? "inference ready" : "inference setup"}
        </StatusBadge>
      </PageHeader>
      {message && <p className="notice">{message}</p>}
      <div className="stats-grid">
        <MetricCard label="Machine" value={system?.machine ?? "Loading"} />
        <MetricCard label="Memory" value={system ? `${system.total_memory_gb} GB` : "Loading"} />
        <MetricCard label="Python" value={system?.python_version ?? "Loading"} />
        <MetricCard label="MLX" value={inference?.available ? "Installed" : "Missing"} />
      </div>
      {system && system.recommendations.length > 0 && (
        <section className="panel recommendation-list">
          {system.recommendations.map((recommendation) => (
            <p key={recommendation}>{recommendation}</p>
          ))}
        </section>
      )}
    </>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
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

function countDatasetExamples(dataset: TuneJob["dataset"]) {
  return dataset.files.reduce((total, file) => total + file.examples, 0);
}

async function readChatStream(
  response: Response,
  onPayload: (payload: ChatStreamPayload) => void,
) {
  if (!response.body) {
    throw new Error("Chat response did not include a stream.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const event of events) {
      handleStreamEvent(event, onPayload);
    }
  }

  buffer += decoder.decode();
  if (buffer.trim()) {
    handleStreamEvent(buffer, onPayload);
  }
}

function handleStreamEvent(event: string, onPayload: (payload: ChatStreamPayload) => void) {
  const data = event
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart())
    .join("\n");
  if (!data || data === "[DONE]") {
    return;
  }
  onPayload(JSON.parse(data));
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

function formatTokensPerSecond(value: number) {
  if (!Number.isFinite(value)) {
    return "0.0";
  }
  return value.toFixed(value >= 10 ? 1 : 2);
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

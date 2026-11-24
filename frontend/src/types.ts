export type View = "chat" | "models" | "tune" | "adapters" | "system";

export type ModelProfile = {
  id: string;
  label: string;
  family: string;
  size: string;
  quantization: string;
  recommended_memory_gb: number;
  context_length?: number;
  qat?: boolean;
  modality: string;
  default: boolean;
  downloaded: boolean;
  local_path: string | null;
  notes: string;
};

export type DownloadJob = {
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
  created_at?: string;
};

export type MlxInstallJob = {
  id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  command: string[];
  cwd: string | null;
  output: string[];
  returncode: number | null;
  error: string | null;
  message: string;
};

export type ModelMemoryStatus = {
  loaded: boolean;
  loaded_count: number;
  loaded_models: Array<{
    id: string;
    path: string;
    adapter_path?: string | null;
    backend?: "text" | "vlm";
  }>;
  process_memory_bytes: number | null;
  baseline_memory_bytes: number | null;
  model_memory_bytes: number | null;
  mlx_active_memory_bytes?: number | null;
  mlx_cache_memory_bytes?: number | null;
  mlx_peak_memory_bytes?: number | null;
  note: string;
};

export type InferenceStatus = {
  available: boolean;
  multimodal_available?: boolean;
  installing: boolean;
  install_job_id: string | null;
  command: string[];
  cwd: string | null;
  message: string;
  error: string | null;
  job: MlxInstallJob | null;
  model_memory: ModelMemoryStatus;
  active_adapter?: string | null;
};

export type ModelSource = "downloaded" | "path";

export type SystemInfo = {
  os_name: string;
  os_version: string;
  machine: string;
  python_version: string;
  total_memory_gb: number;
  is_macos: boolean;
  is_apple_silicon: boolean;
  mlx_installed?: boolean;
  mlx_vlm_installed?: boolean;
  mlx_ready: boolean;
  multimodal_ready?: boolean;
  default_model?: string;
  recommendations: string[];
};

export type TuneJob = {
  id: string;
  status: "queued" | "running" | "ready" | "succeeded" | "failed" | "cancelled";
  request: {
    model: string;
    data_path: string;
    adapter_name: string;
    iters: number;
    batch_size: number;
    learning_rate: number;
    fine_tune_type: "lora" | "dora" | "full";
    num_layers?: number | null;
    grad_checkpoint?: boolean;
  };
  dataset: {
    data_path: string;
    files: Array<{
      path: string;
      format: "chat" | "completion" | "text";
      examples: number;
    }>;
  };
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
  progress_percent: number;
  current_step: string;
  current_iter?: number;
  train_loss?: number | null;
  val_loss?: number | null;
  tokens_per_second?: number | null;
  peak_memory_gb?: number | null;
  adapter_path?: string | null;
  error?: string | null;
  logs: string[];
  message: string;
};

export type SyntheticFormat = "chat" | "completion" | "text";

export type SyntheticDatasetResult = {
  data_path: string;
  train_path: string;
  valid_path?: string | null;
  format: SyntheticFormat;
  examples: number;
  report: TuneJob["dataset"];
};

export type AdapterInfo = {
  id: string;
  path: string;
  active: boolean;
  has_weights?: boolean;
};

export type GenerationStats = {
  completionTokens: number;
  tokensPerSecond: number;
  elapsedSeconds: number;
};

export type MediaAttachment = {
  id: string;
  kind: "image" | "audio";
  name: string;
  mimeType: string;
  dataUrl: string;
  sizeBytes: number;
};

export type ChatTurn = {
  id: string;
  role: "user" | "assistant";
  content: string;
  attachments?: MediaAttachment[];
  stats?: GenerationStats | null;
  stopped?: boolean;
};

export type ChatStreamPayload = {
  choices?: Array<{
    delta?: { role?: string; content?: string };
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

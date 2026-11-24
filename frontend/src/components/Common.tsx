import React from "react";
import { AlertCircle, CheckCircle2, Circle, Cpu, Trash2, Wrench } from "lucide-react";

import { formatCommand, formatOptionalBytes } from "../format";
import type { InferenceStatus, MlxInstallJob, ModelMemoryStatus } from "../types";

export function PageHeader({
  title,
  eyebrow,
  children,
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

export function StatusBadge({
  tone = "neutral",
  children,
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

export function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function ModelMemoryCard({
  memory,
  onUnload,
  isUnloading = false,
}: {
  memory: ModelMemoryStatus | null;
  onUnload?: () => void;
  isUnloading?: boolean;
}) {
  const value = memory
    ? formatOptionalBytes(memory.model_memory_bytes, memory.loaded ? "Unavailable" : "Not loaded")
    : "Loading";
  const processValue = formatOptionalBytes(memory?.process_memory_bytes ?? null);
  const mlxCacheValue =
    typeof memory?.mlx_cache_memory_bytes === "number"
      ? formatOptionalBytes(memory.mlx_cache_memory_bytes)
      : null;
  const mlxPeakValue =
    typeof memory?.mlx_peak_memory_bytes === "number"
      ? formatOptionalBytes(memory.mlx_peak_memory_bytes)
      : null;

  return (
    <section className="panel memory-card" aria-label="Model memory usage">
      <div className="block-heading memory-heading">
        <span className="heading-inline">
          <Cpu size={18} />
          <strong>Model memory</strong>
        </span>
        {memory?.loaded && onUnload && (
          <button
            type="button"
            className="secondary compact-button"
            disabled={isUnloading}
            data-busy={isUnloading || undefined}
            onClick={onUnload}
            title="Unload loaded models from unified memory"
          >
            <Trash2 size={14} />
            <span>{isUnloading ? "Unloading" : "Unload"}</span>
          </button>
        )}
      </div>
      <strong className="memory-value">{value}</strong>
      <p>{memory?.note ?? "Reading local memory status..."}</p>
      <dl className="memory-details">
        <div>
          <dt>Loaded</dt>
          <dd>
            {memory
              ? memory.loaded
                ? `${memory.loaded_count} model${memory.loaded_count === 1 ? "" : "s"}`
                : "No model loaded"
              : "Loading"}
          </dd>
        </div>
        <div>
          <dt>Process</dt>
          <dd>{processValue}</dd>
        </div>
        {mlxCacheValue && (
          <div>
            <dt>MLX Cache</dt>
            <dd>{mlxCacheValue}</dd>
          </div>
        )}
        {mlxPeakValue && (
          <div>
            <dt>MLX Peak</dt>
            <dd>{mlxPeakValue}</dd>
          </div>
        )}
      </dl>
      {memory && memory.loaded_models.length > 0 && (
        <ul className="loaded-models-list">
          {memory.loaded_models.map((item) => (
            <li key={`${item.id}-${item.adapter_path ?? "base"}`}>
              <strong>{item.id}</strong>
              {item.adapter_path && <span className="inline-pill">LoRA: {item.adapter_path}</span>}
              <small>{item.path}</small>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export function InferenceSetup({
  status,
  job,
  isInstalling,
  onInstall,
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
      <button
        type="button"
        className="secondary"
        disabled={isInstalling}
        data-busy={isInstalling || undefined}
        onClick={onInstall}
      >
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

import React from "react";
import { Check, Copy, RefreshCw } from "lucide-react";

import { fetchJson } from "../api";
import type { InferenceStatus, SystemInfo } from "../types";
import { MetricCard, ModelMemoryCard, NoticeBanner, PageHeader, StatusBadge } from "./Common";

export function SystemPanel({
  onStatusChange,
  onSystemLoaded,
}: {
  onStatusChange?: (status: InferenceStatus) => void;
  onSystemLoaded?: (system: SystemInfo) => void;
}) {
  const [system, setSystem] = React.useState<SystemInfo | null>(null);
  const [inference, setInference] = React.useState<InferenceStatus | null>(null);
  const [isUnloading, setIsUnloading] = React.useState(false);
  const [isRefreshing, setIsRefreshing] = React.useState(false);
  const [copiedDiag, setCopiedDiag] = React.useState(false);
  const [message, setMessage] = React.useState("");

  React.useEffect(() => {
    refresh();
  }, []);

  async function refresh() {
    setIsRefreshing(true);
    try {
      const [systemData, inferenceData] = await Promise.all([
        fetchJson<SystemInfo>("/api/system", undefined, "Could not load system status."),
        fetchJson<InferenceStatus>(
          "/api/inference/status",
          undefined,
          "Could not load inference status.",
        ),
      ]);
      setSystem(systemData);
      setInference(inferenceData);
      onSystemLoaded?.(systemData);
      onStatusChange?.(inferenceData);
    } catch (error: unknown) {
      setMessage(error instanceof Error ? error.message : "Could not load system status.");
    } finally {
      setIsRefreshing(false);
    }
  }

  async function unloadModels() {
    setIsUnloading(true);
    setMessage("");
    try {
      await fetchJson("/api/models/unload", { method: "POST" }, "Could not unload models.");
      await refresh();
    } catch (error: unknown) {
      setMessage(error instanceof Error ? error.message : "Could not unload models.");
    } finally {
      setIsUnloading(false);
    }
  }

  async function copyDiagnostics() {
    if (!system) {
      return;
    }
    try {
      await navigator.clipboard.writeText(JSON.stringify(system, null, 2));
      setCopiedDiag(true);
      window.setTimeout(() => setCopiedDiag(false), 1600);
    } catch {
      // Clipboard unavailable.
    }
  }

  return (
    <>
      <PageHeader title="System" eyebrow="Apple Silicon & runtime diagnostics">
        <StatusBadge tone={system?.mlx_ready ? "good" : "warn"}>
          {system?.mlx_ready ? "MLX ready" : "MLX setup"}
        </StatusBadge>
        {system && (
          <button
            type="button"
            className="secondary compact-button"
            onClick={copyDiagnostics}
            title="Copy doctor JSON for bug reports"
          >
            {copiedDiag ? <Check size={14} /> : <Copy size={14} />}
            <span>{copiedDiag ? "Copied JSON" : "Copy doctor JSON"}</span>
          </button>
        )}
        <button
          type="button"
          className="secondary compact-button"
          disabled={isRefreshing}
          data-busy={isRefreshing || undefined}
          onClick={refresh}
        >
          <RefreshCw size={14} />
          <span>Refresh</span>
        </button>
      </PageHeader>
      {message && <NoticeBanner message={message} onDismiss={() => setMessage("")} />}
      <div className="stats-grid">
        <MetricCard label="Machine" value={system?.machine ?? "Loading"} />
        <MetricCard
          label="Unified Memory"
          value={system ? `${system.total_memory_gb} GB` : "Loading"}
        />
        <MetricCard label="Python" value={system?.python_version ?? "Loading"} />
        <MetricCard
          label="MLX Text"
          value={
            system
              ? system.mlx_ready
                ? "Ready (Apple Silicon)"
                : system.mlx_installed
                  ? "Installed (non-Apple Silicon)"
                  : "Not installed"
              : "Loading"
          }
        />
        <MetricCard
          label="Vision & Audio (MLX-VLM)"
          value={
            system
              ? system.multimodal_ready
                ? "Ready (Apple Silicon)"
                : system.mlx_vlm_installed
                  ? "Installed (non-Apple Silicon)"
                  : "Not installed"
              : "Loading"
          }
        />
        <MetricCard label="Active Adapter" value={inference?.active_adapter ?? "None"} />
      </div>
      <div className="system-memory-wrap">
        <ModelMemoryCard
          memory={inference?.model_memory ?? null}
          totalSystemGb={system?.total_memory_gb}
          onUnload={unloadModels}
          isUnloading={isUnloading}
        />
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

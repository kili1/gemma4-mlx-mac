import React from "react";

import { fetchJson } from "../api";
import type { InferenceStatus, SystemInfo } from "../types";
import { MetricCard, ModelMemoryCard, PageHeader, StatusBadge } from "./Common";

export function SystemPanel() {
  const [system, setSystem] = React.useState<SystemInfo | null>(null);
  const [inference, setInference] = React.useState<InferenceStatus | null>(null);
  const [isUnloading, setIsUnloading] = React.useState(false);
  const [message, setMessage] = React.useState("");

  React.useEffect(() => {
    refresh();
  }, []);

  async function refresh() {
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
    } catch (error: unknown) {
      setMessage(error instanceof Error ? error.message : "Could not load system status.");
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

  return (
    <>
      <PageHeader title="System" eyebrow="Runtime status">
        <StatusBadge tone={system?.mlx_ready ? "good" : "warn"}>
          {system?.mlx_ready ? "MLX ready" : "MLX setup"}
        </StatusBadge>
      </PageHeader>
      {message && <p className="notice">{message}</p>}
      <div className="stats-grid">
        <MetricCard label="Machine" value={system?.machine ?? "Loading"} />
        <MetricCard label="Memory" value={system ? `${system.total_memory_gb} GB` : "Loading"} />
        <MetricCard label="Python" value={system?.python_version ?? "Loading"} />
        <MetricCard
          label="MLX"
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
        <MetricCard label="Default Model" value={system?.default_model ?? "Loading"} />
        <MetricCard label="Active Adapter" value={inference?.active_adapter ?? "None"} />
      </div>
      <div className="system-memory-wrap">
        <ModelMemoryCard
          memory={inference?.model_memory ?? null}
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

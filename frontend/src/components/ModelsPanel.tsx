import React from "react";
import { ArrowRight, Download, FolderOpen, Settings2 } from "lucide-react";

import { fetchJson } from "../api";
import { formatContextLength, modelSlug, progressDetails } from "../format";
import { memoryFitStatus } from "../storage";
import type { DownloadJob, ModelProfile } from "../types";
import { NoticeBanner, PageHeader, StatusBadge } from "./Common";

type ModelFilter = "all" | "downloaded" | "qat" | "any-to-any" | "bf16";

export function ModelsPanel({
  totalMemoryGb,
  onChatWithModel,
}: {
  totalMemoryGb?: number;
  onChatWithModel: (modelId: string) => void;
}) {
  const [models, setModels] = React.useState<ModelProfile[]>([]);
  const [downloads, setDownloads] = React.useState<Record<string, DownloadJob>>({});
  const [startingModels, setStartingModels] = React.useState<Record<string, boolean>>({});
  const [downloadFolder, setDownloadFolder] = React.useState("");
  const [showFolderSettings, setShowFolderSettings] = React.useState(false);
  const [filter, setFilter] = React.useState<ModelFilter>("all");
  const [message, setMessage] = React.useState<string>("Loading model profiles…");

  const mountedRef = React.useRef(true);
  const timersRef = React.useRef<Map<string, number>>(new Map());

  React.useEffect(() => {
    mountedRef.current = true;
    const timers = timersRef.current;

    Promise.all([
      fetchJson<{ models: ModelProfile[] }>("/api/models", undefined, "Could not load models."),
      fetchJson<{ downloads: DownloadJob[] }>(
        "/api/models/downloads",
        undefined,
        "Could not load downloads.",
      ),
    ])
      .then(([modelsData, downloadsData]) => {
        if (!mountedRef.current) {
          return;
        }
        setModels(modelsData.models);
        setMessage("");

        const latestByModel: Record<string, DownloadJob> = {};
        for (const job of downloadsData.downloads) {
          latestByModel[job.model] = job;
        }
        setDownloads(latestByModel);
        for (const job of Object.values(latestByModel)) {
          if (job.status === "queued" || job.status === "running") {
            schedulePoll(job.model, job.id);
          }
        }
      })
      .catch((error: unknown) => {
        if (mountedRef.current) {
          setMessage(error instanceof Error ? error.message : "Could not load models.");
        }
      });

    return () => {
      mountedRef.current = false;
      for (const timer of timers.values()) {
        window.clearTimeout(timer);
      }
      timers.clear();
    };
  }, []);

  function schedulePoll(model: string, jobId: string) {
    const existing = timersRef.current.get(model);
    if (existing !== undefined) {
      window.clearTimeout(existing);
    }
    const timer = window.setTimeout(() => pollDownload(model, jobId), 800);
    timersRef.current.set(model, timer);
  }

  async function downloadModel(model: string) {
    setStartingModels((prev) => ({ ...prev, [model]: true }));
    setMessage("");
    const targetFolder = downloadFolder.trim();
    try {
      const data = await fetchJson<DownloadJob>(
        "/api/models/download",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model,
            local_dir: targetFolder || null,
            subdir_per_model: Boolean(targetFolder),
          }),
        },
        "Download failed.",
      );
      if (!mountedRef.current) {
        return;
      }
      setDownloads((current) => ({ ...current, [model]: data }));
      pollDownload(model, data.id);
    } catch (error: unknown) {
      if (mountedRef.current) {
        setMessage(error instanceof Error ? error.message : "Download failed.");
      }
    } finally {
      if (mountedRef.current) {
        setStartingModels((prev) => {
          const next = { ...prev };
          delete next[model];
          return next;
        });
      }
    }
  }

  async function pollDownload(model: string, jobId: string) {
    try {
      const job = await fetchJson<DownloadJob>(
        `/api/models/download/${jobId}`,
        undefined,
        "Could not read download progress.",
      );
      if (!mountedRef.current) {
        return;
      }
      setDownloads((current) => ({ ...current, [model]: job }));
      if (job.status === "queued" || job.status === "running") {
        schedulePoll(model, jobId);
        return;
      }
      timersRef.current.delete(model);
      if (job.status === "failed") {
        setMessage(job.error ?? "Download failed.");
      }
      if (job.status === "succeeded") {
        setModels((current) =>
          current.map((profile) =>
            profile.id === model
              ? { ...profile, downloaded: true, local_path: job.path }
              : profile,
          ),
        );
      }
    } catch (error: unknown) {
      timersRef.current.delete(model);
      if (mountedRef.current) {
        setMessage(error instanceof Error ? error.message : "Could not read download progress.");
      }
    }
  }

  const trimmedFolder = downloadFolder.trim().replace(/\/+$/, "");
  const filteredModels = models.filter((model) => {
    if (filter === "downloaded") {
      return model.downloaded;
    }
    if (filter === "qat") {
      return Boolean(model.qat);
    }
    if (filter === "any-to-any") {
      return model.modality === "any-to-any";
    }
    if (filter === "bf16") {
      return model.quantization.toLowerCase() === "bf16";
    }
    return true;
  });

  return (
    <>
      <PageHeader title="Models" eyebrow="Gemma 4 Hugging Face snapshots">
        {totalMemoryGb ? (
          <StatusBadge tone="neutral">{totalMemoryGb} GB RAM detected</StatusBadge>
        ) : null}
        <StatusBadge tone={models.some((model) => model.downloaded) ? "good" : "neutral"}>
          {models.filter((model) => model.downloaded).length} downloaded
        </StatusBadge>
      </PageHeader>

      <div className="filter-bar">
        <div className="filter-pills" role="group" aria-label="Filter models">
          {(
            [
              ["all", "All"],
              ["downloaded", "Downloaded"],
              ["qat", "QAT 4-bit"],
              ["any-to-any", "Vision + Audio"],
              ["bf16", "Workstation bf16"],
            ] as const
          ).map(([id, label]) => (
            <button
              type="button"
              key={id}
              className={`pill-toggle${filter === id ? " active" : ""}`}
              onClick={() => setFilter(id)}
            >
              {label}
            </button>
          ))}
        </div>

        <button
          type="button"
          className="secondary compact-button"
          onClick={() => setShowFolderSettings((open) => !open)}
        >
          <Settings2 size={14} />
          <span>{trimmedFolder ? "Custom folder set" : "Download folder"}</span>
        </button>
      </div>

      {showFolderSettings && (
        <section className="download-settings panel" aria-label="Model download location">
          <label className="field">
            <span>Custom base folder (each model saves to its own subfolder)</span>
            <div className="path-input">
              <FolderOpen size={17} />
              <input
                value={downloadFolder}
                onChange={(event) => setDownloadFolder(event.target.value)}
                placeholder="~/Models/gemma4"
              />
            </div>
            <small>
              {trimmedFolder
                ? `Target path pattern: ${trimmedFolder}/<model-name>`
                : "Leave blank to use the default Hugging Face cache (~/.cache/huggingface/hub)"}
            </small>
          </label>
        </section>
      )}

      {message && <NoticeBanner message={message} onDismiss={() => setMessage("")} />}

      <div className="model-list">
        {filteredModels.length === 0 ? (
          <div className="panel empty-state">
            <strong>No models match this filter</strong>
            <small>Switch back to All to browse every Gemma 4 MLX profile.</small>
          </div>
        ) : (
          filteredModels.map((model) => {
            const job = downloads[model.id];
            const isActive =
              Boolean(startingModels[model.id]) ||
              job?.status === "queued" ||
              job?.status === "running";
            const contextBadge = formatContextLength(model.context_length);
            const fit = memoryFitStatus(model.recommended_memory_gb, totalMemoryGb);
            const buttonLabel = startingModels[model.id]
              ? "Starting"
              : job?.status === "queued"
                ? "Queued"
                : job?.status === "running"
                  ? "Downloading"
                  : model.downloaded
                    ? "Re-download"
                    : "Download";

            return (
              <div className="model-card" key={model.id}>
                <div className="model-row">
                  <div>
                    <div className="model-title">
                      <strong>{model.label}</strong>
                      <div className="badges">
                        {model.default && <span>Default</span>}
                        {model.qat && <span>QAT</span>}
                        <span className={fit.tone}>{fit.label}</span>
                        {model.downloaded && <span className="good">Downloaded</span>}
                      </div>
                    </div>
                    <p className="model-id-code">{model.id}</p>
                    {model.notes && <p className="model-notes">{model.notes}</p>}
                    <div className="model-meta">
                      <span className="tabular">{model.recommended_memory_gb} GB+ min</span>
                      <span>{model.quantization}</span>
                      {contextBadge && <span>{contextBadge}</span>}
                      <span>{model.modality}</span>
                      {trimmedFolder && (
                        <span>
                          {trimmedFolder}/{modelSlug(model.id)}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="model-row-actions">
                    {model.downloaded && (
                      <button
                        type="button"
                        className="primary compact-button"
                        onClick={() => onChatWithModel(model.id)}
                        title={`Chat with ${model.label}`}
                      >
                        <span>Chat</span>
                        <ArrowRight size={14} />
                      </button>
                    )}
                    <button
                      type="button"
                      className="secondary compact-button"
                      disabled={isActive}
                      data-busy={isActive || undefined}
                      onClick={() => downloadModel(model.id)}
                      title={`Download ${model.label}`}
                    >
                      <Download size={15} />
                      <span>{buttonLabel}</span>
                    </button>
                  </div>
                </div>
                {job && <DownloadProgress job={job} />}
              </div>
            );
          })
        )}
      </div>
    </>
  );
}

function DownloadProgress({ job }: { job: DownloadJob }) {
  const percent = job.percent ?? 0;
  const progressLabel = job.percent === null ? "Preparing" : `${job.percent}%`;
  const terminalMessage =
    job.status === "succeeded" && job.path ? `Saved to ${job.path}` : job.error;

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

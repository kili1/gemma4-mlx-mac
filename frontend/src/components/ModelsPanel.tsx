import React from "react";
import { Download, FolderOpen } from "lucide-react";

import { fetchJson } from "../api";
import { formatContextLength, modelSlug, progressDetails } from "../format";
import type { DownloadJob, ModelProfile } from "../types";
import { PageHeader, StatusBadge } from "./Common";

export function ModelsPanel() {
  const [models, setModels] = React.useState<ModelProfile[]>([]);
  const [downloads, setDownloads] = React.useState<Record<string, DownloadJob>>({});
  const [startingModels, setStartingModels] = React.useState<Record<string, boolean>>({});
  const [downloadFolder, setDownloadFolder] = React.useState("");
  const [message, setMessage] = React.useState<string>("Loading model profiles...");

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

  return (
    <>
      <PageHeader title="Models" eyebrow="Hugging Face snapshots">
        <StatusBadge tone={models.some((model) => model.downloaded) ? "good" : "neutral"}>
          {models.filter((model) => model.downloaded).length} downloaded
        </StatusBadge>
      </PageHeader>

      <section className="download-settings panel" aria-label="Model download location">
        <label className="field">
          <span>Download folder (each model is saved in its own subfolder)</span>
          <div className="path-input">
            <FolderOpen size={18} />
            <input
              value={downloadFolder}
              onChange={(event) => setDownloadFolder(event.target.value)}
              placeholder="~/Models/gemma4"
            />
          </div>
          <small>
            {trimmedFolder
              ? `Saves into ${trimmedFolder}/<model-name>`
              : "Default Hugging Face cache"}
          </small>
        </label>
      </section>
      {message && <p className="notice">{message}</p>}
      <div className="model-list">
        {models.map((model) => {
          const job = downloads[model.id];
          const isActive =
            Boolean(startingModels[model.id]) ||
            job?.status === "queued" ||
            job?.status === "running";
          const contextBadge = formatContextLength(model.context_length);
          const buttonLabel = startingModels[model.id]
            ? "Starting"
            : job?.status === "queued"
              ? "Queued"
              : job?.status === "running"
                ? "Downloading"
                : model.downloaded
                  ? "Download again"
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
                      {model.downloaded && <span className="good">Downloaded</span>}
                    </div>
                  </div>
                  <p>{model.id}</p>
                  {model.notes && <p className="model-notes">{model.notes}</p>}
                  <div className="model-meta">
                    <span>{model.recommended_memory_gb} GB+</span>
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
                <button
                  type="button"
                  className="secondary"
                  disabled={isActive}
                  data-busy={isActive || undefined}
                  onClick={() => downloadModel(model.id)}
                  title={`Download ${model.label}`}
                >
                  <Download size={18} />
                  <span>{buttonLabel}</span>
                </button>
              </div>
              {job && <DownloadProgress job={job} />}
            </div>
          );
        })}
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

import React from "react";
import { FileText, SlidersHorizontal, Square, Terminal } from "lucide-react";

import { fetchJson } from "../api";
import {
  clampProgress,
  countDatasetExamples,
  isTuneActive,
  parsePositiveFloat,
  parsePositiveInt,
  tuneStatusTone,
} from "../format";
import type { ModelProfile, SyntheticDatasetResult, SyntheticFormat, TuneJob } from "../types";
import { PageHeader, StatusBadge } from "./Common";

export function TunePanel() {
  const [models, setModels] = React.useState<ModelProfile[]>([]);
  const [model, setModel] = React.useState("");
  const [dataPath, setDataPath] = React.useState("examples/data");
  const [adapterName, setAdapterName] = React.useState("demo-adapter");
  const [itersInput, setItersInput] = React.useState("100");
  const [batchSizeInput, setBatchSizeInput] = React.useState("1");
  const [learningRateInput, setLearningRateInput] = React.useState("1e-5");
  const [fineTuneType, setFineTuneType] = React.useState<"lora" | "dora" | "full">("lora");
  const [numLayersInput, setNumLayersInput] = React.useState("");
  const [gradCheckpoint, setGradCheckpoint] = React.useState(false);

  const [syntheticTopic, setSyntheticTopic] = React.useState("local Apple Silicon AI");
  const [syntheticExamplesInput, setSyntheticExamplesInput] = React.useState("24");
  const [syntheticFormat, setSyntheticFormat] = React.useState<SyntheticFormat>("chat");
  const [syntheticOutput, setSyntheticOutput] = React.useState("examples/synthetic");

  const [message, setMessage] = React.useState("");
  const [jobs, setJobs] = React.useState<TuneJob[]>([]);
  const [selectedJobId, setSelectedJobId] = React.useState<string | null>(null);
  const [syntheticResult, setSyntheticResult] = React.useState<SyntheticDatasetResult | null>(null);
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [isCancelling, setIsCancelling] = React.useState(false);
  const [isGeneratingData, setIsGeneratingData] = React.useState(false);

  const mountedRef = React.useRef(true);
  const pollTimerRef = React.useRef<number | null>(null);

  const job =
    (selectedJobId ? jobs.find((entry) => entry.id === selectedJobId) : null) ??
    jobs[jobs.length - 1] ??
    null;

  React.useEffect(() => {
    mountedRef.current = true;
    Promise.all([
      fetchJson<{ models: ModelProfile[] }>("/api/models", undefined, "Could not load models."),
      fetchJson<{ tunes: TuneJob[] }>("/api/tunes", undefined, "Could not load fine-tune jobs."),
    ])
      .then(([modelsData, tunesData]) => {
        if (!mountedRef.current) {
          return;
        }
        setModels(modelsData.models);
        setModel(
          modelsData.models.find((profile) => profile.default)?.id ??
            modelsData.models[0]?.id ??
            "",
        );
        setJobs(tunesData.tunes);
        const latest = tunesData.tunes[tunesData.tunes.length - 1];
        if (latest) {
          setSelectedJobId(latest.id);
          if (isTuneActive(latest.status)) {
            schedulePoll(latest.id);
          }
        }
      })
      .catch((error: unknown) => {
        if (mountedRef.current) {
          setMessage(error instanceof Error ? error.message : "Could not load model profiles.");
        }
      });

    return () => {
      mountedRef.current = false;
      if (pollTimerRef.current !== null) {
        window.clearTimeout(pollTimerRef.current);
      }
    };
  }, []);

  function schedulePoll(jobId: string) {
    if (pollTimerRef.current !== null) {
      window.clearTimeout(pollTimerRef.current);
    }
    pollTimerRef.current = window.setTimeout(() => pollTune(jobId), 800);
  }

  function upsertJob(updated: TuneJob) {
    setJobs((prev) => {
      const exists = prev.some((item) => item.id === updated.id);
      return exists
        ? prev.map((item) => (item.id === updated.id ? updated : item))
        : [...prev, updated];
    });
  }

  async function startTune() {
    setIsSubmitting(true);
    setMessage("");
    const iters = parsePositiveInt(itersInput, 100, 1);
    const batchSize = parsePositiveInt(batchSizeInput, 1, 1);
    const learningRate = parsePositiveFloat(learningRateInput, 1e-5, 1e-9);
    const parsedNumLayers = numLayersInput.trim()
      ? Number.parseInt(numLayersInput.trim(), 10)
      : null;

    try {
      const data = await fetchJson<TuneJob>(
        "/api/tunes",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model,
            data_path: dataPath.trim(),
            adapter_name: adapterName.trim(),
            iters,
            batch_size: batchSize,
            learning_rate: learningRate,
            fine_tune_type: fineTuneType,
            num_layers: Number.isFinite(parsedNumLayers) ? parsedNumLayers : null,
            grad_checkpoint: gradCheckpoint,
          }),
        },
        "Fine-tune job failed.",
      );
      if (!mountedRef.current) {
        return;
      }
      upsertJob(data);
      setSelectedJobId(data.id);
      if (isTuneActive(data.status)) {
        schedulePoll(data.id);
      }
    } catch (error: unknown) {
      if (mountedRef.current) {
        setMessage(error instanceof Error ? error.message : "Fine-tune job failed.");
      }
    } finally {
      if (mountedRef.current) {
        setIsSubmitting(false);
      }
    }
  }

  async function cancelTune(jobId: string) {
    setIsCancelling(true);
    setMessage("");
    try {
      const updated = await fetchJson<TuneJob>(
        `/api/tunes/${jobId}/cancel`,
        { method: "POST" },
        "Could not cancel fine-tune job.",
      );
      if (mountedRef.current) {
        upsertJob(updated);
      }
    } catch (error: unknown) {
      if (mountedRef.current) {
        setMessage(error instanceof Error ? error.message : "Could not cancel fine-tune job.");
      }
    } finally {
      if (mountedRef.current) {
        setIsCancelling(false);
      }
    }
  }

  async function pollTune(jobId: string) {
    try {
      const data = await fetchJson<TuneJob>(
        `/api/tunes/${jobId}`,
        undefined,
        "Could not read job.",
      );
      if (!mountedRef.current) {
        return;
      }
      upsertJob(data);
      if (isTuneActive(data.status)) {
        schedulePoll(jobId);
      }
    } catch (error: unknown) {
      if (mountedRef.current) {
        setMessage(error instanceof Error ? error.message : "Could not read fine-tune progress.");
      }
    }
  }

  async function generateSyntheticData() {
    setIsGeneratingData(true);
    setMessage("");
    setSyntheticResult(null);
    const examples = parsePositiveInt(syntheticExamplesInput, 24, 1, 500);
    try {
      const data = await fetchJson<SyntheticDatasetResult>(
        "/api/datasets/synthetic",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            topic: syntheticTopic.trim(),
            examples,
            output_dir: syntheticOutput.trim(),
            format: syntheticFormat,
            system_prompt: "You are a concise, helpful assistant.",
          }),
        },
        "Synthetic dataset generation failed.",
      );
      if (!mountedRef.current) {
        return;
      }
      setSyntheticResult(data);
      setDataPath(data.data_path);
      setMessage(`Synthetic dataset created at ${data.data_path} (train + valid JSONL).`);
    } catch (error: unknown) {
      if (mountedRef.current) {
        setMessage(
          error instanceof Error ? error.message : "Synthetic dataset generation failed.",
        );
      }
    } finally {
      if (mountedRef.current) {
        setIsGeneratingData(false);
      }
    }
  }

  return (
    <>
      <PageHeader title="Fine-tune" eyebrow="MLX-LM LoRA / QLoRA">
        <StatusBadge tone={tuneStatusTone(job?.status)}>{job?.status ?? "idle"}</StatusBadge>
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
                  value={syntheticExamplesInput}
                  onChange={(event) => setSyntheticExamplesInput(event.target.value)}
                />
              </label>
            </div>
            <button
              type="button"
              className="secondary"
              disabled={isGeneratingData || !syntheticTopic.trim()}
              data-busy={isGeneratingData || undefined}
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
                  value={itersInput}
                  onChange={(event) => setItersInput(event.target.value)}
                />
              </label>
              <label className="field">
                <span>Batch</span>
                <input
                  min={1}
                  type="number"
                  value={batchSizeInput}
                  onChange={(event) => setBatchSizeInput(event.target.value)}
                />
              </label>
              <label className="field">
                <span>Learning rate</span>
                <input
                  type="text"
                  value={learningRateInput}
                  onChange={(event) => setLearningRateInput(event.target.value)}
                  placeholder="1e-5"
                />
              </label>
              <label className="field">
                <span>Method</span>
                <select
                  value={fineTuneType}
                  onChange={(event) =>
                    setFineTuneType(event.target.value as "lora" | "dora" | "full")
                  }
                >
                  <option value="lora">LoRA / QLoRA</option>
                  <option value="dora">DoRA</option>
                  <option value="full">Full</option>
                </select>
              </label>
            </div>
            <label className="field">
              <span>Trainable layers (optional, -1 for all)</span>
              <input
                type="number"
                min={-1}
                value={numLayersInput}
                onChange={(event) => setNumLayersInput(event.target.value)}
                placeholder="Default (16)"
              />
            </label>
            <label className="switch-row">
              <span className="switch-copy">
                <strong>Gradient checkpointing</strong>
                <small>Reduces unified memory usage at a modest speed cost.</small>
              </span>
              <input
                aria-label="Enable gradient checkpointing"
                checked={gradCheckpoint}
                className="switch-control"
                onChange={(event) => setGradCheckpoint(event.target.checked)}
                type="checkbox"
              />
            </label>
            <button
              type="button"
              className="primary"
              disabled={isSubmitting || !model || !adapterName.trim()}
              data-busy={isSubmitting || undefined}
              onClick={startTune}
            >
              {isSubmitting ? "Starting" : "Start Fine-tune"}
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
                  <dd>{syntheticResult.examples} train</dd>
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

          {jobs.length > 1 && (
            <label className="field">
              <span>Job history</span>
              <select
                value={job?.id ?? ""}
                onChange={(event) => setSelectedJobId(event.target.value)}
              >
                {[...jobs].reverse().map((entry) => (
                  <option key={entry.id} value={entry.id}>
                    {entry.request.adapter_name} ({entry.status}) — {entry.id.slice(0, 8)}
                  </option>
                ))}
              </select>
            </label>
          )}

          {job ? (
            <>
              <div className="block-heading memory-heading">
                <span className="heading-inline">
                  <Terminal size={18} />
                  <strong>{job.request.adapter_name}</strong>
                </span>
                {isTuneActive(job.status) && (
                  <button
                    type="button"
                    className="secondary compact-button"
                    disabled={isCancelling}
                    data-busy={isCancelling || undefined}
                    onClick={() => cancelTune(job.id)}
                  >
                    <Square size={14} />
                    <span>{isCancelling ? "Cancelling" : "Cancel"}</span>
                  </button>
                )}
              </div>
              <TuneProgress job={job} />
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
                  <dt>Method</dt>
                  <dd>{job.request.fine_tune_type.toUpperCase()}</dd>
                </div>
                {job.adapter_path && (
                  <div>
                    <dt>Output</dt>
                    <dd>{job.adapter_path}</dd>
                  </div>
                )}
              </dl>
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

function TuneProgress({ job }: { job: TuneJob }) {
  const percent = clampProgress(job.progress_percent);
  const recentLogs = job.logs.slice(-12);

  return (
    <div className="tune-progress" aria-label={`Fine-tune progress for ${job.id}`}>
      <div className="progress-meta">
        <span>{job.current_step}</span>
        <span>{percent}%</span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${percent}%` }} />
      </div>
      <p>{job.error ?? job.message}</p>
      {(job.train_loss !== undefined && job.train_loss !== null) ||
      (job.val_loss !== undefined && job.val_loss !== null) ||
      (job.tokens_per_second !== undefined && job.tokens_per_second !== null) ||
      (job.peak_memory_gb !== undefined && job.peak_memory_gb !== null) ? (
        <div className="tune-metrics">
          {job.train_loss != null && <span>Train loss: {job.train_loss.toFixed(3)}</span>}
          {job.val_loss != null && <span>Val loss: {job.val_loss.toFixed(3)}</span>}
          {job.tokens_per_second != null && (
            <span>{job.tokens_per_second.toFixed(1)} tok/s</span>
          )}
          {job.peak_memory_gb != null && <span>Peak: {job.peak_memory_gb.toFixed(2)} GB</span>}
        </div>
      ) : null}
      {recentLogs.length > 0 && (
        <div className="job-log">
          <strong>Activity</strong>
          <ol>
            {recentLogs.map((entry, index) => (
              <li key={`${index}-${entry.slice(0, 24)}`}>{entry}</li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}

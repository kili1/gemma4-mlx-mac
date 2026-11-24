import type { DownloadJob, TuneJob } from "./types";

export function formatBytes(bytes: number): string {
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

export function formatOptionalBytes(
  bytes: number | null | undefined,
  fallback = "Unavailable",
): string {
  return typeof bytes === "number" ? formatBytes(bytes) : fallback;
}

export function formatCommand(command: string[]): string {
  return command.map((part) => (/\s/.test(part) ? `"${part}"` : part)).join(" ");
}

export function formatTokensPerSecond(value: number): string {
  if (!Number.isFinite(value)) {
    return "0.0";
  }
  return value.toFixed(value >= 10 ? 1 : 2);
}

export function formatContextLength(tokens: number | undefined): string | null {
  if (!tokens || !Number.isFinite(tokens)) {
    return null;
  }
  if (tokens >= 1000) {
    return `${Math.round(tokens / 1000)}K ctx`;
  }
  return `${tokens} ctx`;
}

export function clampProgress(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(100, Math.max(0, Math.round(value)));
}

export function countDatasetExamples(dataset: TuneJob["dataset"]): number {
  return dataset.files.reduce((total, file) => total + file.examples, 0);
}

export function isTuneActive(status: TuneJob["status"]): boolean {
  return status === "queued" || status === "running";
}

export function tuneStatusTone(
  status: TuneJob["status"] | undefined,
): "neutral" | "good" | "warn" | "bad" {
  if (status === "failed" || status === "cancelled") {
    return "bad";
  }
  if (status === "running" || status === "queued") {
    return "warn";
  }
  if (status === "ready" || status === "succeeded") {
    return "good";
  }
  return "neutral";
}

export function progressDetails(job: DownloadJob): string {
  if (job.bytes_total && job.bytes_total >= job.bytes_downloaded) {
    return `${formatBytes(job.bytes_downloaded)} / ${formatBytes(job.bytes_total)}`;
  }
  if (job.files_total) {
    return `${job.files_downloaded} / ${job.files_total} files`;
  }
  return job.message;
}

export function modelSlug(modelId: string): string {
  const tail = modelId.trim().split("/").pop() ?? modelId;
  return tail.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[.-]+|[.-]+$/g, "") || "model";
}

export function parsePositiveInt(raw: string, fallback: number, min = 1, max?: number): number {
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed < min) {
    return fallback;
  }
  if (max !== undefined && parsed > max) {
    return max;
  }
  return parsed;
}

export function parsePositiveFloat(raw: string, fallback: number, min = 0, max?: number): number {
  const parsed = Number.parseFloat(raw);
  if (!Number.isFinite(parsed) || parsed < min) {
    return fallback;
  }
  if (max !== undefined && parsed > max) {
    return max;
  }
  return parsed;
}

export function splitThinkingContent(content: string): {
  thinking: string | null;
  answer: string;
} {
  if (!content.startsWith("Thinking:\n")) {
    return { thinking: null, answer: content };
  }
  const rest = content.slice("Thinking:\n".length);
  const marker = "\n\nAnswer:\n";
  const splitIdx = rest.indexOf(marker);
  if (splitIdx === -1) {
    return { thinking: rest.trimEnd(), answer: "" };
  }
  return {
    thinking: rest.slice(0, splitIdx).trim(),
    answer: rest.slice(splitIdx + marker.length),
  };
}

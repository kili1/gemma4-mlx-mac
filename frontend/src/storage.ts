import type { ChatSession, LossPoint, ThemeMode } from "./types";

const THEME_KEY = "gemma4-mlx-mac:theme";
const SESSIONS_KEY = "gemma4-mlx-mac:chat-sessions";
const MAX_SESSIONS = 25;

export function loadThemePreference(): ThemeMode {
  try {
    const raw = window.localStorage.getItem(THEME_KEY);
    if (raw === "light" || raw === "dark" || raw === "system") {
      return raw;
    }
  } catch {
    // Ignore storage access failures in private browsing.
  }
  return "system";
}

export function saveThemePreference(theme: ThemeMode): void {
  try {
    window.localStorage.setItem(THEME_KEY, theme);
  } catch {
    // Ignore storage quota errors.
  }
}

export function resolveTheme(theme: ThemeMode, prefersDark: boolean): "light" | "dark" {
  if (theme === "system") {
    return prefersDark ? "dark" : "light";
  }
  return theme;
}

export function summarizeSessionTitle(prompt: string): string {
  const cleaned = prompt.replace(/\s+/g, " ").trim();
  if (!cleaned) {
    return "New chat";
  }
  return cleaned.length > 48 ? `${cleaned.slice(0, 47).trimEnd()}…` : cleaned;
}

export function loadChatSessions(): ChatSession[] {
  try {
    const raw = window.localStorage.getItem(SESSIONS_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed
      .filter((item): item is ChatSession => Boolean(item && typeof item.id === "string"))
      .slice(0, MAX_SESSIONS);
  } catch {
    return [];
  }
}

export function saveChatSessions(sessions: ChatSession[]): void {
  try {
    // Strip heavy base64 blobs before persisting to localStorage so quota is never exceeded.
    const slim: ChatSession[] = sessions.slice(0, MAX_SESSIONS).map((session) => ({
      ...session,
      turns: session.turns.map((turn) => ({
        ...turn,
        attachments: undefined,
      })),
    }));
    window.localStorage.setItem(SESSIONS_KEY, JSON.stringify(slim));
  } catch {
    // Ignore quota errors cleanly.
  }
}

export function memoryFitStatus(
  recommendedGb: number,
  totalMemoryGb: number | undefined,
): { label: string; tone: "good" | "warn" | "bad" | "neutral" } {
  if (!totalMemoryGb || totalMemoryGb <= 0) {
    return { label: `${recommendedGb} GB+`, tone: "neutral" };
  }
  if (totalMemoryGb >= recommendedGb * 1.25) {
    return { label: "Fits comfortably", tone: "good" };
  }
  if (totalMemoryGb >= recommendedGb) {
    return { label: "Fits", tone: "good" };
  }
  if (totalMemoryGb >= recommendedGb * 0.75) {
    return { label: "Tight fit", tone: "warn" };
  }
  return { label: "May swap / OOM", tone: "bad" };
}

const TRAIN_LOSS_RE = /Iter\s+(\d+):\s+Train loss\s+([0-9.]+)/i;
const VAL_LOSS_RE = /Iter\s+(\d+):\s+Val loss\s+([0-9.]+)/i;

export function parseLossHistory(logs: string[]): LossPoint[] {
  const byIter = new Map<number, LossPoint>();
  for (const line of logs) {
    const trainMatch = TRAIN_LOSS_RE.exec(line);
    if (trainMatch) {
      const iter = Number.parseInt(trainMatch[1], 10);
      const loss = Number.parseFloat(trainMatch[2]);
      if (Number.isFinite(iter) && Number.isFinite(loss)) {
        const existing = byIter.get(iter) ?? { iter };
        existing.trainLoss = loss;
        byIter.set(iter, existing);
      }
      continue;
    }
    const valMatch = VAL_LOSS_RE.exec(line);
    if (valMatch) {
      const iter = Number.parseInt(valMatch[1], 10);
      const loss = Number.parseFloat(valMatch[2]);
      if (Number.isFinite(iter) && Number.isFinite(loss)) {
        const existing = byIter.get(iter) ?? { iter };
        existing.valLoss = loss;
        byIter.set(iter, existing);
      }
    }
  }
  return [...byIter.values()].sort((a, b) => a.iter - b.iter);
}

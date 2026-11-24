import type { ChatStreamPayload } from "./types";

export async function readErrorMessage(response: Response, fallback: string): Promise<string> {
  const raw = await response.text().catch(() => "");
  if (!raw.trim()) {
    return `${fallback} (HTTP ${response.status})`;
  }
  try {
    const parsed = JSON.parse(raw) as {
      detail?: string | Array<{ msg?: string }>;
      error?: string | { message?: string };
    };
    if (typeof parsed.error === "string" && parsed.error) {
      return parsed.error;
    }
    if (parsed.error && typeof parsed.error === "object" && parsed.error.message) {
      return parsed.error.message;
    }
    if (typeof parsed.detail === "string" && parsed.detail) {
      return parsed.detail;
    }
    if (Array.isArray(parsed.detail) && parsed.detail[0]?.msg) {
      return parsed.detail.map((entry) => entry.msg).filter(Boolean).join("; ");
    }
  } catch {
    // Non-JSON error body (e.g. HTML 500 or proxy error).
  }
  return raw.slice(0, 240).trim() || fallback;
}

export async function fetchJson<T>(
  input: string,
  init?: RequestInit,
  fallbackError = "Request failed.",
): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, fallbackError));
  }
  return (await response.json()) as T;
}

export function handleStreamEvent(
  event: string,
  onPayload: (payload: ChatStreamPayload) => void,
): void {
  const data = event
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart())
    .join("\n");
  if (!data || data === "[DONE]") {
    return;
  }
  onPayload(JSON.parse(data) as ChatStreamPayload);
}

export async function readChatStream(
  response: Response,
  onPayload: (payload: ChatStreamPayload) => void,
): Promise<void> {
  if (!response.body) {
    throw new Error("Chat response did not include a stream.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
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
  } finally {
    reader.releaseLock();
  }
}

import type { MediaAttachment } from "./types";

const MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024;

const AUDIO_MIME_TO_FORMAT: Record<string, string> = {
  "audio/wav": "wav",
  "audio/x-wav": "wav",
  "audio/wave": "wav",
  "audio/mpeg": "mp3",
  "audio/mp3": "mp3",
  "audio/mp4": "m4a",
  "audio/m4a": "m4a",
  "audio/x-m4a": "m4a",
  "audio/ogg": "ogg",
  "audio/flac": "flac",
  "audio/webm": "webm",
  "audio/aac": "aac",
};

export function inferAudioFormat(mimeType: string, filename = ""): string {
  const normalized = mimeType.toLowerCase().trim();
  if (normalized in AUDIO_MIME_TO_FORMAT) {
    return AUDIO_MIME_TO_FORMAT[normalized];
  }
  const ext = filename.split(".").pop()?.toLowerCase() ?? "";
  if (["wav", "mp3", "m4a", "ogg", "flac", "webm", "aac"].includes(ext)) {
    return ext;
  }
  return "wav";
}

export function buildOpenAiContentParts(
  text: string,
  attachments: MediaAttachment[] = [],
): string | Array<Record<string, unknown>> {
  if (attachments.length === 0) {
    return text;
  }

  const parts: Array<Record<string, unknown>> = [];
  for (const item of attachments) {
    if (item.kind === "image") {
      parts.push({
        type: "image_url",
        image_url: { url: item.dataUrl },
      });
    } else {
      const commaIdx = item.dataUrl.indexOf(",");
      const b64Data = commaIdx === -1 ? item.dataUrl : item.dataUrl.slice(commaIdx + 1);
      parts.push({
        type: "input_audio",
        input_audio: {
          data: b64Data,
          format: inferAudioFormat(item.mimeType, item.name),
        },
      });
    }
  }
  if (text.trim()) {
    parts.push({ type: "text", text });
  }
  return parts;
}

export function readFileAsAttachment(file: File): Promise<MediaAttachment> {
  return new Promise((resolve, reject) => {
    const mime = file.type.toLowerCase();
    const kind: "image" | "audio" | null = mime.startsWith("image/")
      ? "image"
      : mime.startsWith("audio/")
        ? "audio"
        : null;

    if (!kind) {
      reject(new Error(`Unsupported file type for "${file.name}". Attach an image or audio file.`));
      return;
    }
    if (file.size > MAX_ATTACHMENT_BYTES) {
      reject(new Error(`"${file.name}" exceeds the 25 MB local attachment limit.`));
      return;
    }

    const reader = new FileReader();
    reader.onerror = () => reject(new Error(`Could not read "${file.name}".`));
    reader.onload = () => {
      if (typeof reader.result !== "string") {
        reject(new Error(`Could not encode "${file.name}".`));
        return;
      }
      resolve({
        id: `media-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        kind,
        name: file.name || `${kind}.${kind === "image" ? "png" : "wav"}`,
        mimeType: mime,
        dataUrl: reader.result,
        sizeBytes: file.size,
      });
    };
    reader.readAsDataURL(file);
  });
}

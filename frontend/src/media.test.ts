import assert from "node:assert/strict";
import test from "node:test";

import { buildOpenAiContentParts, inferAudioFormat } from "./media.ts";

test("buildOpenAiContentParts returns plain string when no attachments are present", () => {
  assert.equal(buildOpenAiContentParts("Hello Gemma", []), "Hello Gemma");
});

test("buildOpenAiContentParts builds OpenAI image_url and input_audio parts", () => {
  const parts = buildOpenAiContentParts("Compare these", [
    {
      id: "1",
      kind: "image",
      name: "chart.png",
      mimeType: "image/png",
      dataUrl: "data:image/png;base64,iVBORw0KGgo=",
      sizeBytes: 12,
    },
    {
      id: "2",
      kind: "audio",
      name: "clip.mp3",
      mimeType: "audio/mpeg",
      dataUrl: "data:audio/mpeg;base64,SUQz",
      sizeBytes: 4,
    },
  ]);

  assert.deepEqual(parts, [
    { type: "image_url", image_url: { url: "data:image/png;base64,iVBORw0KGgo=" } },
    { type: "input_audio", input_audio: { data: "SUQz", format: "mp3" } },
    { type: "text", text: "Compare these" },
  ]);
});

test("inferAudioFormat maps common MIME types and file extensions", () => {
  assert.equal(inferAudioFormat("audio/x-wav", "voice.wav"), "wav");
  assert.equal(inferAudioFormat("audio/mpeg", "note.mp3"), "mp3");
  assert.equal(inferAudioFormat("application/octet-stream", "memo.m4a"), "m4a");
});

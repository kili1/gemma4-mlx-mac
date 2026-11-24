import assert from "node:assert/strict";
import test from "node:test";

import {
  clampProgress,
  formatBytes,
  formatContextLength,
  modelSlug,
  parsePositiveFloat,
  parsePositiveInt,
  splitThinkingContent,
} from "./format.ts";

test("splitThinkingContent separates native Thinking and Answer blocks", () => {
  const split = splitThinkingContent("Thinking:\nStep 1\nStep 2\n\nAnswer:\n42");
  assert.equal(split.thinking, "Step 1\nStep 2");
  assert.equal(split.answer, "42");

  const streamingThought = splitThinkingContent("Thinking:\nStill thinking");
  assert.equal(streamingThought.thinking, "Still thinking");
  assert.equal(streamingThought.answer, "");

  const plain = splitThinkingContent("Just an answer");
  assert.equal(plain.thinking, null);
  assert.equal(plain.answer, "Just an answer");
});

test("modelSlug extracts safe folder suffix from Hugging Face ids", () => {
  assert.equal(
    modelSlug("mlx-community/gemma-4-E2B-it-qat-4bit"),
    "gemma-4-E2B-it-qat-4bit",
  );
  assert.equal(modelSlug("org/weird name!!"), "weird-name");
});

test("numeric parsers fall back safely on empty or invalid strings", () => {
  assert.equal(parsePositiveInt("", 100), 100);
  assert.equal(parsePositiveInt("0", 100, 1), 100);
  assert.equal(parsePositiveInt("600", 24, 1, 500), 500);
  assert.equal(parsePositiveFloat("1e-4", 1e-5, 1e-9), 0.0001);
  assert.equal(parsePositiveFloat("abc", 0.7, 0, 2), 0.7);
});

test("formatting helpers clamp and format values cleanly", () => {
  assert.equal(clampProgress(120), 100);
  assert.equal(clampProgress(-5), 0);
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(1024 * 1024 * 15), "15.0 MB");
  assert.equal(formatContextLength(256000), "256K ctx");
});

import assert from "node:assert/strict";
import test from "node:test";

import {
  memoryFitStatus,
  parseLossHistory,
  resolveTheme,
  summarizeSessionTitle,
} from "./storage.ts";

test("parseLossHistory extracts train and val loss points in iteration order", () => {
  const points = parseLossHistory([
    "Loading pretrained model...",
    "Iter 10: Train loss 2.450, Tokens/sec 310.2, Peak mem 4.10 GB",
    "Iter 10: Val loss 2.310, Val took 0.9s",
    "Iter 20: Train loss 1.980, Tokens/sec 325.0, Peak mem 4.12 GB",
  ]);
  assert.deepEqual(points, [
    { iter: 10, trainLoss: 2.45, valLoss: 2.31 },
    { iter: 20, trainLoss: 1.98 },
  ]);
});

test("memoryFitStatus classifies model RAM fit accurately", () => {
  assert.equal(memoryFitStatus(8, 32).tone, "good");
  assert.equal(memoryFitStatus(16, 16).label, "Fits");
  assert.equal(memoryFitStatus(24, 18).tone, "warn");
  assert.equal(memoryFitStatus(64, 16).tone, "bad");
});

test("summarizeSessionTitle trims whitespace and truncates long prompts", () => {
  assert.equal(summarizeSessionTitle("   Hello   world  "), "Hello world");
  assert.equal(summarizeSessionTitle(""), "New chat");
  const long = "A".repeat(70);
  const title = summarizeSessionTitle(long);
  assert.ok(title.endsWith("…") && title.length <= 49);
});

test("resolveTheme handles light, dark, and system fallback", () => {
  assert.equal(resolveTheme("light", true), "light");
  assert.equal(resolveTheme("dark", false), "dark");
  assert.equal(resolveTheme("system", true), "dark");
  assert.equal(resolveTheme("system", false), "light");
});

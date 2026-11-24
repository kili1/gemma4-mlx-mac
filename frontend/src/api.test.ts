import assert from "node:assert/strict";
import test from "node:test";

import { handleStreamEvent, readErrorMessage } from "./api.ts";

test("handleStreamEvent parses SSE JSON chunks and ignores [DONE]", () => {
  const received: unknown[] = [];
  handleStreamEvent('data: {"choices":[{"delta":{"role":"assistant","content":"hi"}}]}', (p) =>
    received.push(p),
  );
  handleStreamEvent("data: [DONE]", (p) => received.push(p));

  assert.equal(received.length, 1);
  assert.deepEqual(received[0], {
    choices: [{ delta: { role: "assistant", content: "hi" } }],
  });
});

test("readErrorMessage extracts JSON details and handles plain HTML/text errors", async () => {
  const jsonResp = new Response(JSON.stringify({ error: { message: "Model not downloaded" } }), {
    status: 503,
  });
  assert.equal(await readErrorMessage(jsonResp, "fallback"), "Model not downloaded");

  const detailResp = new Response(JSON.stringify({ detail: "Invalid adapter" }), { status: 400 });
  assert.equal(await readErrorMessage(detailResp, "fallback"), "Invalid adapter");

  const htmlResp = new Response("<html><body>Bad Gateway</body></html>", { status: 502 });
  assert.match(await readErrorMessage(htmlResp, "fallback"), /Bad Gateway/);
});

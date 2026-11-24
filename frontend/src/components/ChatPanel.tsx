import React from "react";
import {
  Activity,
  FileAudio,
  HardDrive,
  ImagePlus,
  MessageSquareText,
  RotateCcw,
  Send,
  SlidersHorizontal,
  Square,
  X,
} from "lucide-react";

import { fetchJson, readChatStream, readErrorMessage } from "../api";
import { formatBytes, formatTokensPerSecond, parsePositiveFloat, parsePositiveInt } from "../format";
import { buildOpenAiContentParts, readFileAsAttachment } from "../media";
import type {
  ChatTurn,
  InferenceStatus,
  MediaAttachment,
  MlxInstallJob,
  ModelProfile,
  ModelSource,
} from "../types";
import { InferenceSetup, ModelMemoryCard, PageHeader, StatusBadge } from "./Common";
import { MarkdownMessage } from "./MarkdownMessage";

export function ChatPanel() {
  const [models, setModels] = React.useState<ModelProfile[]>([]);
  const [selectedModel, setSelectedModel] = React.useState("");
  const [modelSource, setModelSource] = React.useState<ModelSource>("downloaded");
  const [modelPath, setModelPath] = React.useState("");
  const [prompt, setPrompt] = React.useState("");
  const [attachments, setAttachments] = React.useState<MediaAttachment[]>([]);
  const [isDragOver, setIsDragOver] = React.useState(false);
  const [systemPrompt, setSystemPrompt] = React.useState("");
  const [temperatureInput, setTemperatureInput] = React.useState("0.7");
  const [topPInput, setTopPInput] = React.useState("0.95");
  const [maxTokensInput, setMaxTokensInput] = React.useState("1024");
  const [turns, setTurns] = React.useState<ChatTurn[]>([]);
  const [message, setMessage] = React.useState("");
  const [showThinking, setShowThinking] = React.useState(false);
  const [isSending, setIsSending] = React.useState(false);
  const [isUnloading, setIsUnloading] = React.useState(false);
  const [inferenceStatus, setInferenceStatus] = React.useState<InferenceStatus | null>(null);
  const [installJob, setInstallJob] = React.useState<MlxInstallJob | null>(null);

  const mountedRef = React.useRef(true);
  const pollTimerRef = React.useRef<number | null>(null);
  const abortRef = React.useRef<AbortController | null>(null);
  const threadEndRef = React.useRef<HTMLDivElement | null>(null);
  const imageInputRef = React.useRef<HTMLInputElement | null>(null);
  const audioInputRef = React.useRef<HTMLInputElement | null>(null);

  const downloadedModels = models.filter((model) => model.downloaded);
  const selectedDownloadedModel =
    downloadedModels.find((model) => model.id === selectedModel) ?? null;
  const activeModel =
    modelSource === "path" ? modelPath.trim() : selectedDownloadedModel?.id ?? "";
  const activeModelLabel =
    modelSource === "path" ? modelPath.trim() || "No model path selected" : activeModel;
  const inferenceReady = inferenceStatus?.available === true;
  const multimodalReady = inferenceStatus?.multimodal_available === true;
  const supportsAudio =
    modelSource === "path" ||
    !selectedDownloadedModel ||
    selectedDownloadedModel.modality === "any-to-any";
  const isInstalling =
    installJob?.status === "queued" ||
    installJob?.status === "running" ||
    inferenceStatus?.installing;

  React.useEffect(() => {
    mountedRef.current = true;
    loadModels();
    loadInferenceStatus();
    return () => {
      mountedRef.current = false;
      if (pollTimerRef.current !== null) {
        window.clearTimeout(pollTimerRef.current);
      }
      abortRef.current?.abort();
    };
  }, []);

  React.useEffect(() => {
    threadEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  async function loadModels() {
    try {
      const data = await fetchJson<{ models: ModelProfile[] }>(
        "/api/models",
        undefined,
        "Could not load models.",
      );
      if (!mountedRef.current) {
        return;
      }
      setModels(data.models);
      const availableModels = data.models.filter((model) => model.downloaded);
      const defaultModel = availableModels.find((model) => model.default) ?? availableModels[0];
      if (defaultModel) {
        setSelectedModel((prev) => prev || defaultModel.id);
        if (defaultModel.local_path) {
          setModelPath((prev) => prev || defaultModel.local_path || "");
        }
      } else {
        setSelectedModel("");
      }
    } catch (error: unknown) {
      if (mountedRef.current) {
        setMessage(error instanceof Error ? error.message : "Could not load models.");
      }
    }
  }

  async function loadInferenceStatus() {
    try {
      const status = await fetchJson<InferenceStatus>(
        "/api/inference/status",
        undefined,
        "Could not read MLX status.",
      );
      if (!mountedRef.current) {
        return;
      }
      setInferenceStatus(status);
      if (status.job) {
        setInstallJob(status.job);
      }
      const installJobId = status.install_job_id;
      if (status.installing && installJobId) {
        scheduleInstallPoll(installJobId);
      }
    } catch (error: unknown) {
      if (mountedRef.current) {
        setMessage(error instanceof Error ? error.message : "Could not read MLX status.");
      }
    }
  }

  function scheduleInstallPoll(jobId: string) {
    if (pollTimerRef.current !== null) {
      window.clearTimeout(pollTimerRef.current);
    }
    pollTimerRef.current = window.setTimeout(() => pollInstall(jobId), 1000);
  }

  async function startInstall() {
    setMessage("");
    try {
      const job = await fetchJson<MlxInstallJob>(
        "/api/inference/install",
        { method: "POST" },
        "Could not start MLX install.",
      );
      if (!mountedRef.current) {
        return;
      }
      setInstallJob(job);
      setInferenceStatus((current) =>
        current ? { ...current, installing: job.status !== "succeeded", job } : current,
      );
      if (job.status === "queued" || job.status === "running") {
        pollInstall(job.id);
      } else {
        loadInferenceStatus();
      }
    } catch (error: unknown) {
      if (mountedRef.current) {
        setMessage(error instanceof Error ? error.message : "Could not start MLX install.");
      }
    }
  }

  async function pollInstall(jobId: string) {
    try {
      const job = await fetchJson<MlxInstallJob>(
        `/api/inference/install/${jobId}`,
        undefined,
        "Could not read MLX install progress.",
      );
      if (!mountedRef.current) {
        return;
      }
      setInstallJob(job);
      if (job.status === "queued" || job.status === "running") {
        scheduleInstallPoll(jobId);
        return;
      }
      loadInferenceStatus();
      if (job.status === "failed") {
        setMessage(job.error ?? "MLX install failed.");
      }
    } catch (error: unknown) {
      if (mountedRef.current) {
        setMessage(error instanceof Error ? error.message : "Could not read MLX install progress.");
      }
    }
  }

  async function unloadModels() {
    setIsUnloading(true);
    setMessage("");
    try {
      await fetchJson("/api/models/unload", { method: "POST" }, "Could not unload models.");
      await loadInferenceStatus();
    } catch (error: unknown) {
      if (mountedRef.current) {
        setMessage(error instanceof Error ? error.message : "Could not unload models.");
      }
    } finally {
      if (mountedRef.current) {
        setIsUnloading(false);
      }
    }
  }

  async function addFiles(fileList: FileList | File[]) {
    setMessage("");
    const incoming = Array.from(fileList);
    if (incoming.length === 0) {
      return;
    }
    const added: MediaAttachment[] = [];
    for (const file of incoming) {
      try {
        const attachment = await readFileAsAttachment(file);
        if (attachment.kind === "audio" && !supportsAudio) {
          setMessage(
            `${selectedDownloadedModel?.label ?? "Selected model"} supports text + image only. Switch to E2B, E4B, or 12B for audio.`,
          );
          continue;
        }
        added.push(attachment);
      } catch (error: unknown) {
        setMessage(error instanceof Error ? error.message : "Could not attach file.");
      }
    }
    if (added.length > 0 && mountedRef.current) {
      setAttachments((prev) => [...prev, ...added]);
    }
  }

  function removeAttachment(id: string) {
    setAttachments((prev) => prev.filter((item) => item.id !== id));
  }

  function handlePaste(event: React.ClipboardEvent<HTMLTextAreaElement>) {
    const items = event.clipboardData?.items;
    if (!items) {
      return;
    }
    const mediaFiles: File[] = [];
    for (const item of Array.from(items)) {
      if (item.kind === "file" && (item.type.startsWith("image/") || item.type.startsWith("audio/"))) {
        const file = item.getAsFile();
        if (file) {
          mediaFiles.push(file);
        }
      }
    }
    if (mediaFiles.length > 0) {
      event.preventDefault();
      addFiles(mediaFiles);
    }
  }

  function handleDragOver(event: React.DragEvent<HTMLElement>) {
    event.preventDefault();
    if (!isDragOver) {
      setIsDragOver(true);
    }
  }

  function handleDragLeave(event: React.DragEvent<HTMLElement>) {
    if (event.currentTarget.contains(event.relatedTarget as Node | null)) {
      return;
    }
    setIsDragOver(false);
  }

  function handleDrop(event: React.DragEvent<HTMLElement>) {
    event.preventDefault();
    setIsDragOver(false);
    if (event.dataTransfer?.files?.length) {
      addFiles(event.dataTransfer.files);
    }
  }

  function chooseModelSource(source: ModelSource) {
    setModelSource(source);
    if (source === "path" && !modelPath && selectedDownloadedModel?.local_path) {
      setModelPath(selectedDownloadedModel.local_path);
    }
    setMessage("");
  }

  function chooseDownloadedModel(modelId: string) {
    setSelectedModel(modelId);
    const model = downloadedModels.find((profile) => profile.id === modelId);
    if (model?.local_path && modelSource === "path") {
      setModelPath(model.local_path);
    }
  }

  function stopGeneration() {
    abortRef.current?.abort();
  }

  function clearConversation() {
    abortRef.current?.abort();
    setTurns([]);
    setAttachments([]);
    setMessage("");
  }

  async function sendPrompt() {
    const content = prompt.trim();
    const pendingAttachments = [...attachments];
    const hasMedia = pendingAttachments.length > 0;

    if (!inferenceReady) {
      setMessage("Install MLX inference before chatting.");
      return;
    }
    if (hasMedia && !multimodalReady) {
      setMessage(
        "Image and audio chat require `mlx-vlm`. Click Install MLX or run `gemma4-mlx-mac install-mlx`.",
      );
      return;
    }
    if (!activeModel) {
      setMessage(
        modelSource === "path"
          ? "Enter a local model path first."
          : "Download a model in the Models tab before chatting.",
      );
      return;
    }
    if (!content && !hasMedia) {
      setMessage("Enter a prompt or attach an image/audio file first.");
      return;
    }

    const effectivePrompt =
      content ||
      (pendingAttachments.some((item) => item.kind === "audio")
        ? "Listen to the attached audio and describe or transcribe it."
        : "Describe the attached image.");

    const temperature = parsePositiveFloat(temperatureInput, 0.7, 0, 2);
    const topP = parsePositiveFloat(topPInput, 0.95, 0.05, 1);
    const maxTokens = parsePositiveInt(maxTokensInput, 1024, 1, 32768);

    const userTurnId = `user-${Date.now()}`;
    const assistantTurnId = `assistant-${Date.now()}`;
    const historyMessages: Array<{
      role: "system" | "user" | "assistant";
      content: string | Array<Record<string, unknown>>;
    }> = [];
    if (systemPrompt.trim()) {
      historyMessages.push({ role: "system", content: systemPrompt.trim() });
    }
    for (const turn of turns) {
      if (turn.content.trim() || (turn.attachments && turn.attachments.length > 0)) {
        historyMessages.push({
          role: turn.role,
          content: buildOpenAiContentParts(turn.content, turn.attachments),
        });
      }
    }
    historyMessages.push({
      role: "user",
      content: buildOpenAiContentParts(effectivePrompt, pendingAttachments),
    });

    const controller = new AbortController();
    abortRef.current = controller;

    setIsSending(true);
    setMessage("");
    setPrompt("");
    setAttachments([]);
    setTurns((prev) => [
      ...prev,
      {
        id: userTurnId,
        role: "user",
        content: effectivePrompt,
        attachments: pendingAttachments.length > 0 ? pendingAttachments : undefined,
      },
      {
        id: assistantTurnId,
        role: "assistant",
        content: "",
        stats: { completionTokens: 0, tokensPerSecond: 0, elapsedSeconds: 0 },
      },
    ]);

    let receivedAnyToken = false;
    try {
      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          model: activeModel,
          messages: historyMessages,
          max_tokens: maxTokens,
          temperature,
          top_p: topP,
          show_thinking: showThinking,
          stream: true,
        }),
      });
      if (!response.ok) {
        throw new Error(await readErrorMessage(response, "Chat request failed."));
      }

      if (!response.body) {
        const data = await response.json();
        const text = data.choices?.[0]?.message?.content ?? JSON.stringify(data, null, 2);
        receivedAnyToken = true;
        if (mountedRef.current) {
          setTurns((prev) =>
            prev.map((turn) => (turn.id === assistantTurnId ? { ...turn, content: text } : turn)),
          );
        }
        return;
      }

      let generatedText = "";
      let fallbackTokens = 0;
      const startedAt = performance.now();
      await readChatStream(response, (payload) => {
        if (payload.error) {
          throw new Error(payload.error.message);
        }
        const delta = payload.choices?.[0]?.delta?.content ?? "";
        if (delta) {
          receivedAnyToken = true;
          generatedText += delta;
          fallbackTokens += 1;
        }
        const elapsedSeconds = Math.max((performance.now() - startedAt) / 1000, 0.001);
        const completionTokens = payload.metrics?.completion_tokens ?? fallbackTokens;
        const stats = {
          completionTokens,
          tokensPerSecond: payload.metrics?.tokens_per_second ?? completionTokens / elapsedSeconds,
          elapsedSeconds: payload.metrics?.elapsed_seconds ?? elapsedSeconds,
        };
        if (mountedRef.current) {
          setTurns((prev) =>
            prev.map((turn) =>
              turn.id === assistantTurnId ? { ...turn, content: generatedText, stats } : turn,
            ),
          );
        }
      });
    } catch (error: unknown) {
      if (!mountedRef.current) {
        return;
      }
      if (error instanceof DOMException && error.name === "AbortError") {
        setTurns((prev) =>
          prev.map((turn) => (turn.id === assistantTurnId ? { ...turn, stopped: true } : turn)),
        );
      } else {
        setMessage(error instanceof Error ? error.message : "Chat request failed.");
        if (!receivedAnyToken) {
          setPrompt(content);
          setAttachments(pendingAttachments);
          setTurns((prev) =>
            prev.filter((turn) => turn.id !== userTurnId && turn.id !== assistantTurnId),
          );
        }
      }
    } finally {
      abortRef.current = null;
      if (mountedRef.current) {
        setIsSending(false);
        loadInferenceStatus();
      }
    }
  }

  function handleComposerKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      if (!isSending && activeModel && inferenceReady) {
        sendPrompt();
      }
    }
  }

  const showNoModelsHint =
    modelSource === "downloaded" && models.length > 0 && downloadedModels.length === 0;
  const needsMlxInstall = inferenceStatus && (!inferenceReady || !multimodalReady);

  return (
    <>
      <PageHeader title="Chat" eyebrow="Local text, vision & audio inference">
        {inferenceStatus?.active_adapter && (
          <StatusBadge tone="good">Adapter: {inferenceStatus.active_adapter}</StatusBadge>
        )}
        {multimodalReady && <StatusBadge tone="good">Vision + Audio</StatusBadge>}
        <StatusBadge tone={inferenceReady ? "good" : "warn"}>
          {inferenceReady ? "MLX ready" : "MLX setup"}
        </StatusBadge>
        {turns.length > 0 && (
          <button
            type="button"
            className="secondary compact-button"
            onClick={clearConversation}
            title="Start a new chat"
          >
            <RotateCcw size={15} />
            <span>New chat</span>
          </button>
        )}
      </PageHeader>

      <div className="chat-layout">
        <section
          className={`conversation-panel${isDragOver ? " drag-over" : ""}`}
          aria-label="Conversation"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {message && <p className="notice">{message}</p>}
          {!message && showNoModelsHint && (
            <p className="notice">
              Download a Gemma 4 model in the Models tab or switch Model source to Path.
            </p>
          )}
          <div className="chat-thread">
            {turns.length === 0 && (
              <div className="empty-state">
                <MessageSquareText size={28} />
                <strong>Ready</strong>
                <small>
                  Multi-turn local chat with Gemma 4. Drop or paste images/audio, press Enter to
                  send, Shift+Enter for newline.
                </small>
              </div>
            )}
            {turns.map((turn, idx) => {
              const isStreamingTurn =
                isSending && idx === turns.length - 1 && turn.role === "assistant";
              if (turn.role === "user") {
                return (
                  <article className="message user-message" key={turn.id}>
                    <span>User</span>
                    {turn.attachments && turn.attachments.length > 0 && (
                      <div className="turn-attachments">
                        {turn.attachments.map((item) =>
                          item.kind === "image" ? (
                            <img
                              key={item.id}
                              src={item.dataUrl}
                              alt={item.name}
                              className="turn-image-preview"
                            />
                          ) : (
                            <div key={item.id} className="turn-audio-chip">
                              <FileAudio size={15} />
                              <span>{item.name}</span>
                              <audio controls src={item.dataUrl} preload="metadata" />
                            </div>
                          ),
                        )}
                      </div>
                    )}
                    <p>{turn.content}</p>
                  </article>
                );
              }
              return (
                <article className="message assistant-message" key={turn.id}>
                  <div className="message-meta">
                    <span>Gemma{turn.stopped ? " (stopped)" : ""}</span>
                    {turn.stats && (
                      <div className="generation-stats">
                        <span>{turn.stats.completionTokens} tokens</span>
                        <span>{formatTokensPerSecond(turn.stats.tokensPerSecond)} tok/s</span>
                      </div>
                    )}
                  </div>
                  {turn.content ? (
                    <MarkdownMessage content={turn.content} streaming={isStreamingTurn} />
                  ) : isStreamingTurn ? (
                    <p className="pending-copy">Waiting for first token...</p>
                  ) : (
                    <p className="pending-copy">(No output)</p>
                  )}
                </article>
              );
            })}
            <div ref={threadEndRef} />
          </div>

          <div className="composer">
            {attachments.length > 0 && (
              <div className="attachment-tray" aria-label="Attached media">
                {attachments.map((item) => (
                  <div className="attachment-chip" key={item.id}>
                    {item.kind === "image" ? (
                      <img src={item.dataUrl} alt={item.name} className="attachment-thumb" />
                    ) : (
                      <FileAudio size={18} />
                    )}
                    <div className="attachment-meta">
                      <strong>{item.name}</strong>
                      <small>{formatBytes(item.sizeBytes)}</small>
                    </div>
                    <button
                      type="button"
                      className="attachment-remove"
                      onClick={() => removeAttachment(item.id)}
                      aria-label={`Remove ${item.name}`}
                    >
                      <X size={14} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            <input
              ref={imageInputRef}
              type="file"
              accept="image/*"
              multiple
              hidden
              onChange={(event) => {
                if (event.target.files) {
                  addFiles(event.target.files);
                }
                event.target.value = "";
              }}
            />
            <input
              ref={audioInputRef}
              type="file"
              accept="audio/*"
              multiple
              hidden
              onChange={(event) => {
                if (event.target.files) {
                  addFiles(event.target.files);
                }
                event.target.value = "";
              }}
            />

            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={handleComposerKeyDown}
              onPaste={handlePaste}
              placeholder="Ask Gemma 4 anything, or paste/drop an image or audio clip..."
            />
            <div className="chat-actions">
              <div className="composer-left">
                <button
                  type="button"
                  className="secondary compact-button"
                  disabled={isSending}
                  onClick={() => imageInputRef.current?.click()}
                  title="Attach image (or paste/drop)"
                >
                  <ImagePlus size={16} />
                  <span>Image</span>
                </button>
                <button
                  type="button"
                  className="secondary compact-button"
                  disabled={isSending || !supportsAudio}
                  onClick={() => audioInputRef.current?.click()}
                  title={
                    supportsAudio
                      ? "Attach audio clip (E2B/E4B/12B any-to-any)"
                      : "Audio requires an any-to-any Gemma 4 model (E2B, E4B, or 12B)"
                  }
                >
                  <FileAudio size={16} />
                  <span>Audio</span>
                </button>
                <span className="active-model-label">{activeModelLabel}</span>
              </div>
              <div className="action-buttons">
                {isSending && (
                  <button
                    type="button"
                    className="secondary icon-button"
                    onClick={stopGeneration}
                    title="Stop generation"
                  >
                    <Square size={16} />
                    <span>Stop</span>
                  </button>
                )}
                <button
                  type="button"
                  className="primary icon-button"
                  disabled={isSending || !activeModel || !inferenceReady}
                  data-busy={isSending || undefined}
                  onClick={sendPrompt}
                  title="Send"
                >
                  <Send size={18} />
                  <span>{isSending ? "Streaming" : "Send"}</span>
                </button>
              </div>
            </div>
          </div>
        </section>

        <aside className="control-stack" aria-label="Chat controls">
          <section className="control-panel" aria-label="Model and response controls">
            <div className="panel-block">
              <div className="block-heading">
                <HardDrive size={18} />
                <strong>Model</strong>
              </div>
              <div className="model-picker">
                <div className="source-toggle" aria-label="Model source">
                  <button
                    className={modelSource === "downloaded" ? "active" : ""}
                    onClick={() => chooseModelSource("downloaded")}
                    type="button"
                  >
                    Downloaded
                  </button>
                  <button
                    className={modelSource === "path" ? "active" : ""}
                    onClick={() => chooseModelSource("path")}
                    type="button"
                  >
                    Path
                  </button>
                </div>
                {modelSource === "downloaded" ? (
                  <label className="field">
                    <span>Profile</span>
                    <select
                      value={selectedModel}
                      onChange={(event) => chooseDownloadedModel(event.target.value)}
                      disabled={downloadedModels.length === 0 || isSending}
                    >
                      {downloadedModels.length === 0 ? (
                        <option value="">No downloaded models</option>
                      ) : (
                        downloadedModels.map((model) => (
                          <option key={model.id} value={model.id}>
                            {model.label}
                          </option>
                        ))
                      )}
                    </select>
                    {selectedDownloadedModel?.local_path && (
                      <small>{selectedDownloadedModel.local_path}</small>
                    )}
                  </label>
                ) : (
                  <label className="field">
                    <span>Path</span>
                    <input
                      value={modelPath}
                      onChange={(event) => setModelPath(event.target.value)}
                      disabled={isSending}
                      list="downloaded-model-paths"
                      placeholder="/Users/me/.cache/huggingface/.../snapshots/..."
                    />
                    <datalist id="downloaded-model-paths">
                      {downloadedModels
                        .filter((model) => model.local_path)
                        .map((model) => (
                          <option key={model.id} value={model.local_path ?? ""}>
                            {model.label}
                          </option>
                        ))}
                    </datalist>
                  </label>
                )}
              </div>
            </div>

            <div className="panel-block">
              <div className="block-heading">
                <Activity size={18} />
                <strong>Thinking Mode</strong>
              </div>
              <label className="switch-row">
                <span className="switch-copy">
                  <strong>Native Thinking</strong>
                  <small>Enable Gemma 4 thought channel and show collapsible reasoning.</small>
                </span>
                <input
                  aria-label="Enable Gemma 4 Thinking Mode"
                  checked={showThinking}
                  className="switch-control"
                  disabled={isSending}
                  onChange={(event) => setShowThinking(event.target.checked)}
                  type="checkbox"
                />
              </label>
            </div>

            <div className="panel-block">
              <div className="block-heading">
                <SlidersHorizontal size={18} />
                <strong>Parameters</strong>
              </div>
              <div className="form-grid">
                <label className="field">
                  <span>Temp</span>
                  <input
                    type="number"
                    min={0}
                    max={2}
                    step={0.1}
                    value={temperatureInput}
                    disabled={isSending}
                    onChange={(event) => setTemperatureInput(event.target.value)}
                  />
                </label>
                <label className="field">
                  <span>Max tokens</span>
                  <input
                    type="number"
                    min={1}
                    max={32768}
                    step={64}
                    value={maxTokensInput}
                    disabled={isSending}
                    onChange={(event) => setMaxTokensInput(event.target.value)}
                  />
                </label>
              </div>
              <label className="field">
                <span>Top-p</span>
                <input
                  type="number"
                  min={0.05}
                  max={1}
                  step={0.05}
                  value={topPInput}
                  disabled={isSending}
                  onChange={(event) => setTopPInput(event.target.value)}
                />
              </label>
              <label className="field">
                <span>System prompt</span>
                <textarea
                  className="compact-textarea"
                  rows={2}
                  value={systemPrompt}
                  disabled={isSending}
                  onChange={(event) => setSystemPrompt(event.target.value)}
                  placeholder="Optional system instructions..."
                />
              </label>
            </div>

            {needsMlxInstall && (
              <InferenceSetup
                status={inferenceStatus}
                job={installJob}
                isInstalling={Boolean(isInstalling)}
                onInstall={startInstall}
              />
            )}
          </section>
          <ModelMemoryCard
            memory={inferenceStatus?.model_memory ?? null}
            onUnload={unloadModels}
            isUnloading={isUnloading}
          />
        </aside>
      </div>
    </>
  );
}

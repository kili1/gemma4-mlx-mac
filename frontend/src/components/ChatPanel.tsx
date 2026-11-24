import React from "react";
import {
  ArrowRight,
  Brain,
  Check,
  Copy,
  FileAudio,
  HardDrive,
  ImagePlus,
  MessageSquarePlus,
  PanelRightClose,
  PanelRightOpen,
  RefreshCw,
  Send,
  SlidersHorizontal,
  Square,
  Trash2,
  X,
} from "lucide-react";

import { fetchJson, readChatStream, readErrorMessage } from "../api";
import { formatBytes, formatTokensPerSecond, parsePositiveFloat, parsePositiveInt } from "../format";
import { buildOpenAiContentParts, readFileAsAttachment } from "../media";
import { loadChatSessions, saveChatSessions, summarizeSessionTitle } from "../storage";
import type {
  ChatSession,
  ChatTurn,
  InferenceStatus,
  MediaAttachment,
  MlxInstallJob,
  ModelProfile,
  ModelSource,
  View,
} from "../types";
import { InferenceSetup, ModelMemoryCard, NoticeBanner, PageHeader, StatusBadge } from "./Common";
import { MarkdownMessage } from "./MarkdownMessage";

const STARTER_PROMPTS: Array<{
  title: string;
  subtitle: string;
  prompt: string;
  thinking?: boolean;
}> = [
  {
    title: "Compare MoE vs dense",
    subtitle: "Why Gemma 4 26B-A4B runs fast on Apple Silicon",
    prompt:
      "Compare Gemma 4 26B-A4B MoE vs dense 31B for local Apple Silicon inference in a concise table.",
  },
  {
    title: "Reason with Thinking Mode",
    subtitle: "Step-by-step native Gemma 4 thought channel",
    prompt: "A 16 GB Mac loads a 4.5 GB model and KV cache grows at 12 MB per 1K tokens. How much headroom remains at 64K context?",
    thinking: true,
  },
  {
    title: "Write a clean Python CLI",
    subtitle: "Code generation with syntax copy block",
    prompt: "Write a minimal Python 3.11 script using pathlib to find the 10 largest files in a folder and print human-readable sizes.",
  },
  {
    title: "Local architecture review",
    subtitle: "Or drop/paste any screenshot or audio clip",
    prompt: "Give me a 5-point checklist for designing a local-only AI desktop tool that protects user privacy.",
  },
];

export function ChatPanel({
  activeModelHint,
  totalSystemGb,
  onNavigate,
  onStatusChange,
  isVisible,
}: {
  activeModelHint?: string | null;
  totalSystemGb?: number;
  onNavigate: (view: View) => void;
  onStatusChange?: (status: InferenceStatus) => void;
  isVisible: boolean;
}) {
  const [models, setModels] = React.useState<ModelProfile[]>([]);
  const [selectedModel, setSelectedModel] = React.useState("");
  const [modelSource, setModelSource] = React.useState<ModelSource>("downloaded");
  const [modelPath, setModelPath] = React.useState("");
  const [prompt, setPrompt] = React.useState("");
  const [attachments, setAttachments] = React.useState<MediaAttachment[]>([]);
  const [isDragOver, setIsDragOver] = React.useState(false);
  const [inspectorOpen, setInspectorOpen] = React.useState(true);

  const [systemPrompt, setSystemPrompt] = React.useState("");
  const [temperature, setTemperature] = React.useState(0.7);
  const [topP, setTopP] = React.useState(0.95);
  const [maxTokensInput, setMaxTokensInput] = React.useState("1024");
  const [showThinking, setShowThinking] = React.useState(false);

  const [sessions, setSessions] = React.useState<ChatSession[]>(() => loadChatSessions());
  const [activeSessionId, setActiveSessionId] = React.useState<string>(() => {
    const loaded = loadChatSessions();
    return loaded[0]?.id ?? `session-${Date.now()}`;
  });
  const [turns, setTurns] = React.useState<ChatTurn[]>(() => loadChatSessions()[0]?.turns ?? []);

  const [copiedTurnId, setCopiedTurnId] = React.useState<string | null>(null);
  const [message, setMessage] = React.useState("");
  const [isSending, setIsSending] = React.useState(false);
  const [isUnloading, setIsUnloading] = React.useState(false);
  const [inferenceStatus, setInferenceStatus] = React.useState<InferenceStatus | null>(null);
  const [installJob, setInstallJob] = React.useState<MlxInstallJob | null>(null);

  const mountedRef = React.useRef(true);
  const pollTimerRef = React.useRef<number | null>(null);
  const abortRef = React.useRef<AbortController | null>(null);
  const threadEndRef = React.useRef<HTMLDivElement | null>(null);
  const textareaRef = React.useRef<HTMLTextAreaElement | null>(null);
  const imageInputRef = React.useRef<HTMLInputElement | null>(null);
  const audioInputRef = React.useRef<HTMLInputElement | null>(null);

  const downloadedModels = models.filter((model) => model.downloaded);
  const selectedDownloadedModel =
    downloadedModels.find((model) => model.id === selectedModel) ?? null;
  const activeModel =
    modelSource === "path" ? modelPath.trim() : selectedDownloadedModel?.id ?? "";
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

  // Refresh models/status whenever the user switches back to Chat (e.g. after downloading or activating an adapter).
  React.useEffect(() => {
    if (isVisible) {
      loadModels();
      loadInferenceStatus();
    }
  }, [isVisible]);

  // Honor cross-tab "Chat with this model" navigation from ModelsPanel.
  React.useEffect(() => {
    if (activeModelHint) {
      setModelSource("downloaded");
      setSelectedModel(activeModelHint);
    }
  }, [activeModelHint]);

  // Auto-grow composer textarea smoothly.
  React.useEffect(() => {
    const el = textareaRef.current;
    if (!el) {
      return;
    }
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [prompt]);

  React.useEffect(() => {
    threadEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  // Persist current session to localStorage whenever turns settle.
  React.useEffect(() => {
    if (turns.length === 0) {
      return;
    }
    const firstUserTurn = turns.find((t) => t.role === "user");
    const title = summarizeSessionTitle(firstUserTurn?.content ?? "New chat");
    setSessions((prev) => {
      const updated: ChatSession = {
        id: activeSessionId,
        title,
        model: activeModel,
        updatedAt: Date.now(),
        turns,
      };
      const next = [updated, ...prev.filter((item) => item.id !== activeSessionId)];
      saveChatSessions(next);
      return next;
    });
  }, [turns, activeSessionId, activeModel]);

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
        setSelectedModel((prev) => {
          if (prev && availableModels.some((item) => item.id === prev)) {
            return prev;
          }
          return defaultModel.id;
        });
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
      onStatusChange?.(status);
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

  function startNewSession() {
    abortRef.current?.abort();
    const freshId = `session-${Date.now()}`;
    setActiveSessionId(freshId);
    setTurns([]);
    setAttachments([]);
    setMessage("");
    textareaRef.current?.focus();
  }

  function selectSession(sessionId: string) {
    abortRef.current?.abort();
    const target = sessions.find((item) => item.id === sessionId);
    if (!target) {
      return;
    }
    setActiveSessionId(target.id);
    setTurns(target.turns);
    setAttachments([]);
    setMessage("");
  }

  function deleteActiveSession() {
    abortRef.current?.abort();
    const remaining = sessions.filter((item) => item.id !== activeSessionId);
    saveChatSessions(remaining);
    setSessions(remaining);
    if (remaining[0]) {
      setActiveSessionId(remaining[0].id);
      setTurns(remaining[0].turns);
    } else {
      setActiveSessionId(`session-${Date.now()}`);
      setTurns([]);
    }
    setAttachments([]);
  }

  async function copyTurn(turn: ChatTurn) {
    try {
      await navigator.clipboard.writeText(turn.content);
      setCopiedTurnId(turn.id);
      window.setTimeout(() => setCopiedTurnId((curr) => (curr === turn.id ? null : curr)), 1500);
    } catch {
      // Clipboard may be blocked.
    }
  }

  function stopGeneration() {
    abortRef.current?.abort();
  }

  async function runCompletion({
    baseTurns,
    userText,
    pendingAttachments,
    thinkingOverride,
  }: {
    baseTurns: ChatTurn[];
    userText: string;
    pendingAttachments: MediaAttachment[];
    thinkingOverride?: boolean;
  }) {
    const useThinking = thinkingOverride ?? showThinking;
    const hasMedia = pendingAttachments.length > 0;
    if (!inferenceReady) {
      setMessage("Install MLX inference before chatting.");
      return;
    }
    if (hasMedia && !multimodalReady) {
      setMessage("Image and audio chat require `mlx-vlm`. Click Install MLX first.");
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

    const effectivePrompt =
      userText.trim() ||
      (pendingAttachments.some((item) => item.kind === "audio")
        ? "Listen to the attached audio and describe or transcribe it."
        : "Describe the attached image.");

    const maxTokens = parsePositiveInt(maxTokensInput, 1024, 1, 32768);
    const clampedTemp = parsePositiveFloat(String(temperature), 0.7, 0, 2);
    const clampedTopP = parsePositiveFloat(String(topP), 0.95, 0.05, 1);

    const userTurnId = `user-${Date.now()}`;
    const assistantTurnId = `assistant-${Date.now()}`;

    const historyMessages: Array<{
      role: "system" | "user" | "assistant";
      content: string | Array<Record<string, unknown>>;
    }> = [];
    if (systemPrompt.trim()) {
      historyMessages.push({ role: "system", content: systemPrompt.trim() });
    }
    for (const turn of baseTurns) {
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
    setTurns([
      ...baseTurns,
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
          temperature: clampedTemp,
          top_p: clampedTopP,
          show_thinking: useThinking,
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
          setPrompt(userText);
          setAttachments(pendingAttachments);
          setTurns(baseTurns);
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

  function sendPrompt() {
    if (!prompt.trim() && attachments.length === 0) {
      setMessage("Enter a prompt or attach an image/audio file first.");
      return;
    }
    runCompletion({
      baseTurns: turns,
      userText: prompt,
      pendingAttachments: [...attachments],
    });
  }

  function regenerateLastReply() {
    if (isSending || turns.length < 2) {
      return;
    }
    const lastTurn = turns[turns.length - 1];
    const prevUser = turns[turns.length - 2];
    if (lastTurn.role !== "assistant" || prevUser.role !== "user") {
      return;
    }
    runCompletion({
      baseTurns: turns.slice(0, -2),
      userText: prevUser.content,
      pendingAttachments: prevUser.attachments ?? [],
    });
  }

  function handleStarterClick(starter: (typeof STARTER_PROMPTS)[number]) {
    if (starter.thinking) {
      setShowThinking(true);
    }
    if (inferenceReady && activeModel) {
      runCompletion({
        baseTurns: [],
        userText: starter.prompt,
        pendingAttachments: [],
        thinkingOverride: starter.thinking ?? showThinking,
      });
    } else {
      setPrompt(starter.prompt);
      textareaRef.current?.focus();
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
      <PageHeader title="Chat" eyebrow="Local text, vision & audio">
        {inferenceStatus?.active_adapter && (
          <StatusBadge tone="good">Adapter: {inferenceStatus.active_adapter}</StatusBadge>
        )}
        {multimodalReady && <StatusBadge tone="good">Vision + Audio</StatusBadge>}
        <StatusBadge tone={inferenceReady ? "good" : "warn"}>
          {inferenceReady ? "MLX ready" : "MLX setup"}
        </StatusBadge>
        <button
          type="button"
          className="secondary compact-button"
          onClick={() => setInspectorOpen((open) => !open)}
          title={inspectorOpen ? "Hide inspector" : "Show inspector"}
        >
          {inspectorOpen ? <PanelRightClose size={15} /> : <PanelRightOpen size={15} />}
          <span>{inspectorOpen ? "Hide panel" : "Controls"}</span>
        </button>
      </PageHeader>

      {message && <NoticeBanner message={message} onDismiss={() => setMessage("")} />}
      {!message && showNoModelsHint && (
        <NoticeBanner
          message="No Gemma 4 models are downloaded yet."
          action={
            <button
              type="button"
              className="secondary compact-button"
              onClick={() => onNavigate("models")}
            >
              <span>Download a model</span>
              <ArrowRight size={14} />
            </button>
          }
        />
      )}

      <div className={`chat-layout${inspectorOpen ? "" : " inspector-collapsed"}`}>
        <section
          className={`conversation-panel${isDragOver ? " drag-over" : ""}`}
          aria-label="Conversation"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="chat-topbar">
            <div className="session-controls">
              <select
                aria-label="Chat history"
                className="session-select"
                value={sessions.some((s) => s.id === activeSessionId) ? activeSessionId : "__new__"}
                onChange={(event) => {
                  if (event.target.value === "__new__") {
                    startNewSession();
                  } else {
                    selectSession(event.target.value);
                  }
                }}
              >
                {!sessions.some((s) => s.id === activeSessionId) && (
                  <option value="__new__">New conversation</option>
                )}
                {sessions.map((session) => (
                  <option key={session.id} value={session.id}>
                    {session.title}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="secondary compact-button"
                onClick={startNewSession}
                title="Start a new chat"
              >
                <MessageSquarePlus size={14} />
                <span>New</span>
              </button>
              {sessions.some((s) => s.id === activeSessionId) && (
                <button
                  type="button"
                  className="ghost-button"
                  onClick={deleteActiveSession}
                  title="Delete conversation"
                >
                  <Trash2 size={14} />
                </button>
              )}
            </div>
            <span className="status-badge neutral tabular">
              {selectedDownloadedModel?.label ?? (activeModel || "No model selected")}
            </span>
          </div>

          <div className="chat-thread">
            {turns.length === 0 && (
              <div className="empty-state">
                <strong>Local Gemma 4 on Apple Silicon</strong>
                <small>
                  100% private unified-memory chat. Paste screenshots with ⌘V, drag-and-drop images
                  or audio clips, or try a starter prompt below.
                </small>
                <div className="starter-grid">
                  {STARTER_PROMPTS.map((starter) => (
                    <button
                      type="button"
                      key={starter.title}
                      className="starter-chip"
                      onClick={() => handleStarterClick(starter)}
                    >
                      <strong>{starter.title}</strong>
                      <span>{starter.subtitle}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {turns.map((turn, idx) => {
              const isLastTurn = idx === turns.length - 1;
              const isStreamingTurn = isSending && isLastTurn && turn.role === "assistant";
              if (turn.role === "user") {
                return (
                  <article className="message user-message" key={turn.id}>
                    <span>You</span>
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
                    <span>Gemma 4{turn.stopped ? " (stopped)" : ""}</span>
                    <div className="message-actions">
                      {turn.stats && (
                        <div className="generation-stats">
                          <span>{turn.stats.completionTokens} tok</span>
                          <span>{formatTokensPerSecond(turn.stats.tokensPerSecond)} tok/s</span>
                        </div>
                      )}
                      {turn.content && !isStreamingTurn && (
                        <button
                          type="button"
                          className="ghost-button"
                          onClick={() => copyTurn(turn)}
                          title="Copy response"
                        >
                          {copiedTurnId === turn.id ? <Check size={13} /> : <Copy size={13} />}
                          <span>{copiedTurnId === turn.id ? "Copied" : "Copy"}</span>
                        </button>
                      )}
                      {isLastTurn && !isSending && (
                        <button
                          type="button"
                          className="ghost-button"
                          onClick={regenerateLastReply}
                          title="Regenerate response"
                        >
                          <RefreshCw size={13} />
                          <span>Retry</span>
                        </button>
                      )}
                    </div>
                  </div>
                  {turn.content ? (
                    <MarkdownMessage content={turn.content} streaming={isStreamingTurn} />
                  ) : isStreamingTurn ? (
                    <div className="assistant-body">
                      <p className="pending-copy">Waiting for first token…</p>
                    </div>
                  ) : (
                    <div className="assistant-body">
                      <p className="pending-copy">(No output)</p>
                    </div>
                  )}
                </article>
              );
            })}
            <div ref={threadEndRef} />
          </div>

          <div className="composer">
            <div className="composer-box">
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
                ref={textareaRef}
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                onPaste={handlePaste}
                placeholder="Ask Gemma 4 anything, or paste/drop an image or audio clip… (Enter to send)"
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
                    <ImagePlus size={15} />
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
                        : "Audio requires E2B, E4B, or 12B any-to-any profile"
                    }
                  >
                    <FileAudio size={15} />
                    <span>Audio</span>
                  </button>
                  <button
                    type="button"
                    className={`pill-toggle${showThinking ? " active" : ""}`}
                    disabled={isSending}
                    onClick={() => setShowThinking((prev) => !prev)}
                    aria-pressed={showThinking}
                    title="Toggle native Gemma 4 Thinking Mode"
                  >
                    <Brain size={14} />
                    <span>Thinking</span>
                  </button>
                </div>

                <div className="action-buttons">
                  {isSending && (
                    <button
                      type="button"
                      className="secondary compact-button"
                      onClick={stopGeneration}
                      title="Stop generation"
                    >
                      <Square size={14} />
                      <span>Stop</span>
                    </button>
                  )}
                  <button
                    type="button"
                    className="primary compact-button"
                    disabled={isSending || !activeModel || !inferenceReady}
                    data-busy={isSending || undefined}
                    onClick={sendPrompt}
                    title="Send prompt"
                  >
                    <Send size={15} />
                    <span>{isSending ? "Streaming" : "Send"}</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>

        {inspectorOpen && (
          <aside className="control-stack" aria-label="Chat controls">
            <section className="control-panel" aria-label="Model and sampling controls">
              <div className="panel-block">
                <div className="block-heading">
                  <HardDrive size={16} />
                  <strong>Model</strong>
                </div>
                <div className="model-picker">
                  <div className="source-toggle" aria-label="Model source">
                    <button
                      className={modelSource === "downloaded" ? "active" : ""}
                      onClick={() => setModelSource("downloaded")}
                      type="button"
                    >
                      Downloaded
                    </button>
                    <button
                      className={modelSource === "path" ? "active" : ""}
                      onClick={() => {
                        setModelSource("path");
                        if (!modelPath && selectedDownloadedModel?.local_path) {
                          setModelPath(selectedDownloadedModel.local_path);
                        }
                      }}
                      type="button"
                    >
                      Local path
                    </button>
                  </div>
                  {modelSource === "downloaded" ? (
                    <label className="field">
                      <span>Profile</span>
                      <select
                        value={selectedModel}
                        onChange={(event) => setSelectedModel(event.target.value)}
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
                      <span>Snapshot directory</span>
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
                  <SlidersHorizontal size={16} />
                  <strong>Sampling</strong>
                </div>

                <label className="field">
                  <div className="field-label-row">
                    <span>Temperature</span>
                    <small className="tabular">{temperature.toFixed(2)}</small>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={2}
                    step={0.05}
                    value={temperature}
                    disabled={isSending}
                    onChange={(event) => setTemperature(Number(event.target.value))}
                  />
                </label>

                <label className="field">
                  <div className="field-label-row">
                    <span>Top-p</span>
                    <small className="tabular">{topP.toFixed(2)}</small>
                  </div>
                  <input
                    type="range"
                    min={0.1}
                    max={1}
                    step={0.05}
                    value={topP}
                    disabled={isSending}
                    onChange={(event) => setTopP(Number(event.target.value))}
                  />
                </label>

                <label className="field">
                  <span>Max output tokens</span>
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

                <label className="field">
                  <span>System prompt</span>
                  <textarea
                    className="compact-textarea"
                    rows={2}
                    value={systemPrompt}
                    disabled={isSending}
                    onChange={(event) => setSystemPrompt(event.target.value)}
                    placeholder="Optional system instructions…"
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
              totalSystemGb={totalSystemGb}
              onUnload={unloadModels}
              isUnloading={isUnloading}
            />
          </aside>
        )}
      </div>
    </>
  );
}

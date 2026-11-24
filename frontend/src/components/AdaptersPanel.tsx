import React from "react";
import { Activity, ArrowRight, CheckCircle2, RefreshCw, XCircle } from "lucide-react";

import { fetchJson } from "../api";
import type { AdapterInfo, View } from "../types";
import { NoticeBanner, PageHeader, StatusBadge } from "./Common";

export function AdaptersPanel({
  onNavigate,
  onOpenChat,
}: {
  onNavigate: (view: View) => void;
  onOpenChat: () => void;
}) {
  const [adapters, setAdapters] = React.useState<AdapterInfo[]>([]);
  const [busyId, setBusyId] = React.useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = React.useState(false);
  const [message, setMessage] = React.useState("");
  const [noticeTone, setNoticeTone] = React.useState<"warn" | "good">("good");

  React.useEffect(() => {
    refreshAdapters();
  }, []);

  async function refreshAdapters() {
    setIsRefreshing(true);
    try {
      const data = await fetchJson<{ adapters: AdapterInfo[] }>(
        "/api/adapters",
        undefined,
        "Could not load adapters.",
      );
      setAdapters(data.adapters);
    } catch (error: unknown) {
      setNoticeTone("warn");
      setMessage(error instanceof Error ? error.message : "Could not load adapters.");
    } finally {
      setIsRefreshing(false);
    }
  }

  async function activateAdapter(adapterId: string) {
    setBusyId(adapterId);
    setMessage("");
    try {
      await fetchJson(
        `/api/adapters/${encodeURIComponent(adapterId)}/activate`,
        { method: "POST" },
        "Could not activate adapter.",
      );
      setNoticeTone("good");
      setMessage(`Activated adapter "${adapterId}". New chat turns will load it automatically.`);
      await refreshAdapters();
    } catch (error: unknown) {
      setNoticeTone("warn");
      setMessage(error instanceof Error ? error.message : "Could not activate adapter.");
    } finally {
      setBusyId(null);
    }
  }

  async function deactivateAdapter() {
    setBusyId("__deactivate__");
    setMessage("");
    try {
      await fetchJson(
        "/api/adapters/deactivate",
        { method: "POST" },
        "Could not deactivate adapter.",
      );
      setNoticeTone("good");
      setMessage("Deactivated adapter. Chat will use base model weights.");
      await refreshAdapters();
    } catch (error: unknown) {
      setNoticeTone("warn");
      setMessage(error instanceof Error ? error.message : "Could not deactivate adapter.");
    } finally {
      setBusyId(null);
    }
  }

  const activeAdapter = adapters.find((adapter) => adapter.active) ?? null;

  return (
    <>
      <PageHeader title="Adapters" eyebrow="Local LoRA / QLoRA outputs">
        <StatusBadge tone={activeAdapter ? "good" : "neutral"}>
          {activeAdapter ? `Active: ${activeAdapter.id}` : `${adapters.length} found`}
        </StatusBadge>
        {activeAdapter && (
          <button
            type="button"
            className="primary compact-button"
            onClick={onOpenChat}
          >
            <span>Test in Chat</span>
            <ArrowRight size={14} />
          </button>
        )}
        <button
          type="button"
          className="secondary compact-button"
          disabled={isRefreshing}
          data-busy={isRefreshing || undefined}
          onClick={refreshAdapters}
        >
          <RefreshCw size={14} />
          <span>Refresh</span>
        </button>
      </PageHeader>
      {message && (
        <NoticeBanner message={message} tone={noticeTone} onDismiss={() => setMessage("")} />
      )}
      <div className="model-list">
        {adapters.length === 0 ? (
          <div className="panel empty-state">
            <Activity size={26} />
            <strong>No local adapters yet</strong>
            <small>Train a LoRA or QLoRA adapter to save weights under ./adapters.</small>
            <button
              type="button"
              className="primary compact-button"
              onClick={() => onNavigate("tune")}
            >
              <span>Start a Fine-tune</span>
              <ArrowRight size={14} />
            </button>
          </div>
        ) : (
          adapters.map((adapter) => (
            <div className="model-card" key={adapter.id}>
              <div className="model-row">
                <div>
                  <div className="model-title">
                    <strong>{adapter.id}</strong>
                    <div className="badges">
                      {adapter.active && <span className="good">Active</span>}
                      <span>{adapter.has_weights ? "Weights ready" : "No weights yet"}</span>
                    </div>
                  </div>
                  <p className="model-id-code">{adapter.path}</p>
                </div>
                <div className="model-row-actions">
                  {adapter.active ? (
                    <>
                      <button
                        type="button"
                        className="primary compact-button"
                        onClick={onOpenChat}
                      >
                        <span>Chat</span>
                        <ArrowRight size={14} />
                      </button>
                      <button
                        type="button"
                        className="secondary compact-button"
                        disabled={busyId !== null}
                        data-busy={busyId === "__deactivate__" || undefined}
                        onClick={deactivateAdapter}
                      >
                        <XCircle size={15} />
                        <span>Deactivate</span>
                      </button>
                    </>
                  ) : (
                    <button
                      type="button"
                      className="primary compact-button"
                      disabled={busyId !== null}
                      data-busy={busyId === adapter.id || undefined}
                      onClick={() => activateAdapter(adapter.id)}
                    >
                      <CheckCircle2 size={15} />
                      <span>{busyId === adapter.id ? "Activating" : "Activate"}</span>
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </>
  );
}

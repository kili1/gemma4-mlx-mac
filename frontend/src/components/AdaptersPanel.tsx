import React from "react";
import { Activity, CheckCircle2, RefreshCw, XCircle } from "lucide-react";

import { fetchJson } from "../api";
import type { AdapterInfo } from "../types";
import { PageHeader, StatusBadge } from "./Common";

export function AdaptersPanel() {
  const [adapters, setAdapters] = React.useState<AdapterInfo[]>([]);
  const [busyId, setBusyId] = React.useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = React.useState(false);
  const [message, setMessage] = React.useState("");

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
      setMessage(`Activated adapter "${adapterId}". New chat turns will load it automatically.`);
      await refreshAdapters();
    } catch (error: unknown) {
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
      setMessage("Deactivated adapter. Chat will use base model weights.");
      await refreshAdapters();
    } catch (error: unknown) {
      setMessage(error instanceof Error ? error.message : "Could not deactivate adapter.");
    } finally {
      setBusyId(null);
    }
  }

  const activeAdapter = adapters.find((adapter) => adapter.active) ?? null;

  return (
    <>
      <PageHeader title="Adapters" eyebrow="Local LoRA outputs">
        <StatusBadge tone={activeAdapter ? "good" : "neutral"}>
          {activeAdapter ? `Active: ${activeAdapter.id}` : `${adapters.length} found`}
        </StatusBadge>
        {activeAdapter && (
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
        )}
        <button
          type="button"
          className="secondary compact-button"
          disabled={isRefreshing}
          data-busy={isRefreshing || undefined}
          onClick={refreshAdapters}
        >
          <RefreshCw size={15} />
          <span>Refresh</span>
        </button>
      </PageHeader>
      {message && <p className="notice">{message}</p>}
      <div className="model-list">
        {adapters.length === 0 ? (
          <div className="panel empty-state">
            <Activity size={28} />
            <strong>No adapters</strong>
            <small>Run a LoRA fine-tuning job to create an adapter under ./adapters.</small>
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
                  <p>{adapter.path}</p>
                </div>
                {adapter.active ? (
                  <button
                    type="button"
                    className="secondary"
                    disabled={busyId !== null}
                    data-busy={busyId === "__deactivate__" || undefined}
                    onClick={deactivateAdapter}
                  >
                    <XCircle size={16} />
                    <span>Deactivate</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    className="primary icon-button"
                    disabled={busyId !== null}
                    data-busy={busyId === adapter.id || undefined}
                    onClick={() => activateAdapter(adapter.id)}
                  >
                    <CheckCircle2 size={16} />
                    <span>{busyId === adapter.id ? "Activating" : "Activate"}</span>
                  </button>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </>
  );
}

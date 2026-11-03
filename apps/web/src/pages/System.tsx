import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { Panel } from "../components/ui";

export default function System() {
  const [h, setH] = useState<any>(null);
  const [sys, setSys] = useState<any>(null);
  const [offline, setOffline] = useState(false);

  const load = () => {
    api.get("/ai/health").then((d: any) => { setH(d); setOffline(!!d.offline_mode); }).catch(() => {});
    api.get("/metrics/system").then(setSys).catch(() => {});
  };
  useEffect(() => { load(); const t = setInterval(load, 4000); return () => clearInterval(t); }, []);

  const toggle = async () => {
    const d = await api.post<{ offline: boolean }>("/demo/offline", { offline: !offline });
    setOffline(d.offline); load();
  };

  return (
    <div className="space-y-4">
      <Panel
        title="Capability health"
        right={
          <button className="btn btn-ghost" onClick={toggle}>
            {offline ? "Offline mode: ON" : "Offline mode: OFF"}
          </button>
        }
      >
        <p className="text-xs muted mb-3">
          The system reports its own degraded modes. If the venue network dies, flip offline mode —
          every AI path switches to a local fallback and the demo keeps running.
        </p>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {h && Object.entries(h.capabilities).map(([name, v]: [string, any]) => (
            <div key={name} className="p-3 rounded-lg" style={{ background: "var(--panel-2)" }}>
              <div className="flex items-center justify-between mb-2">
                <b className="text-sm capitalize">{name}</b>
                <span className="chip" style={{ color: v.ok ? "var(--accent-2)" : "var(--danger)" }}>
                  {v.ok ? "ready" : "error"}
                </span>
              </div>
              <pre className="mono text-[11px] muted overflow-x-auto whitespace-pre-wrap">
                {JSON.stringify(v, null, 1)}
              </pre>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Live telemetry">
        <div className="flex flex-wrap gap-2 mb-3">
          <span className="chip">records {sys?.items ?? "—"}</span>
          <span className="chip">audit events {sys?.events ?? "—"}</span>
          <span className="chip">ws clients {sys?.ws_clients ?? "—"}</span>
        </div>
        <div className="scroll max-h-64 overflow-y-auto space-y-1">
          {(sys?.recent_events ?? []).map((e: any, i: number) => (
            <div key={i} className="mono text-xs flex gap-3 px-2 py-1 rounded" style={{ background: "var(--panel-2)" }}>
              <span className="muted">{new Date(e.at).toLocaleTimeString()}</span>
              <span style={{ color: "var(--accent)" }}>{e.kind}</span>
              {e.item_id && <span className="muted">#{e.item_id}</span>}
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}

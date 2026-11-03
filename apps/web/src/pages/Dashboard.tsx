import { useCallback, useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, CartesianGrid } from "recharts";
import { api, type Stats } from "../lib/api";
import { useWS } from "../lib/useWS";
import { GeoScatter, Kpi, Panel, SevChip } from "../components/ui";

const CAT_COLORS = ["#4f8cff", "#22d3a7", "#f0b429", "#ff9b57", "#f2565b", "#a78bfa"];

export default function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [feed, setFeed] = useState<{ id: number; title: string; category: string; severity: number }[]>([]);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => { api.get<Stats>("/items/stats").then(setStats).catch(() => {}); }, []);
  useEffect(() => { load(); }, [load]);

  const connected = useWS((m) => {
    if (m.kind === "item.created") {
      setFeed((f) => [m.data, ...f].slice(0, 12));
      load();   // keep aggregates honest as live data lands
    }
  });

  const seed = async () => { setBusy(true); await api.post("/demo/seed", { n: 150 }).catch(() => {}); setBusy(false); load(); };
  const simulate = async () => { await api.post("/demo/simulate", { seconds: 20, per_second: 1.5 }).catch(() => {}); };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="chip" style={{ color: connected ? "var(--accent-2)" : "var(--muted)" }}>
          ● {connected ? "Live" : "Reconnecting"}
        </span>
        <button className="btn" onClick={simulate}>Simulate live intake</button>
        <button className="btn btn-ghost" onClick={seed} disabled={busy}>
          {busy ? "Seeding…" : "Reseed demo data"}
        </button>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Kpi label="Total records" value={stats?.total ?? "—"} sub="ingested across all channels" />
        <Kpi label="Resolved" value={stats ? `${(stats.resolution_rate * 100).toFixed(0)}%` : "—"}
             sub={stats ? `${stats.resolved} of ${stats.total}` : ""} tone="good" />
        <Kpi label="Avg severity" value={stats?.avg_severity ?? "—"} sub="1 = low · 5 = critical" tone="warn" />
        <Kpi label="Critical open" value={stats?.by_severity.find(s => s.key === "5")?.count ?? 0}
             sub="needs dispatch within the hour" tone="bad" />
      </div>

      <div className="grid lg:grid-cols-3 gap-4">
        <Panel title="By category" className="lg:col-span-2">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={stats?.by_category ?? []}>
              <CartesianGrid stroke="#1b2437" vertical={false} />
              <XAxis dataKey="key" stroke="#8b97ad" fontSize={12} />
              <YAxis stroke="#8b97ad" fontSize={12} />
              <Tooltip contentStyle={{ background: "#151b2b", border: "1px solid #222b3f", borderRadius: 10, color: "#e8edf7" }} />
              <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                {(stats?.by_category ?? []).map((_, i) => <Cell key={i} fill={CAT_COLORS[i % CAT_COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="By status">
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie data={stats?.by_status ?? []} dataKey="count" nameKey="key" innerRadius={52} outerRadius={88} paddingAngle={3}>
                {(stats?.by_status ?? []).map((_, i) => <Cell key={i} fill={CAT_COLORS[i % CAT_COLORS.length]} />)}
              </Pie>
              <Tooltip contentStyle={{ background: "#151b2b", border: "1px solid #222b3f", borderRadius: 10, color: "#e8edf7" }} />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap gap-2 justify-center mt-1">
            {(stats?.by_status ?? []).map((s, i) => (
              <span key={s.key} className="chip" style={{ color: CAT_COLORS[i % CAT_COLORS.length] }}>
                {s.key} · {s.count}
              </span>
            ))}
          </div>
        </Panel>
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        <Panel title="Geographic spread" right={<span className="text-xs muted">size & colour = severity</span>}>
          <GeoScatter points={stats?.geo ?? []} />
        </Panel>

        <Panel title="Live intake feed" right={<span className="text-xs muted">{feed.length} since page load</span>}>
          <div className="scroll max-h-[260px] overflow-y-auto space-y-1.5">
            {feed.length === 0 && <div className="muted text-sm">Hit “Simulate live intake” and watch this fill in real time.</div>}
            {feed.map((f, i) => (
              <div key={`${f.id}-${i}`} className="flash flex items-center justify-between gap-3 px-3 py-2 rounded-lg" style={{ background: "var(--panel-2)" }}>
                <div className="min-w-0">
                  <div className="text-sm truncate">{f.title}</div>
                  <div className="text-xs muted">{f.category}</div>
                </div>
                <SevChip n={f.severity} />
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}

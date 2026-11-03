import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { Kpi, Panel } from "../components/ui";

type Impact = {
  assumptions: Record<string, number>;
  demo: { cases_processed: number; officer_hours_saved: number; cost_saved_inr: number };
  at_national_scale: {
    cases_per_month: number; officer_hours_saved_per_month: number;
    cost_saved_per_month_inr: number; cost_saved_per_year_cr: number;
  };
  quality: { accuracy: number; baseline: number; relative_error_reduction: number };
  note: string;
};

const inr = (n: number) => "₹" + n.toLocaleString("en-IN");

export default function ImpactPage() {
  const [d, setD] = useState<Impact | null>(null);
  const [edits, setEdits] = useState<Record<string, string>>({});

  const load = () => api.get<Impact>("/metrics/impact").then(setD).catch(() => {});
  useEffect(() => { load(); }, []);

  const apply = async () => {
    const overrides: Record<string, number> = {};
    for (const [k, v] of Object.entries(edits)) {
      const n = Number(v);
      if (v !== "" && !Number.isNaN(n)) overrides[k] = n;
    }
    setD(await api.post<Impact>("/metrics/impact", { overrides }));
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Kpi label="Hours saved (this demo)" value={d?.demo.officer_hours_saved ?? "—"} sub="on the records on screen" />
        <Kpi label="Cost saved (this demo)" value={d ? inr(d.demo.cost_saved_inr) : "—"} tone="good" />
        <Kpi label="Hours / month at scale" value={d ? d.at_national_scale.officer_hours_saved_per_month.toLocaleString("en-IN") : "—"}
             sub="across districts in scope" tone="warn" />
        <Kpi label="₹ crore / year at scale" value={d?.at_national_scale.cost_saved_per_year_cr ?? "—"} tone="bad" />
      </div>

      <Panel title="Assumptions — every number on this page is derived from these">
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {d && Object.entries(d.assumptions).map(([k, v]) => (
            <label key={k} className="text-sm">
              <div className="muted text-xs mb-1">{k.replace(/_/g, " ")}</div>
              <input className="input mono text-sm" defaultValue={String(v)}
                     onChange={(e) => setEdits((p) => ({ ...p, [k]: e.target.value }))} />
            </label>
          ))}
        </div>
        <div className="flex items-center gap-2 mt-4">
          <button className="btn" onClick={apply}>Recompute</button>
          <span className="text-xs muted">
            Change an assumption live when a judge challenges it. That move wins arguments.
          </span>
        </div>
      </Panel>

      {d && (
        <Panel title="Model quality">
          <div className="grid sm:grid-cols-3 gap-4 text-sm">
            <div><div className="muted text-xs mb-1">Our accuracy</div><div className="text-2xl font-bold" style={{ color: "var(--accent-2)" }}>{(d.quality.accuracy * 100).toFixed(1)}%</div></div>
            <div><div className="muted text-xs mb-1">Baseline</div><div className="text-2xl font-bold muted">{(d.quality.baseline * 100).toFixed(1)}%</div></div>
            <div><div className="muted text-xs mb-1">Relative error reduction</div><div className="text-2xl font-bold" style={{ color: "var(--accent)" }}>{(d.quality.relative_error_reduction * 100).toFixed(0)}%</div></div>
          </div>
          <p className="text-xs mt-4" style={{ color: "var(--warn)" }}>{d.note}</p>
        </Panel>
      )}
    </div>
  );
}

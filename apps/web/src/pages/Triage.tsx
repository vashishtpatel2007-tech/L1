import { useState } from "react";
import { api, type Triage as T } from "../lib/api";
import { Panel, SevChip, Spinner } from "../components/ui";

const SAMPLE = `The overhead water tank in our ward has been leaking sewage into the drinking water line for 5 days. Three children in my building now have diarrhoea. Nobody from the municipal office has visited despite four phone calls.`;

export default function Triage() {
  const [text, setText] = useState(SAMPLE);
  const [out, setOut] = useState<T | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const run = async () => {
    setBusy(true); setErr(""); setOut(null);
    try {
      setOut(await api.post<T>("/ai/triage", { text }));
    } catch (e) { setErr(String(e)); }
    setBusy(false);
  };

  return (
    <div className="grid lg:grid-cols-2 gap-4">
      <Panel title="Unstructured input">
        <textarea className="textarea mono text-sm" value={text} onChange={(e) => setText(e.target.value)} />
        <div className="flex items-center gap-2 mt-3">
          <button className="btn" onClick={run} disabled={busy || !text.trim()}>Run triage</button>
          <button className="btn btn-ghost" onClick={() => { setText(SAMPLE); setOut(null); }}>Reset sample</button>
          {busy && <Spinner label="Classifying" />}
        </div>
        <p className="text-xs muted mt-3">
          Accepts anything a citizen, field officer or support agent would actually write —
          messy, multilingual, no structure. Output is a routable decision.
        </p>
      </Panel>

      <Panel title="Structured decision" right={out && <span className="chip">{out.engine === "llm" ? "LLM" : "rules engine"}</span>}>
        {err && <div className="text-sm" style={{ color: "var(--danger)" }}>{err}</div>}
        {!out && !err && <div className="muted text-sm">Run triage to see the structured output.</div>}
        {out && (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <SevChip n={out.severity} />
              <span className="chip" style={{ color: "var(--accent)" }}>{out.category}</span>
              <span className="chip">{(out.confidence * 100).toFixed(0)}% confidence</span>
            </div>

            <Field label="Routed to">{out.department}</Field>
            <Field label="Action">{out.suggested_action}</Field>
            <Field label="Summary">{out.summary}</Field>

            {out.reasons && out.reasons.length > 0 && (
              <div>
                <div className="text-xs uppercase tracking-wider muted mb-1.5">Why this severity</div>
                <div className="flex flex-wrap gap-1.5">
                  {out.reasons.map((r) => (
                    <span key={r.signal} className="chip" style={{ color: "var(--warn)" }}>
                      {r.signal} +{r.weight}
                    </span>
                  ))}
                </div>
                {out.duration_detected && (
                  <div className="text-xs muted mt-2">
                    Unresolved duration detected: <b>{out.duration_detected}</b> → escalated
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </Panel>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wider muted mb-1">{label}</div>
      <div className="text-sm">{children}</div>
    </div>
  );
}

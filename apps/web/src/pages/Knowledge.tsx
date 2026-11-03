import { useState } from "react";
import { api } from "../lib/api";
import { Panel, Spinner } from "../components/ui";

type Src = { n: number; doc_id: string; score: number; excerpt: string };
type Ans = { answer: string; sources: Src[]; mode?: string };

export default function Knowledge() {
  const [q, setQ] = useState("Which wards have repeated unresolved water complaints?");
  const [ans, setAns] = useState<Ans | null>(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");

  const index = async () => {
    setBusy(true); setNote("");
    try {
      const r = await api.post<{ chunks: number; mode: string }>("/ai/rag/index-from-items");
      setNote(`Indexed ${r.chunks} chunks · ${r.mode} embeddings`);
    } catch (e) { setNote(String(e)); }
    setBusy(false);
  };

  const ask = async () => {
    setBusy(true); setAns(null);
    try { setAns(await api.post<Ans>("/ai/rag/ask", { query: q, k: 5 })); }
    catch (e) { setNote(String(e)); }
    setBusy(false);
  };

  return (
    <div className="space-y-4">
      <Panel title="Knowledge base" right={note && <span className="chip">{note}</span>}>
        <div className="flex flex-wrap items-center gap-2">
          <button className="btn btn-ghost" onClick={index} disabled={busy}>Index everything in the database</button>
          <span className="text-xs muted">Builds a retrieval index over all records — no manual upload needed.</span>
        </div>
      </Panel>

      <Panel title="Ask a grounded question">
        <div className="flex gap-2">
          <input className="input" value={q} onChange={(e) => setQ(e.target.value)}
                 onKeyDown={(e) => e.key === "Enter" && ask()} placeholder="Ask anything about the indexed records…" />
          <button className="btn" onClick={ask} disabled={busy || !q.trim()}>Ask</button>
        </div>
        {busy && <div className="mt-3"><Spinner label="Retrieving" /></div>}

        {ans && (
          <div className="mt-4 space-y-4">
            <div className="text-sm whitespace-pre-wrap leading-relaxed">{ans.answer}</div>
            {ans.sources.length > 0 && (
              <div>
                <div className="text-xs uppercase tracking-wider muted mb-2">
                  Retrieved evidence — every claim is traceable
                </div>
                <div className="space-y-2">
                  {ans.sources.map((s) => (
                    <div key={s.n} className="p-3 rounded-lg text-sm" style={{ background: "var(--panel-2)" }}>
                      <div className="flex justify-between mb-1">
                        <span className="chip" style={{ color: "var(--accent)" }}>[{s.n}] record #{s.doc_id}</span>
                        <span className="mono muted text-xs">similarity {s.score}</span>
                      </div>
                      <div className="muted">{s.excerpt}…</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </Panel>
    </div>
  );
}

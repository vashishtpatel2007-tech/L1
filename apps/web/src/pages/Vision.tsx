import { useState } from "react";
import { api } from "../lib/api";
import { Panel, Spinner } from "../components/ui";

type Pred = { label: string; confidence: number };
type Out = { tier: string; device?: string; predictions: Pred[]; heatmap_png_b64?: string | null; degraded_from?: string };

export default function Vision() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>("");
  const [out, setOut] = useState<Out | null>(null);
  const [busy, setBusy] = useState(false);
  const [explain, setExplain] = useState(true);

  const pick = (f: File | null) => {
    setFile(f); setOut(null);
    setPreview(f ? URL.createObjectURL(f) : "");
  };

  const run = async () => {
    if (!file) return;
    setBusy(true);
    const form = new FormData();
    form.append("file", file);
    form.append("topk", "4");
    form.append("explain", String(explain));
    try { setOut(await api.upload<Out>("/ai/vision/predict", form)); }
    catch (e) { setOut({ tier: "error", predictions: [], degraded_from: String(e) }); }
    setBusy(false);
  };

  return (
    <div className="grid lg:grid-cols-2 gap-4">
      <Panel title="Image input">
        <label
          className="block rounded-xl p-6 text-center cursor-pointer"
          style={{ border: "1.5px dashed var(--line)", background: "var(--panel-2)" }}
        >
          <input type="file" accept="image/*" className="hidden" onChange={(e) => pick(e.target.files?.[0] ?? null)} />
          {preview
            ? <img src={preview} alt="Selected input" className="max-h-64 mx-auto rounded-lg" />
            : <span className="muted text-sm">Click to choose an image, or drop one here</span>}
        </label>

        <label className="flex items-center gap-2 mt-3 text-sm">
          <input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)} />
          Generate Grad-CAM explanation
        </label>

        <div className="flex items-center gap-2 mt-3">
          <button className="btn" onClick={run} disabled={!file || busy}>Classify</button>
          {busy && <Spinner label="Running inference" />}
        </div>
      </Panel>

      <Panel
        title="Model output"
        right={out && <span className="chip">{out.tier}{out.device ? ` · ${out.device}` : ""}</span>}
      >
        {!out && <div className="muted text-sm">Upload an image and classify it.</div>}
        {out?.degraded_from && (
          <div className="text-xs mb-2" style={{ color: "var(--warn)" }}>
            Degraded to fallback: {out.degraded_from}
          </div>
        )}
        {out && (
          <div className="space-y-3">
            {out.predictions.map((p) => (
              <div key={p.label}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="truncate">{p.label}</span>
                  <span className="mono muted">{(p.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2 rounded-full overflow-hidden" style={{ background: "var(--panel-2)" }}>
                  <div className="h-full rounded-full" style={{ width: `${p.confidence * 100}%`, background: "var(--accent)" }} />
                </div>
              </div>
            ))}

            {out.heatmap_png_b64 && (
              <div className="pt-2">
                <div className="text-xs uppercase tracking-wider muted mb-1.5">
                  Grad-CAM · red = pixels that drove the decision
                </div>
                <img src={`data:image/png;base64,${out.heatmap_png_b64}`} alt="Grad-CAM attention heatmap" className="rounded-lg w-full" />
              </div>
            )}
          </div>
        )}
      </Panel>
    </div>
  );
}

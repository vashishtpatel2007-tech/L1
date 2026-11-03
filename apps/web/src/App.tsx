import { useEffect, useState } from "react";
import { Activity, BrainCircuit, Image as ImageIcon, LayoutDashboard, LineChart, Search } from "lucide-react";
import Dashboard from "./pages/Dashboard";
import Triage from "./pages/Triage";
import Vision from "./pages/Vision";
import Knowledge from "./pages/Knowledge";
import ImpactPage from "./pages/Impact";
import System from "./pages/System";
import { api } from "./lib/api";

const TABS = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard, el: <Dashboard /> },
  { id: "triage", label: "Triage", icon: BrainCircuit, el: <Triage /> },
  { id: "vision", label: "Vision", icon: ImageIcon, el: <Vision /> },
  { id: "knowledge", label: "Knowledge", icon: Search, el: <Knowledge /> },
  { id: "impact", label: "Impact", icon: LineChart, el: <ImpactPage /> },
  { id: "system", label: "System", icon: Activity, el: <System /> },
] as const;

export default function App() {
  // Hash routing: survives refresh and deep-links a specific tab for the demo.
  const [tab, setTab] = useState<string>(() => location.hash.slice(1) || "dashboard");
  const [app, setApp] = useState<{ app: string; offline_mode: boolean } | null>(null);

  useEffect(() => {
    const onHash = () => setTab(location.hash.slice(1) || "dashboard");
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => { api.get<any>("/health").then(setApp).catch(() => {}); }, []);

  const active = TABS.find((t) => t.id === tab) ?? TABS[0];

  return (
    <div className="min-h-full flex flex-col">
      <header
        className="sticky top-0 z-10 px-4 lg:px-6 py-3 flex flex-wrap items-center gap-x-5 gap-y-2"
        style={{ background: "rgba(8,11,20,.88)", backdropFilter: "blur(10px)", borderBottom: "1px solid var(--line)" }}
      >
        <div className="flex items-center gap-2.5">
          <div
            className="w-8 h-8 rounded-lg grid place-items-center font-bold"
            style={{ background: "linear-gradient(135deg,var(--accent),var(--accent-2))", color: "#04102a" }}
          >
            {(app?.app ?? "H")[0]}
          </div>
          <div>
            <div className="font-bold leading-tight">{app?.app ?? "Hackathon Kit"}</div>
            <div className="text-[11px] muted leading-tight">national-level build</div>
          </div>
        </div>

        <nav className="flex items-center gap-1 flex-wrap">
          {TABS.map((t) => {
            const Icon = t.icon;
            const on = t.id === active.id;
            return (
              <a
                key={t.id}
                href={`#${t.id}`}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm transition-colors"
                style={{
                  background: on ? "var(--panel-2)" : "transparent",
                  color: on ? "var(--text)" : "var(--muted)",
                  border: `1px solid ${on ? "var(--line)" : "transparent"}`,
                  fontWeight: on ? 650 : 500,
                }}
              >
                <Icon size={15} /> {t.label}
              </a>
            );
          })}
        </nav>

        {app?.offline_mode && (
          <span className="chip ml-auto" style={{ color: "var(--warn)" }}>offline mode</span>
        )}
      </header>

      <main className="flex-1 p-4 lg:p-6 max-w-[1400px] w-full mx-auto">{active.el}</main>

      <footer className="px-6 py-3 text-xs muted" style={{ borderTop: "1px solid var(--line)" }}>
        API docs at <code className="mono">/docs</code> · websocket at <code className="mono">/ws</code>
      </footer>
    </div>
  );
}

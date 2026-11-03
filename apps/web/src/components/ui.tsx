import type { ReactNode } from "react";

export function Panel({ title, right, children, className = "" }: {
  title?: string; right?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`panel p-4 ${className}`}>
      {(title || right) && (
        <header className="flex items-center justify-between mb-3">
          {title && <h3 className="text-sm font-semibold tracking-wide uppercase muted">{title}</h3>}
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export function Kpi({ label, value, sub, tone = "accent" }: {
  label: string; value: ReactNode; sub?: string; tone?: "accent" | "good" | "warn" | "bad";
}) {
  const color = { accent: "var(--accent)", good: "var(--accent-2)", warn: "var(--warn)", bad: "var(--danger)" }[tone];
  return (
    <div className="panel p-4">
      <div className="text-xs uppercase tracking-wider muted mb-1">{label}</div>
      <div className="text-3xl font-bold leading-none" style={{ color }}>{value}</div>
      {sub && <div className="text-xs muted mt-1.5">{sub}</div>}
    </div>
  );
}

export function SevChip({ n }: { n: number }) {
  const labels = ["", "Low", "Minor", "Moderate", "High", "Critical"];
  return <span className={`chip sev-${n}`}>● S{n} {labels[n] ?? ""}</span>;
}

export function Spinner({ label = "Working" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 muted text-sm">
      <span className="inline-block w-3.5 h-3.5 rounded-full border-2 border-current border-t-transparent animate-spin" />
      {label}…
    </div>
  );
}

/** Offline-safe geo scatter. Deliberately NOT map tiles: tiles need the
 *  internet, and a blank grey square on stage is a dead demo. */
export function GeoScatter({ points }: {
  points: { lat: number; lon: number; severity: number; category: string }[];
}) {
  if (!points.length) return <div className="muted text-sm">No geocoded records.</div>;
  const lats = points.map(p => p.lat), lons = points.map(p => p.lon);
  const [a, b] = [Math.min(...lats), Math.max(...lats)];
  const [c, d] = [Math.min(...lons), Math.max(...lons)];
  const nx = (v: number, lo: number, hi: number) => (hi - lo < 1e-9 ? 0.5 : (v - lo) / (hi - lo));
  const colors: Record<number, string> = { 1: "#7fe3c0", 2: "#9ec5ff", 3: "#f0b429", 4: "#ff9b57", 5: "#f2565b" };

  return (
    <svg viewBox="0 0 100 70" className="w-full h-[260px]" role="img" aria-label="Geographic distribution of records by severity">
      <defs>
        <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
          <path d="M10 0H0v10" fill="none" stroke="#1b2437" strokeWidth="0.4" />
        </pattern>
      </defs>
      <rect width="100" height="70" fill="url(#grid)" />
      {points.map((p, i) => (
        <circle
          key={i}
          cx={4 + nx(p.lon, c, d) * 92}
          cy={66 - nx(p.lat, a, b) * 62}
          r={0.9 + p.severity * 0.42}
          fill={colors[p.severity] ?? "#9ec5ff"}
          opacity={0.78}
        />
      ))}
    </svg>
  );
}

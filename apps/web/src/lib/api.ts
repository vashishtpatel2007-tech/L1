// Single place that knows how to talk to the backend.
// Dev: Vite proxies /api -> :8000. Prod: FastAPI serves this bundle, same origin.
const BASE = import.meta.env.DEV ? "/api" : "";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.status === 204 ? (null as T) : res.json();
}

export const api = {
  get: <T,>(p: string) => req<T>(p),
  post: <T,>(p: string, body?: unknown) =>
    req<T>(p, { method: "POST", body: JSON.stringify(body ?? {}) }),
  patch: <T,>(p: string, body?: unknown) =>
    req<T>(p, { method: "PATCH", body: JSON.stringify(body ?? {}) }),
  upload: async <T,>(p: string, form: FormData): Promise<T> => {
    const res = await fetch(`${BASE}${p}`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    return res.json();
  },
};

export type Item = {
  id: number; title: string; body: string; category: string;
  status: string; severity: number; score: number;
  lat: number | null; lon: number | null;
  meta: Record<string, unknown>; created_at: string;
};

export type Stats = {
  total: number; resolved: number; resolution_rate: number; avg_severity: number;
  by_category: { key: string; count: number }[];
  by_status: { key: string; count: number }[];
  by_severity: { key: string; count: number }[];
  geo: { lat: number; lon: number; severity: number; category: string; id: number }[];
};

export type Triage = {
  category: string; severity: number; summary: string;
  suggested_action: string; department: string; confidence: number;
  engine: string; reasons?: { signal: string; weight: number }[];
  duration_detected?: string | null;
};

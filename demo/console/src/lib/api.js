// Thin client for the Python backend (demo/agent/server.py).
export async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

export async function postJSON(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `${path}: ${r.status}`);
  return data;
}

export const fmt = {
  int: (n) => (n ?? 0).toLocaleString("en-US"),
  bytes: (b) => (b >= 1048576 ? `${(b / 1048576).toFixed(1)} MB` : `${Math.round((b || 0) / 1024)} KB`),
  pct: (x, d = 0) => `${(100 * (x || 0)).toFixed(d)}%`,
  time: (ts) => (ts ? new Date(ts * 1000).toLocaleTimeString("en-GB", { hour12: false }) : "—"),
  mwords: (w) => (w >= 1e6 ? `${(w / 1e6).toFixed(1)}M` : `${Math.round(w / 1000)}k`),
};

// Document categories in the feed and their colours (spec section 3; story pink, papers blue, ...)
export const KINDS = {
  "Wiki pages": { color: "#2DD4BF", label: "Wiki page" },
  Paper: { color: "#60A5FA", label: "Paper" },
  Article: { color: "#A78BFA", label: "Article" },
  Story: { color: "#F27FBA", label: "Story" },
  "Gov report": { color: "#FBBF24", label: "Gov report" },
  Meeting: { color: "#FB923C", label: "Meeting" },
};
export const kindColor = (k) => KINDS[k]?.color || "#64748B";

// System colours for the evaluation charts
export const SYSTEM_COLORS = {
  "BM25 RAG": "#FB923C", "SketchRAG, always raw": "#4CE0B3", "SketchRAG + gate": "#2DD4BF",
  "SketchRAG gate + agent": "#5EEAD4", "SARA-lite": "#A78BFA", "Full context": "#64748B", "Dense RAG": "#60A5FA",
  "AMBER-lite": "#F27FBA", "SketchRAG LLM agent": "#99F6E4", "SketchRAG, digest only": "#2F7F74",
};

const STOP = new Set("what which who whom whose when where why how does did the and for with from that this into was were are is about after".split(" "));
export const queryTerms = (q) =>
  [...new Set((q.toLowerCase().match(/[a-z0-9]+/g) || []).filter((w) => w.length >= 4 && !STOP.has(w)))];

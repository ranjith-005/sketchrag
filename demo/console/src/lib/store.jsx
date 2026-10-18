import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { getJSON } from "./api";

// ---------------- live polling ----------------
export function usePoll(path, ms = 1000, deps = []) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    if (!path) { setData(null); return undefined; }
    let alive = true, timer;
    const tick = async () => {
      try {
        const d = await getJSON(path);
        if (alive) { setData(d); setError(null); }
      } catch (e) {
        if (alive) setError(e);
      }
      if (alive && ms) timer = setTimeout(tick, ms);
    };
    tick();
    return () => { alive = false; clearTimeout(timer); };
  }, [path, ms, ...deps]);
  return [data, error];
}

const LiveCtx = createContext(null);

/** Stream status (1 s) and service health (4 s), shared by the shell and every page. */
export function LiveProvider({ children }) {
  const [status, statusErr] = usePoll("/api/stream/status", 1000);
  const [health, healthErr] = usePoll("/api/health", 4000);
  const online = !!status && !statusErr && !!health && !healthErr;
  return <LiveCtx.Provider value={{ status, health, online }}>{children}</LiveCtx.Provider>;
}
export const useLive = () => useContext(LiveCtx);

// ---------------- query runs (kept here so a run keeps streaming while you change pages) ----------------
const QueryCtx = createContext(null);

const emptyRun = (question, policy) => ({
  id: Date.now(), question, policy, state: "running", steps: {}, answer: "", done: null, error: null, startedAt: Date.now(),
});

export function QueryProvider({ children }) {
  const [runs, setRuns] = useState([]);
  const esRef = useRef(null);
  const busy = runs.some((r) => r.state === "running");

  const patch = (id, fn) => setRuns((rs) => rs.map((r) => (r.id === id ? fn(r) : r)));

  const ask = useCallback((question, policy = "gate") => {
    if (!question.trim() || busy) return;
    const run = emptyRun(question.trim(), policy);
    setRuns((rs) => [run, ...rs].slice(0, 8));
    const es = new EventSource(`/api/ask?q=${encodeURIComponent(run.question)}&policy=${policy}`);
    esRef.current = es;
    // steps arrive in milliseconds; reveal them ~0.5 s apart so the trace is readable live
    const queue = [];
    let last = 0, finished = false;
    const pump = setInterval(() => {
      const ev = queue[0];
      if (!ev) return;
      if (ev.type === "step" && Date.now() - last < 520) return;
      queue.shift();
      if (ev.type === "step") {
        last = Date.now();
        patch(run.id, (r) => ({ ...r, steps: { ...r.steps, [ev.step]: ev } }));
      } else if (ev.type === "start") {
        patch(run.id, (r) => ({ ...r, start: ev }));
      } else if (ev.type === "token") {
        patch(run.id, (r) => ({ ...r, answer: r.answer + ev.text }));
      } else if (ev.type === "done") {
        patch(run.id, (r) => ({ ...r, state: "done", done: ev }));
        clearInterval(pump);
      } else if (ev.type === "error") {
        patch(run.id, (r) => ({ ...r, state: "error", error: ev.message }));
        clearInterval(pump);
      }
    }, 30);
    es.onmessage = (m) => {
      const ev = JSON.parse(m.data);
      queue.push(ev);
      if (ev.type === "done" || ev.type === "error") { finished = true; es.close(); }
    };
    es.onerror = () => {
      es.close();
      if (!finished) queue.push({ type: "error", message: "Connection to the backend was lost, or another query is running." });
    };
  }, [busy]);

  return <QueryCtx.Provider value={{ runs, ask, busy, latest: runs[0] || null }}>{children}</QueryCtx.Provider>;
}
export const useQueries = () => useContext(QueryCtx);

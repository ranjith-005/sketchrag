import { useEffect, useState } from "react";
import { RotateCcw, Save } from "lucide-react";
import { Button, Note, Offline, PageHeader, Panel, Pill, Provenance } from "../components/ui";
import { fmt, getJSON, postJSON } from "../lib/api";
import { useLive, usePoll } from "../lib/store";

const OPTIONS = { max_cells: [16, 32, 64, 128, 256, 512], cms_width: [64, 128, 256, 512, 1024, 2048] };

function Slider({ label, value, options, onChange, color, fmtv = (v) => v, help }) {
  const i = options.indexOf(value);
  return (
    <div>
      <div className="mb-2 flex justify-between text-[14px] font-semibold text-white"><span>{label}</span><span className="font-mono" style={{ color }}>{fmtv(value)}</span></div>
      <input type="range" min={0} max={options.length - 1} value={i < 0 ? 0 : i} onChange={(e) => onChange(options[+e.target.value])}
        className="w-full" style={{ accentColor: color }} />
      <p className="mt-1 text-[11.5px] text-ink-muted">{help}</p>
    </div>
  );
}

const LEVEL = { info: "text-teal", warn: "text-fallback", error: "text-rose-300" };

export default function Config() {
  const { health, online } = useLive();
  const [cfg, setCfg] = useState(null);
  const [draft, setDraft] = useState(null);
  const [msg, setMsg] = useState(null);
  const [level, setLevel] = useState("all");
  const [events] = usePoll("/api/events?limit=120", 1500);
  useEffect(() => { getJSON("/api/config").then((c) => { setCfg(c); setDraft(c); }).catch(() => {}); }, []);
  if (!online) return (<><PageHeader title="Configuration & logs" subtitle="Inspect research parameters, service health and system events." /><Offline /></>);
  const dirty = draft && cfg && ["max_cells", "cms_width", "join_threshold"].some((k) => draft[k] !== cfg[k]);
  const apply = async (body) => {
    try {
      const c = await postJSON("/api/config", body);
      setCfg(c); setDraft(c); setMsg("Applied: the agent's memory was rebuilt and the feed restarted.");
    } catch (e) { setMsg(`Not applied: ${e.message}`); }
  };
  const shown = (events || []).filter((e) => level === "all" || e.level === level);

  return (
    <>
      <PageHeader title="Configuration & logs" subtitle="Inspect research parameters, service health and system events." />
      <div className="grid gap-4 xl:grid-cols-[1.25fr_1fr]">
        <Panel title="Memory configuration" subtitle="Applying rebuilds the backend's memory and restarts the feed" right={<Provenance />}>
          {draft && (
            <div className="space-y-6">
              <Slider label="Maximum memory cells" value={draft.max_cells} options={OPTIONS.max_cells} color="#2DD4BF"
                onChange={(v) => setDraft({ ...draft, max_cells: v })} help="Hot-memory budget. When full, the two closest cells are merged (Ward linkage)." />
              <Slider label="Count-Min Sketch width" value={draft.cms_width} options={OPTIONS.cms_width} color="#60A5FA"
                onChange={(v) => setDraft({ ...draft, cms_width: v })} help="Wider sketch → tighter εN bound → the gate trusts more digests, at more memory per cell." />
              <div>
                <div className="mb-2 flex justify-between text-[14px] font-semibold text-white"><span>Cell similarity threshold</span>
                  <span className="font-mono text-memory-soft">{draft.join_threshold.toFixed(2)}</span></div>
                <input type="range" min={0.7} max={0.98} step={0.01} value={draft.join_threshold} className="w-full" style={{ accentColor: "#A78BFA" }}
                  onChange={(e) => setDraft({ ...draft, join_threshold: +e.target.value })} />
                <p className="mt-1 text-[11.5px] text-ink-muted">Cosine similarity a chunk needs to join an existing cell directly.</p>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <span className="text-[13px] font-semibold text-white">Answering policy</span>
                <Pill tone="violet">Deterministic error-bound gate</Pill>
                <span className="text-[11.5px] text-ink-muted">digest-only is selectable per query in the playground</span>
              </div>
              <div className="flex flex-wrap gap-3 border-t border-line pt-4">
                <Button disabled={!dirty} onClick={() => apply({ max_cells: draft.max_cells, cms_width: draft.cms_width, join_threshold: draft.join_threshold })}>
                  <Save size={15} /> Apply & restart feed</Button>
                <Button tone="ghost" onClick={() => apply(cfg.defaults)}><RotateCcw size={15} /> Reset defaults</Button>
              </div>
              {draft.max_cells !== cfg.presets_rehearsed_with && (
                <Note tone="warn">Suggested questions were rehearsed with {cfg.presets_rehearsed_with} cells. With a different budget the memory changes, so they may not be answered the same way.</Note>)}
              {msg && <Note>{msg}</Note>}
            </div>)}
        </Panel>

        <Panel title="Runtime status" subtitle="From the backend's real health check, refreshed every 4 s" right={<Provenance />}>
          <div className="space-y-3">
            {[["Frontend", { ok: true, detail: "React console served by the backend" }], ["Ingestion", health?.ingestion],
              ["Retrieval", health?.retrieval], ["LLM provider", health?.llm]].map(([name, h]) => (
              <div key={name} className="flex items-center justify-between rounded-ctl border border-line bg-card px-4 py-3">
                <div><div className="text-[14px] font-semibold text-white">{name}</div><div className="text-[11.5px] text-ink-muted">{h?.detail || "checking…"}</div></div>
                <Pill tone={h?.ok ? "teal" : "amber"}>{h?.ok ? "Running" : "Not ready"}</Pill>
              </div>))}
          </div>
        </Panel>
      </div>

      <Panel className="mt-4" title="System events" subtitle="Ingestion, memory and query events from the running backend" bodyClass="p-0"
        right={<div className="flex items-center gap-2"><Provenance />
          <select value={level} onChange={(e) => setLevel(e.target.value)} className="rounded-ctl border border-line bg-card px-2.5 py-1.5 text-[12px] text-ink">
            <option value="all">All levels</option><option value="info">Info</option><option value="warn">Warn</option><option value="error">Error</option>
          </select></div>}>
        <div className="max-h-[420px] overflow-auto">
          <table className="w-full min-w-[640px] text-[12.5px]">
            <thead className="sticky top-0 bg-panel text-left text-[11px] uppercase tracking-wider text-ink-dim">
              <tr><th className="px-5 py-2.5 font-semibold">Time</th><th className="py-2.5 font-semibold">Level</th><th className="py-2.5 font-semibold">Event</th>
                <th className="py-2.5 font-semibold">Message</th><th className="px-5 py-2.5 font-semibold">Context</th></tr></thead>
            <tbody>{shown.map((e) => (
              <tr key={e.id} className="border-t border-line/60">
                <td className="px-5 py-2 font-mono text-ink-muted">{fmt.time(e.ts)}</td>
                <td className={`py-2 font-mono font-semibold uppercase ${LEVEL[e.level]}`}>{e.level}</td>
                <td className="py-2 font-mono text-ink-muted">{e.kind}</td>
                <td className="max-w-[420px] truncate py-2 pr-4 text-white">{e.text}</td>
                <td className="max-w-[280px] truncate px-5 py-2 text-ink-muted">{e.detail}</td>
              </tr>))}</tbody>
          </table>
        </div>
      </Panel>
    </>
  );
}

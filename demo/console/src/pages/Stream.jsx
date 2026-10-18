import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Coins, FileStack, Pause, Play, RotateCcw, Search, TrendingUp } from "lucide-react";
import { Bar, Button, Kpi, KindDot, Offline, PageHeader, Panel, Pill, Provenance } from "../components/ui";
import { fmt, kindColor, KINDS, postJSON } from "../lib/api";
import { useLive, usePoll } from "../lib/store";

export default function Stream() {
  const { status, online } = useLive();
  const [docs] = usePoll("/api/stream/documents", 1000);
  const [params] = useSearchParams();
  const [q, setQ] = useState(params.get("search") || "");
  const [kind, setKind] = useState("all");
  const [show, setShow] = useState("arrived");
  const control = (body) => postJSON("/api/stream/control", body);

  const rows = useMemo(() => (docs || [])
    .filter((d) => (show === "all" ? true : d.status !== "queued"))
    .filter((d) => kind === "all" || d.kind === kind)
    .filter((d) => !q || d.title.toLowerCase().includes(q.toLowerCase()))
    .reverse(), [docs, q, kind, show]);

  if (!online) return (<><PageHeader title="Live stream" subtitle="Inspect incoming documents, chunking progress and the data folded into memory." /><Offline /></>);

  return (
    <>
      <PageHeader title="Live stream" subtitle="Inspect incoming documents, chunking progress and the data folded into memory." />
      <div className="grid gap-4 md:grid-cols-3">
        <Kpi label="Stream progress" value={fmt.pct(status.progress, 1)} sub={`${fmt.int(status.words)} of ${fmt.mwords(status.total_words)} words`} icon={TrendingUp} />
        <Kpi label="Documents processed" value={`${status.docs_arrived} / ${status.docs_total}`} sub="LongBench documents in fixed stream order" icon={FileStack} tone="blue" />
        <Kpi label="Full-feed size" value={fmt.int(status.tokens_full)} sub="LLM tokens if everything were sent (est.)" icon={Coins} tone="violet" />
      </div>

      <Panel className="mt-4" title="Ingestion controls" subtitle="These control the real ingest thread in the backend" right={<Provenance />}>
        <div className="flex flex-wrap items-center gap-3">
          {status.playing
            ? <Button tone="ghost" onClick={() => control({ action: "pause" })}><Pause size={15} /> Pause ingestion</Button>
            : <Button onClick={() => control({ action: "play" })}><Play size={15} /> Resume</Button>}
          <div className="flex overflow-hidden rounded-ctl border border-line">
            {[1, 4, 16, 64].map((v) => (
              <button key={v} onClick={() => control({ action: "speed", value: v })}
                className={`px-3.5 py-2 text-[13px] font-semibold ${status.speed === v ? "bg-teal text-app" : "text-ink-muted hover:text-ink"}`}>{v}×</button>))}
          </div>
          <Button tone="ghost" onClick={() => control({ action: "restart" })} title="Empty the memory and stream the feed again"><RotateCcw size={15} /> Restart feed</Button>
          <div className="ml-auto flex items-center gap-3 text-[12.5px] text-ink-muted">
            <span>1× ≈ 25 chunks/s</span>
            <Pill tone={status.state === "ingesting" ? "teal" : status.state === "thinking" ? "amber" : "gray"}>{status.state}</Pill>
          </div>
        </div>
        <div className="mt-4"><Bar value={status.progress} max={1} h="h-1.5" /></div>
      </Panel>

      <Panel className="mt-4" title="Document feed" subtitle={`${rows.length} documents shown · newest first`} bodyClass="p-0" right={<Provenance />}>
        <div className="flex flex-wrap items-center gap-2 border-b border-line px-5 py-3">
          <div className="flex min-w-[220px] flex-1 items-center gap-2 rounded-ctl border border-line bg-card px-3 py-2">
            <Search size={15} className="text-ink-dim" />
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search document titles…" className="w-full bg-transparent text-[13px] outline-none placeholder:text-ink-dim" />
          </div>
          <select value={kind} onChange={(e) => setKind(e.target.value)} className="rounded-ctl border border-line bg-card px-3 py-2 text-[13px] text-ink">
            <option value="all">All types</option>
            {Object.keys(KINDS).map((k) => <option key={k} value={k}>{KINDS[k].label}</option>)}
          </select>
          <select value={show} onChange={(e) => setShow(e.target.value)} className="rounded-ctl border border-line bg-card px-3 py-2 text-[13px] text-ink">
            <option value="arrived">Arrived</option><option value="all">Include queued</option>
          </select>
        </div>
        <div className="max-h-[520px] overflow-auto">
          <table className="w-full min-w-[720px] text-[13px]">
            <thead className="sticky top-0 bg-panel text-left text-[11px] uppercase tracking-wider text-ink-dim">
              <tr><th className="px-5 py-2.5 font-semibold">Type</th><th className="py-2.5 font-semibold">Title</th><th className="py-2.5 font-semibold">Chunked</th>
                <th className="py-2.5 text-right font-semibold">Words</th><th className="py-2.5 text-right font-semibold">Arrived</th><th className="px-5 py-2.5 text-right font-semibold">Status</th></tr>
            </thead>
            <tbody>
              {rows.map((d) => (
                <tr key={d.key} className="border-t border-line/60 hover:bg-card/50">
                  <td className="px-5 py-2.5"><span className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-wide" style={{ color: kindColor(d.kind) }}><KindDot color={kindColor(d.kind)} />{d.kind}</span></td>
                  <td className="max-w-[420px] truncate py-2.5 pr-4 text-ink">{d.title}</td>
                  <td className="w-[180px] py-2.5 pr-4"><Bar value={d.progress} max={1} h="h-1.5" color={d.status === "ingested" ? "#2DD4BF" : "#FBBF24"} /></td>
                  <td className="py-2.5 text-right font-mono text-[12px] text-ink-muted">{fmt.int(d.words)}</td>
                  <td className="py-2.5 text-right font-mono text-[12px] text-ink-muted">{fmt.time(d.arrived_at)}</td>
                  <td className="px-5 py-2.5 text-right">
                    <Pill tone={d.status === "ingested" ? "teal" : d.status === "ingesting" ? "amber" : "gray"}>{d.status}</Pill></td>
                </tr>
              ))}
              {!rows.length && <tr><td colSpan={6} className="px-5 py-8 text-center text-ink-muted">No documents match.</td></tr>}
            </tbody>
          </table>
        </div>
      </Panel>
    </>
  );
}

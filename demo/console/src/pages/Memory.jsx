import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRightLeft, Boxes, ScrollText } from "lucide-react";
import { Bar, Kpi, KindDot, Offline, PageHeader, Panel, Pill, Provenance } from "../components/ui";
import { fmt, kindColor, KINDS } from "../lib/api";
import { useLive, usePoll } from "../lib/store";

function CellMap({ cells, maxCells, selected, onSelect, filter, last }) {
  const verdict = last?.verdicts || {};
  const slots = useMemo(() => {
    const sorted = [...(cells || [])].sort((a, b) => a.span[0] - b.span[0]); // stream order of each cell's first chunk
    return Array.from({ length: maxCells }, (_, i) => sorted[i] || null);
  }, [cells, maxCells]);
  const cols = Math.ceil(Math.sqrt(maxCells));
  const pass = (c) => filter === "all" || (filter === "loose" ? c.radius > 0.75 : filter === "compact" ? c.radius <= 0.5 : c.kind === filter)
    || (filter === "query" && verdict[String(c.id)]);
  return (
    <div className="grid gap-[5px]" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
      {slots.map((c, i) => {
        if (!c) return <div key={`e${i}`} className="aspect-[1.6] rounded-[5px] border border-dashed border-line/70" />;
        const v = verdict[String(c.id)];
        const on = filter === "query" ? !!v : pass(c);
        return (
          <button key={c.id} onClick={() => onSelect(c.id)} title={`cell ${c.id} · ${c.n} chunks · ${c.kind} · radius ${c.radius}`}
            className={`cellpop aspect-[1.6] rounded-[5px] transition hover:scale-110 ${selected === c.id ? "ring-2 ring-white ring-offset-2 ring-offset-panel" : ""}
              ${v ? (v === "trusted" ? "ring-2 ring-teal" : "ring-2 ring-fallback") : ""}`}
            style={{ background: kindColor(c.kind), opacity: on ? 0.35 + 0.65 * Math.min(1, Math.log1p(c.n) / Math.log1p(40)) : 0.12 }} />
        );
      })}
    </div>
  );
}

function Inspector({ id }) {
  const [d, err] = usePoll(id !== null ? `/api/memory/cells/${id}` : null, id !== null ? 3000 : 0, [id]);
  if (id === null) return <p className="text-[13px] text-ink-muted">Select a cell in the map to inspect what it stores.</p>;
  if (err) return <p className="text-[13px] text-fallback">Cell {id} was merged into another cell as the stream moved on. Pick another.</p>;
  if (!d) return <p className="text-[13px] text-ink-muted">Loading…</p>;
  const maxEst = Math.max(1, ...d.cms.top_terms.map((t) => t.est));
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <Pill tone={d.loose ? "amber" : "teal"}>{d.loose ? "Loose cluster" : "Compact cluster"}</Pill>
        <Pill tone="gray">positions {d.span[0]}–{d.span[1]}</Pill>
      </div>
      <div className="grid grid-cols-2 gap-2.5">
        {[["Chunk count", d.n], ["Radius", d.radius], ["Cell ID", `cell_${d.id}`], ["Exemplars", `${d.exemplars} / ${d.max_exemplars}`]].map(([l, v]) => (
          <div key={l} className="rounded-ctl border border-line bg-card px-3 py-2"><div className="text-[11px] text-ink-muted">{l}</div>
            <div className="mt-0.5 font-mono text-[16px] text-white">{v}</div></div>))}
      </div>
      <div>
        <div className="mb-1.5 text-[11px] font-bold uppercase tracking-wider text-ink-dim">Extractive digest · {d.digest_words}/{d.digest_limit} words</div>
        <div className="max-h-[132px] overflow-auto rounded-ctl border border-line bg-deep px-3 py-2.5 text-[12.5px] leading-relaxed text-ink">{d.digest}</div>
      </div>
      <div>
        <div className="mb-1.5 text-[11px] font-bold uppercase tracking-wider text-ink-dim">
          Count-Min Sketch · {d.cms.depth} × {d.cms.width} · N = {fmt.int(d.cms.total)} · εN = {d.cms.bound}</div>
        <div className="space-y-1">
          {d.cms.top_terms.slice(0, 8).map((t) => (
            <div key={t.term} className="grid grid-cols-[110px_1fr_44px] items-center gap-2 text-[12px]">
              <span className="truncate font-mono text-retrieval-soft">{t.term}</span><Bar value={t.est} max={maxEst} color="#A78BFA" h="h-1.5" />
              <span className={`text-right font-mono ${t.est > d.cms.bound ? "text-teal" : "text-ink-muted"}`}>{t.est}</span>
            </div>))}
        </div>
        <p className="mt-1 text-[11px] text-ink-dim">Digest terms by estimated count. Teal = above the guaranteed εN bound, so the sketch can vouch for it.</p>
      </div>
      <div>
        <div className="mb-1.5 text-[11px] font-bold uppercase tracking-wider text-ink-dim">Raw chunk pointers · {d.n_pointers} in cold storage</div>
        <div className="space-y-1">
          {d.documents.slice(0, 4).map((doc) => (
            <div key={doc.title} className="flex items-center gap-2 text-[12px]"><KindDot color={kindColor(doc.kind)} />
              <span className="min-w-0 flex-1 truncate text-ink">{doc.title}</span><span className="font-mono text-ink-muted">{doc.chunks} chunks</span></div>))}
        </div>
        <details className="mt-2 text-[12px]">
          <summary className="cursor-pointer text-teal">Show raw text behind the first pointers</summary>
          {d.samples.map((s) => <div key={s.chunk} className="mt-2 rounded-ctl border-l-2 border-fallback bg-card px-3 py-2 text-ink-muted">
            <span className="font-mono text-fallback">#{s.chunk}</span> {s.text}</div>)}
        </details>
      </div>
      <div className="border-t border-line pt-3 text-[12px] text-ink-muted">
        Stored in hot memory: centroid embedding ({d.dim} dimensions), {d.exemplars} exemplar embeddings, a {d.cms.depth}×{d.cms.width}{" "}
        Count-Min table ({d.cms.kb} KB) and the digest. The raw chunks stay in cold storage.
      </div>
    </div>
  );
}

export default function Memory() {
  const { status, online } = useLive();
  const [cells] = usePoll("/api/memory/cells", 1500);
  const [summary] = usePoll("/api/memory/summary", 3000);
  const [last] = usePoll("/api/query/last", 2000);
  const [selected, setSelected] = useState(null);
  const [filter, setFilter] = useState("all");
  useEffect(() => { if (selected === null && cells?.length) setSelected([...cells].sort((a, b) => b.n - a.n)[0].id); }, [cells, selected]);
  if (!online) return (<><PageHeader title="Memory explorer" subtitle="Explore bounded hot-memory cells and inspect the summaries behind retrieval." /><Offline /></>);
  const hasQuery = last && last.hits;

  return (
    <>
      <PageHeader title="Memory explorer" subtitle="Explore bounded hot-memory cells and inspect the summaries behind retrieval." />
      <div className="grid gap-4 md:grid-cols-3">
        <Kpi label="Cell budget" value={`${status.cells} / ${status.max_cells}`} sub="Configured maximum" icon={Boxes} tone="violet" />
        <Kpi label="Merges performed" value={fmt.int(status.merges)} sub="Ward merge-and-reduce" icon={ArrowRightLeft} tone="blue" />
        <Kpi label="Digest limit" value={`${summary?.digest_words ?? 120} words`} sub="Extractive digest per cell" icon={ScrollText} tone="amber" />
      </div>
      <div className="mt-4 grid gap-4 xl:grid-cols-[1.35fr_1fr]">
        <Panel title="Cell map" subtitle="One tile per cell, in stream order · colour = main document type · brightness = size"
          right={<div className="flex items-center gap-2"><Provenance />
            <select value={filter} onChange={(e) => setFilter(e.target.value)} className="rounded-ctl border border-line bg-card px-2.5 py-1.5 text-[12px] text-ink">
              <option value="all">All cells</option><option value="compact">Compact (radius ≤ 0.5)</option><option value="loose">Loose (radius &gt; 0.75)</option>
              {hasQuery && <option value="query">Last query's cells</option>}
              {Object.keys(KINDS).map((k) => <option key={k} value={k}>{KINDS[k].label}</option>)}
            </select></div>}>
          <CellMap cells={cells} maxCells={status.max_cells} selected={selected} onSelect={setSelected} filter={filter} last={last} />
          <div className="mt-4 flex flex-wrap gap-x-5 gap-y-1.5 text-[12px] text-ink-muted">
            {Object.entries(KINDS).map(([k, v]) => <span key={k} className="flex items-center gap-1.5"><KindDot color={v.color} />{v.label}</span>)}
            <span className="flex items-center gap-1.5"><span className="h-2.5 w-3.5 rounded-[3px] border border-dashed border-line" />free slot</span>
          </div>
          {hasQuery && (
            <p className="mt-3 text-[12px] text-ink-muted">Last query: <span className="text-ink">“{last.question.slice(0, 80)}”</span> ·
              <span className="text-teal"> teal ring</span> = digest trusted, <span className="text-fallback">amber ring</span> = escalated to raw ·{" "}
              <Link to="/query" className="text-teal hover:underline">open trace</Link></p>)}
        </Panel>
        <Panel title={selected !== null ? `Cell ${selected} inspector` : "Cell inspector"} subtitle="Selected cell · live metadata from the backend" right={<Provenance />}>
          <Inspector id={selected} />
        </Panel>
      </div>
    </>
  );
}

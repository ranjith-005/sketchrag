import { useState } from "react";
import { Link } from "react-router-dom";
import { Boxes, CheckCircle2, CircleDashed, FileSearch, GitBranch, Layers3, Loader2, Lock, ScanSearch, Sparkles, XCircle } from "lucide-react";
import { Bar, Button, KindDot, Note, Offline, PageHeader, Panel, Pill, Provenance } from "../components/ui";
import { fmt, kindColor, queryTerms } from "../lib/api";
import { useLive, usePoll, useQueries } from "../lib/store";

const esc = (s) => String(s).replace(/&amp;/g, "&").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const highlight = (text, words) => words.reduce((out, w) => out.replace(new RegExp(`\\b(${w})\\b`, "gi"), "<mark>$1</mark>"), esc(text));

const STEPS = [
  ["embed", "Embed question", "Create a query embedding and identify rare terms", ScanSearch],
  ["rank", "Rank memory cells", "Score cells using centroids and exemplars", Boxes],
  ["gate", "Evaluate error-bound gate", "Check whether each digest can be trusted", GitBranch],
  ["raw", "Fetch raw evidence", "Read original chunks for escalated cells", FileSearch],
  ["context", "Build context & answer", "Assemble the context and stream the LLM answer", Layers3],
];

function StepBody({ id, ev }) {
  if (id === "rank") return (
    <div className="mt-2 space-y-1">
      {ev.hits.map((h) => (
        <div key={h.cell} className="grid grid-cols-[78px_80px_1fr] items-center gap-2 text-[12px]">
          <span className="font-mono text-ink">cell {h.cell}</span><Bar value={h.sim} max={1} color="#60A5FA" h="h-1.5" />
          <span className="flex min-w-0 items-center gap-1.5 truncate text-ink-muted"><KindDot color={kindColor(h.kind)} />{h.digest}</span>
        </div>))}
    </div>);
  if (id === "gate") return (
    <div className="mt-2 space-y-2 text-[12px]">
      <div className="flex flex-wrap items-center gap-1.5"><span className="text-ink-muted">Rare terms:</span>
        {ev.rare.length ? ev.rare.slice(0, 12).map((t) => <span key={t} className="rounded bg-retrieval/10 px-1.5 py-0.5 font-mono text-retrieval-soft">{t}</span>)
          : <span className="text-ink-dim">none</span>}</div>
      <div className="flex flex-wrap gap-1.5">
        {Object.entries(ev.verdicts).map(([c, v]) => (
          <span key={c} className={`rounded-full px-2 py-0.5 font-semibold ${v === "trusted" ? "bg-teal/10 text-teal" : "bg-fallback/10 text-fallback"}`}>
            cell {c}: {v === "trusted" ? "digest trusted" : `escalate · ${v}`}</span>))}
      </div>
      {ev.reasons[0] && <p className="text-ink-muted">Why: {ev.reasons[0].text}.</p>}
    </div>);
  if (id === "raw" && !ev.skipped) return <p className="mt-1 text-[12px] text-ink-muted">{ev.chunks.length} chunk(s) from {new Set(ev.chunks.map((c) => c.cell)).size} cell(s) — see the evidence panel.</p>;
  if (id === "context") {
    const pct = (100 * ev.tokens) / Math.max(1, ev.full_tokens);
    return (
      <div className="mt-2 text-[12px]">
        <div className="flex items-center gap-2"><Bar value={ev.tokens} max={ev.full_tokens} h="h-1.5" />
          <span className="shrink-0 font-mono text-ink-muted">{pct < 1 ? pct.toFixed(2) : pct.toFixed(1)}% of feed</span></div>
        {ev.has_answer !== null && <p className={`mt-1 ${ev.has_answer ? "text-teal" : "text-fallback"}`}>
          {ev.has_answer ? "The known answer is present in this context." : "The known answer is not in this context."}</p>}
      </div>);
  }
  return null;
}

function Trace({ run }) {
  return (
    <ol className="relative space-y-1">
      {STEPS.map(([id, title, desc, Icon], i) => {
        const ev = run?.steps[id];
        const active = run && run.state === "running" && !ev && STEPS.slice(0, i).every(([p]) => run.steps[p]);
        const tone = ev ? (id === "gate" && ev.fired ? "amber" : id === "raw" && ev.skipped ? "gray" : "teal") : "gray";
        const color = { teal: "text-teal border-teal/50 bg-teal/10", amber: "text-fallback border-fallback/50 bg-fallback/10", gray: "text-ink-dim border-line bg-card" }[tone];
        return (
          <li key={id} className="grid grid-cols-[34px_1fr] gap-3">
            <div className="flex flex-col items-center">
              <span className={`grid h-8 w-8 place-items-center rounded-full border ${color}`}>
                {active ? <Loader2 size={15} className="animate-spin text-fallback" /> : <Icon size={15} />}</span>
              {i < STEPS.length - 1 && <span className="w-px flex-1 bg-line" />}
            </div>
            <div className={`pb-4 ${ev ? "rise" : ""}`}>
              <div className="flex items-center gap-2"><span className="font-mono text-[11px] text-ink-dim">0{i + 1}</span>
                <span className={`text-[14px] font-semibold ${ev ? "text-white" : "text-ink-muted"}`}>{title}</span>
                {ev?.ms !== undefined && <span className="ml-auto font-mono text-[11px] text-ink-dim">{ev.ms} ms</span>}</div>
              <p className="text-[12px] text-ink-muted">{ev ? ev.text : desc}</p>
              {ev && <StepBody id={id} ev={ev} />}
            </div>
          </li>);
      })}
    </ol>
  );
}

function AnswerPanel({ run, onCompare, busy }) {
  if (!run) return (
    <Panel title="Answer & evidence" subtitle="Query output panel">
      <Note>Ask a question to see the agent's answer, the evidence it read and where that evidence came from. Answers are produced live by
        the local LLM — nothing here is pre-written.</Note>
    </Panel>);
  const words = queryTerms(run.question);
  const raw = run.steps.raw?.chunks || [];
  const d = run.done;
  return (
    <Panel title="Answer & evidence" subtitle={`${run.policy === "gate" ? "SketchRAG gate" : "Digest only"} · asked after ${fmt.int(run.start?.words)} words`}
      right={<Provenance />}>
      <div className="rounded-card border border-teal/30 bg-gradient-to-b from-teal/10 to-transparent px-4 py-3">
        <div className="text-[11px] font-bold uppercase tracking-wider text-teal">Answer</div>
        <div className="mt-1 min-h-[28px] text-[18px] font-semibold text-white">
          {run.answer || (run.state === "running" ? <span className="text-[14px] font-normal text-ink-muted">{run.steps.context ? "The LLM is reading the context" : "Working through the trace"}…</span> : null)}
          {run.state === "running" && <span className="caret" />}
        </div>
        {d && (
          <div className="mt-2 flex flex-wrap items-center gap-2 text-[12px] text-ink-muted">
            {d.gold !== undefined && (d.score >= 0.5
              ? <span className="flex items-center gap-1 font-semibold text-teal"><CheckCircle2 size={14} /> matches the known answer: {d.gold}</span>
              : <span className="flex items-center gap-1 font-semibold text-fallback"><XCircle size={14} /> known answer: {d.gold}</span>)}
            <span>{d.model} · {d.llm_s}s · {fmt.int(d.prompt_tokens)} prompt tokens</span>
          </div>)}
        {run.error && <p className="mt-2 text-[13px] text-fallback">{run.error}</p>}
      </div>
      {d && run.policy === "gate" && (
        <Button tone="ghost" className="mt-3" disabled={busy} onClick={onCompare}><Sparkles size={14} /> Compare: what would the digest alone say?</Button>)}
      <div className="mt-4">
        <div className="mb-2 text-[11px] font-bold uppercase tracking-wider text-ink-dim">Evidence read from cold storage · {raw.length} chunk(s)</div>
        {run.steps.raw?.skipped && <p className="text-[12.5px] text-ink-muted">None — the gate trusted the digests, so the answer comes from compact memory only.</p>}
        <div className="max-h-[360px] space-y-2 overflow-auto pr-1">
          {raw.map((c) => (
            <div key={c.chunk} className="rise rounded-ctl border-l-2 border-fallback bg-card px-3 py-2">
              <div className="mb-1 flex items-center gap-2 text-[12px]"><KindDot color={kindColor(c.doc.kind)} />
                <span className="min-w-0 truncate text-ink-muted">Document: <span className="font-semibold text-white">{c.doc.title}</span></span>
                <span className="ml-auto shrink-0 font-mono text-[11px] text-ink-dim">cell {c.cell} · chunk #{c.chunk}</span></div>
              <p className="text-[12.5px] leading-relaxed text-ink-muted" dangerouslySetInnerHTML={{ __html: highlight(c.text, words) }} />
            </div>))}
        </div>
      </div>
    </Panel>
  );
}

export default function Query() {
  const { status, online } = useLive();
  const { runs, ask, busy, latest } = useQueries();
  const [presets] = usePoll("/api/presets", 0);
  const [q, setQ] = useState("");
  const [policy, setPolicy] = useState("gate");
  const [shown, setShown] = useState(null);
  const run = runs.find((r) => r.id === shown) || latest;
  if (!online) return (<><PageHeader title="Query playground" subtitle="Trace how the error-bound gate decides whether raw evidence is needed." /><Offline /></>);
  const go = (text, pol = policy) => { setShown(null); ask(text, pol); };

  return (
    <>
      <PageHeader title="Query playground" subtitle="Trace how the error-bound gate decides whether raw evidence is needed." />
      <div className="grid gap-4 xl:grid-cols-[1fr_1.15fr]">
        <div className="space-y-4">
          <Panel title="Ask the stream" subtitle={`Question answering over the ${status.docs_arrived} documents ingested so far`} right={<Provenance />}>
            <form onSubmit={(e) => { e.preventDefault(); go(q); }}>
              <textarea value={q} onChange={(e) => setQ(e.target.value)} rows={4} placeholder="Ask about anything the agent has read…"
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); go(q); } }}
                className="w-full resize-none rounded-ctl border border-line bg-deep px-3.5 py-3 text-[14.5px] text-white outline-none placeholder:text-ink-dim focus:border-teal/70" />
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <span className="text-[12px] text-ink-muted">Policy</span>
                <div className="flex overflow-hidden rounded-ctl border border-line text-[12.5px]">
                  {[["gate", "Error-bound gate"], ["sketch", "Digest only"]].map(([k, l]) => (
                    <button type="button" key={k} onClick={() => setPolicy(k)}
                      className={`px-3 py-1.5 font-semibold ${policy === k ? "bg-memory/20 text-memory-soft" : "text-ink-muted hover:text-ink"}`}>{l}</button>))}
                </div>
                <Button type="submit" className="ml-auto" disabled={busy || !q.trim()}>
                  {busy ? <><Loader2 size={15} className="animate-spin" /> Agent is thinking</> : "Run query →"}</Button>
              </div>
            </form>
            <p className="mt-3 text-[11.5px] text-ink-dim">While the agent answers, ingestion pauses. Answers come from {presets?.model || "the local LLM"} on this CPU and take about a minute.</p>
          </Panel>

          <Panel title="Suggested questions" subtitle="About documents in this feed · unlock when the document has arrived">
            <div className="space-y-2">
              {(presets?.presets || []).map((p) => {
                const ready = status.pos >= p.available_at;
                return (
                  <button key={p.question} disabled={!ready || busy} onClick={() => { setQ(p.question); go(p.question); }}
                    className="flex w-full items-start gap-3 rounded-ctl border border-line bg-card px-3.5 py-3 text-left text-[13px] transition hover:border-teal/60 disabled:cursor-not-allowed disabled:opacity-45">
                    <KindDot color={kindColor(p.doc_kind)} />
                    <span className="flex-1 text-ink">{p.question}</span>
                    {ready ? <Pill tone={p.gate_fired ? "amber" : "teal"}>{p.gate_fired ? "gate fires" : "digest ok"}</Pill>
                      : <span className="flex shrink-0 items-center gap-1 text-[11px] text-ink-dim"><Lock size={12} /> at {fmt.pct(p.available_at / status.total)}</span>}
                  </button>);
              })}
            </div>
            {presets && (
              <Note>Over all {presets.n_questions} questions about this feed, the gate's context contains the answer for{" "}
                <b>{fmt.pct(presets.end_of_feed_recall.gate)}</b> (dense RAG {fmt.pct(presets.end_of_feed_recall.dense)}, digest only{" "}
                {fmt.pct(presets.end_of_feed_recall.sketch)}). These suggestions are ones {presets.model} answered correctly in rehearsal;
                asked at a different point in the stream, memory differs and a 3B model can still slip.</Note>)}
          </Panel>

          {runs.length > 1 && (
            <Panel title="History" subtitle="Earlier runs this session" bodyClass="px-5 py-2">
              {runs.map((r) => (
                <button key={r.id} onClick={() => setShown(r.id)} className={`flex w-full items-center gap-2 border-b border-line/60 py-2 text-left text-[12.5px] last:border-0 ${r.id === run?.id ? "text-white" : "text-ink-muted"}`}>
                  {r.state === "running" ? <Loader2 size={13} className="animate-spin text-fallback" /> : r.done?.score >= 0.5 ? <CheckCircle2 size={13} className="text-teal" /> : <CircleDashed size={13} />}
                  <span className="min-w-0 flex-1 truncate">{r.question}</span><Pill tone="gray">{r.policy === "gate" ? "gate" : "digest"}</Pill>
                  <span className="max-w-[140px] truncate font-mono text-[11px]">{r.done?.pred || ""}</span>
                </button>))}
            </Panel>)}
        </div>

        <div className="space-y-4">
          <AnswerPanel run={run} busy={busy} onCompare={() => go(run.question, "sketch")} />
          <Panel title="Retrieval trace" subtitle="Step-by-step path through memory"
            right={run?.steps.rank ? <Link to="/memory" className="text-[12px] font-semibold text-teal hover:underline">See cells in memory explorer →</Link> : null}>
            <Trace run={run} />
          </Panel>
        </div>
      </div>
    </>
  );
}

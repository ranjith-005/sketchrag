import { useMemo, useState } from "react";
import { ArrowUpRight, CheckCheck, Coins, ListChecks } from "lucide-react";
import { Bar, Kpi, PageHeader, Panel, Provenance } from "../components/ui";
import { fmt, SYSTEM_COLORS } from "../lib/api";
import { usePoll } from "../lib/store";

const COLS = [["label", "System"], ["score", "Avg score"], ["context_tokens", "Context tokens"], ["llm_s", "LLM s / q"], ["fallback_rate", "Fallback rate"]];

export default function Evaluation() {
  const [ev] = usePoll("/api/evaluation/results", 0);
  const [sort, setSort] = useState(["score", -1]);
  const systems = ev?.systems || [];
  const rows = useMemo(() => [...systems].sort((a, b) => {
    const [k, dir] = sort;
    return typeof a[k] === "string" ? dir * a[k].localeCompare(b[k]) : dir * (a[k] - b[k]);
  }), [systems, sort]);
  if (!ev) return <PageHeader title="Evaluation lab" subtitle="Loading results…" />;
  const gate = systems.find((s) => s.key === "sketchrag_gate");
  const cal = ev.calibration;
  const byScore = [...systems].sort((a, b) => b.score - a.score);
  const byTokens = [...systems].sort((a, b) => a.context_tokens - b.context_tokens);
  const maxTok = Math.max(...systems.map((s) => s.context_tokens));
  const ss = ev.stress_summary;

  return (
    <>
      <PageHeader title="Evaluation lab" subtitle="Compare answer quality, context cost and memory efficiency across systems."
        right={<Provenance kind="report" />} />
      <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
        <Kpi label="Test questions" value={cal.n} sub="12 LongBench / ∞-Bench tasks" icon={ListChecks} />
        <Kpi label="Gate score" value={gate.score} sub="Average score / 100" icon={CheckCheck} tone="blue" />
        <Kpi label="Mean context" value={fmt.int(gate.context_tokens)} sub="LLM tokens per question" icon={Coins} tone="violet" />
        <Kpi label="Gate fallback" value={fmt.pct(cal.fired / cal.n)} sub={`${cal.fired} / ${cal.n} escalated`} icon={ArrowUpRight} tone="amber" />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[1.3fr_1fr]">
        <Panel title="Answer quality by system" subtitle={`Average score across ${cal.n} questions · score out of 100`} right={<Provenance kind="report" />}>
          <div className="space-y-3.5">
            {byScore.map((s) => (
              <div key={s.key} className="grid grid-cols-[170px_1fr_44px] items-center gap-3 text-[13px]">
                <span className={s.key === "sketchrag_gate" ? "font-bold text-white" : "text-ink"}>{s.label}</span>
                <Bar value={s.score} max={50} color={SYSTEM_COLORS[s.label]} h="h-3" />
                <span className="text-right font-mono font-semibold" style={{ color: SYSTEM_COLORS[s.label] }}>{s.score.toFixed(1)}</span>
              </div>))}
          </div>
          <p className="mt-3 text-[11.5px] text-ink-dim">Bar scale 0–50. Every system answers with the same model (qwen2.5:7b-instruct) and prompt; only the context differs.</p>
        </Panel>
        <div className="space-y-4">
          <Panel title="Context cost" subtitle="Mean LLM context tokens per question" right={<Provenance kind="report" />}>
            <div className="space-y-2.5">
              {byTokens.map((s) => (
                <div key={s.key} className="grid grid-cols-[150px_1fr_56px] items-center gap-3 text-[12.5px]">
                  <span className="text-ink">{s.label}</span><Bar value={s.context_tokens} max={maxTok} color={SYSTEM_COLORS[s.label]} h="h-2" />
                  <span className="text-right font-mono" style={{ color: SYSTEM_COLORS[s.label] }}>{fmt.int(s.context_tokens)}</span>
                </div>))}
            </div>
          </Panel>
          <Panel title="Research interpretation" subtitle="What the metrics say">
            <div className="space-y-2.5 text-[13px] leading-relaxed text-ink">
              <p>SketchRAG Gate keeps competitive quality ({gate.score}) at the lowest context cost ({fmt.int(gate.context_tokens)} tokens).
                BM25 RAG has the highest average ({byScore[0].score}).</p>
              <p className="text-ink-muted">The gate is calibrated: on the {cal.fired} questions it escalates, raw evidence lifts the score from{" "}
                {cal.fired_sketch.toFixed(1)} to {cal.fired_gate.toFixed(1)}; on the {cal.trusted} it trusts, always reading raw would add only{" "}
                {(cal.trusted_raw - cal.trusted_gate).toFixed(1)} points.</p>
              <p className="text-ink-muted">Long stream ({fmt.int(ss.n_words)} words): hot memory stays small, but answer recall falls to{" "}
                {ss.recall_sketchrag_gate.toFixed(2)} against {ss.recall_dense_rag.toFixed(2)} for dense RAG, as many unrelated documents share few cells.</p>
            </div>
          </Panel>
        </div>
      </div>

      <Panel className="mt-4" title="Detailed results" subtitle="Click a column to sort" right={<Provenance kind="report" />} bodyClass="p-0">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-[13px]">
            <thead className="text-left text-[11px] uppercase tracking-wider text-ink-dim">
              <tr>{COLS.map(([k, l]) => (
                <th key={k} className={`cursor-pointer select-none px-5 py-3 font-semibold hover:text-ink ${k !== "label" ? "text-right" : ""}`}
                  onClick={() => setSort(([sk, d]) => [k, sk === k ? -d : k === "label" ? 1 : -1])}>
                  {l}{sort[0] === k ? (sort[1] > 0 ? " ▲" : " ▼") : ""}</th>))}</tr>
            </thead>
            <tbody>{rows.map((s) => (
              <tr key={s.key} className={`border-t border-line/60 ${s.key === "sketchrag_gate" ? "bg-teal/5" : ""}`}>
                <td className="px-5 py-2.5 font-semibold text-white"><span className="mr-2 inline-block h-2.5 w-2.5 rounded-full" style={{ background: SYSTEM_COLORS[s.label] }} />{s.label}</td>
                <td className="px-5 py-2.5 text-right font-mono">{s.score.toFixed(1)}</td>
                <td className="px-5 py-2.5 text-right font-mono">{fmt.int(s.context_tokens)}</td>
                <td className="px-5 py-2.5 text-right font-mono">{s.llm_s.toFixed(1)}</td>
                <td className="px-5 py-2.5 text-right font-mono">{fmt.pct(s.fallback_rate)}</td>
              </tr>))}</tbody>
          </table>
        </div>
        <p className="border-t border-line px-5 py-3 text-[11.5px] text-ink-dim">Source: {ev.source}. These are report results, not live evaluation.</p>
      </Panel>
    </>
  );
}

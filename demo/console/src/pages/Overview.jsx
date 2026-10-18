import { useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, ArrowRightLeft, Boxes, Layers, TrendingUp } from "lucide-react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Bar, Button, Kpi, KindDot, Offline, PageHeader, Panel, Provenance } from "../components/ui";
import { fmt, kindColor } from "../lib/api";
import { useLive, usePoll } from "../lib/store";

const tooltipStyle = { background: "#151F30", border: "1px solid #233248", borderRadius: 8, fontSize: 12 };

function FootprintChart({ evaluation }) {
  const [view, setView] = useState("live");
  const [history] = usePoll("/api/memory/history", view === "live" ? 2000 : 0, [view]);
  let data = [], lines = [], xlabel = "";
  if (view === "live") {
    data = (history || []).map((h) => ({ x: h.words / 1000, hot: h.hot / 1048576, dense: h.dense / 1048576, text: h.text / 1048576 }));
    lines = [["hot", "SketchRAG hot memory", "#2DD4BF"], ["dense", "Dense RAG index", "#60A5FA"], ["text", "Raw text", "#FBBF24"]];
    xlabel = "thousand words streamed";
  } else if (evaluation) {
    const c64 = evaluation.stress_curves["64"], c256 = evaluation.stress_curves["256"];
    data = c256.map((c, i) => ({ x: c.words / 1e6, hot64: c64[i].sketch_hot_bytes / 1048576, hot256: c.sketch_hot_bytes / 1048576,
      dense: c.dense_bytes / 1048576, text: c.text_bytes / 1048576 }));
    lines = [["hot64", "SketchRAG hot memory (64 cells)", "#2DD4BF"], ["hot256", "SketchRAG hot memory (256 cells)", "#4CE0B3"],
      ["dense", "Dense RAG index", "#60A5FA"], ["text", "Raw text", "#FBBF24"]];
    xlabel = "million words streamed";
  }
  return (
    <Panel title="Memory footprint over stream length"
      subtitle={view === "live" ? "This feed, measured by the running backend" : "8.5M-word long-stream stress test (results/stress)"}
      right={<div className="flex items-center gap-3">
        <Provenance kind={view === "live" ? "live" : "report"} />
        <div className="flex overflow-hidden rounded-ctl border border-line text-[12px]">
          {[["live", "This feed"], ["report", "Stress test"]].map(([k, l]) => (
            <button key={k} onClick={() => setView(k)} className={`px-3 py-1.5 font-semibold ${view === k ? "bg-teal text-app" : "text-ink-muted hover:text-ink"}`}>{l}</button>))}
        </div></div>}
      className="xl:col-span-3">
      <div className="h-[290px]">
        {data.length ? (
          <ResponsiveContainer>
            <LineChart data={data} margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
              <CartesianGrid stroke="#1B2638" />
              <XAxis dataKey="x" type="number" domain={[0, "dataMax"]} tick={{ fill: "#8A9AB2", fontSize: 11 }} stroke="#233248"
                tickFormatter={(v) => (view === "live" ? `${Math.round(v)}k` : `${v.toFixed(1)}M`)}
                label={{ value: xlabel, position: "insideBottom", offset: -10, fill: "#5D6B82", fontSize: 11 }} />
              <YAxis tick={{ fill: "#8A9AB2", fontSize: 11 }} stroke="#233248" width={52} tickFormatter={(v) => `${v.toFixed(v < 10 ? 1 : 0)} MB`} />
              <Tooltip contentStyle={tooltipStyle} labelFormatter={(v) => (view === "live" ? `${Math.round(v)}k words` : `${v.toFixed(2)}M words`)}
                formatter={(v) => `${v.toFixed(2)} MB`} />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 6 }} verticalAlign="top" height={30} />
              {lines.map(([k, n, c]) => <Line key={k} dataKey={k} name={n} stroke={c} strokeWidth={2.2} dot={false} isAnimationActive={false} />)}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="grid h-full place-items-center text-[13px] text-ink-muted">Waiting for the stream…</div>}
      </div>
    </Panel>
  );
}

export default function Overview() {
  const { status, health, online } = useLive();
  const [docs] = usePoll("/api/stream/documents", 2000);
  const [evaluation] = usePoll("/api/evaluation/results", 0);
  if (!online) return (<><PageHeader title="Overview" subtitle="Monitor ingestion, bounded memory and retrieval health from one workspace." /><Offline /></>);
  const gate = evaluation?.systems.find((s) => s.key === "sketchrag_gate");
  const bm25 = evaluation?.systems.find((s) => s.key === "bm25_rag");
  const recent = (docs || []).filter((d) => d.status !== "queued").slice(-5).reverse();
  const layers = [["Hot memory", status.hot_bytes, "#2DD4BF", "Cells · sketches · digests · exemplars — the only part that is searched"],
    ["Cold storage", status.cold_bytes, "#FBBF24", "Original chunks behind pointers — read only when the gate escalates; grows with the stream"],
    ["Dense RAG index", status.dense_bytes, "#60A5FA", "What a vector store would hold for the same text (text + one 384-d vector per chunk)"]];
  const top = Math.max(...layers.map((l) => l[1]), 1);

  return (
    <>
      <PageHeader title="Overview" subtitle="Monitor ingestion, bounded memory and retrieval health from one workspace."
        right={<Link to="/query"><Button>Ask the agent <ArrowRight size={15} /></Button></Link>} />

      <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
        <Kpi label="Words processed" value={fmt.int(status.words)} sub={`Across ${status.docs_arrived} of ${status.docs_total} documents`} icon={TrendingUp} />
        <Kpi label="Memory cells" value={`${status.cells} / ${status.max_cells}`} sub="Configured cell budget" icon={Boxes} tone="violet" />
        <Kpi label="Chunks" value={fmt.int(status.pos)} sub={`Ingested of ${fmt.int(status.total)}`} icon={Layers} tone="blue" />
        <Kpi label="Cell merges" value={fmt.int(status.merges)} sub="Merge-and-reduce operations" icon={ArrowRightLeft} tone="amber" />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-5">
        <FootprintChart evaluation={evaluation} />
        <Panel title="Memory layers" subtitle="Searchable summary vs. evidence storage" right={<Provenance />} className="xl:col-span-2">
          <div className="space-y-5">
            {layers.map(([name, v, c, d]) => (
              <div key={name}>
                <div className="mb-1.5 flex justify-between text-[14px] font-semibold text-white"><span>{name}</span>
                  <span className="font-mono" style={{ color: c }}>{fmt.bytes(v)}</span></div>
                <Bar value={v} max={top} color={c} />
                <p className="mt-1.5 text-[11.5px] text-ink-muted">{d}</p>
              </div>
            ))}
            <p className="border-t border-line pt-3 text-[11.5px] text-ink-dim">Bounded hot memory does not mean bounded storage: cold storage still grows with the raw text.</p>
          </div>
        </Panel>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-5">
        <Panel title="Recent ingestion" subtitle="Latest documents entering the stream" right={<Provenance />} className="xl:col-span-3" bodyClass="px-5 py-2">
          {recent.map((d) => (
            <div key={d.key} className="flex items-center gap-4 border-b border-line/60 py-2.5 text-[13px] last:border-0">
              <KindDot kind={d.kind} color={kindColor(d.kind)} />
              <span className="w-24 shrink-0 text-[11px] font-bold uppercase tracking-wide" style={{ color: kindColor(d.kind) }}>{d.kind}</span>
              <span className="min-w-0 flex-1 truncate text-ink">{d.title}</span>
              <span className="shrink-0 font-mono text-[11.5px] text-ink-muted">{fmt.int(d.words)} words</span>
              <span className={`w-20 shrink-0 text-right text-[11px] font-semibold uppercase ${d.status === "ingesting" ? "text-fallback" : "text-teal"}`}>{d.status}</span>
            </div>
          ))}
          <Link to="/stream" className="mt-2 inline-flex items-center gap-1 pb-2 text-[12.5px] font-semibold text-teal hover:underline">Open live stream <ArrowRight size={13} /></Link>
        </Panel>

        <Panel title="Research snapshot" subtitle="Benchmark over 350 test questions" right={<Provenance kind="report" />} className="xl:col-span-2">
          {gate && (
            <>
              <div className="flex items-end justify-between"><span className="text-[15px] font-semibold text-white">SketchRAG · gate</span>
                <span className="font-mono text-[30px] leading-none text-teal">{gate.score}</span></div>
              <p className="mb-2 mt-1 text-[12px] text-ink-muted">Average answer score out of 100 · BM25 RAG {bm25.score} is the strongest system</p>
              <Bar value={gate.score} max={100} />
              <div className="mt-4 grid grid-cols-2 gap-3 text-[12px]">
                <div className="rounded-ctl bg-card px-3 py-2"><div className="font-mono text-[17px] text-retrieval">{fmt.int(gate.context_tokens)}</div>LLM tokens / question</div>
                <div className="rounded-ctl bg-card px-3 py-2"><div className="font-mono text-[17px] text-fallback">{fmt.pct(gate.fallback_rate)}</div>gate fallback rate</div>
              </div>
            </>
          )}
          <div className="mt-4 space-y-1.5 border-t border-line pt-3">
            {health && ["ingestion", "retrieval", "llm"].map((k) => (
              <div key={k} className="flex items-center gap-2 text-[12px]">
                <span className={`h-2 w-2 rounded-full ${health[k].ok ? "bg-teal" : "bg-fallback"}`} />
                <span className="w-20 font-semibold capitalize text-ink">{k === "llm" ? "LLM" : k}</span>
                <span className="truncate text-ink-muted">{health[k].detail}</span>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </>
  );
}

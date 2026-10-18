import { Info } from "lucide-react";

export function PageHeader({ title, subtitle, right }) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-teal">Streaming memory intelligence</div>
        <h1 className="mt-1 text-[28px] font-bold leading-tight text-white">{title}</h1>
        <p className="mt-1 text-[14px] text-ink-muted">{subtitle}</p>
      </div>
      {right}
    </div>
  );
}

/** Where a number comes from (spec section 6): live telemetry from the backend, or a report result from results/. */
export function Provenance({ kind = "live" }) {
  const map = {
    live: ["Live telemetry", "border-teal/40 text-teal bg-teal/10"],
    report: ["Report result", "border-memory/40 text-memory-soft bg-memory/10"],
    offline: ["Backend offline", "border-fallback/40 text-fallback bg-fallback/10"],
  };
  const [label, cls] = map[kind];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[10.5px] font-semibold uppercase tracking-wider ${cls}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${kind === "live" ? "bg-teal animate-pulse" : "bg-current"}`} />
      {label}
    </span>
  );
}

export function Panel({ title, subtitle, right, children, className = "", bodyClass = "p-5" }) {
  return (
    <section className={`rounded-card border border-line bg-panel ${className}`}>
      {(title || right) && (
        <header className="flex items-start justify-between gap-3 border-b border-line px-5 py-3.5">
          <div>
            <h2 className="text-[15px] font-semibold text-white">{title}</h2>
            {subtitle && <p className="mt-0.5 text-[12px] text-ink-muted">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      <div className={bodyClass}>{children}</div>
    </section>
  );
}

const TONES = {
  teal: "text-teal bg-teal/10", blue: "text-retrieval bg-retrieval/10", violet: "text-memory bg-memory/10",
  amber: "text-fallback bg-fallback/10", pink: "text-story bg-story/10",
};

export function Kpi({ label, value, sub, icon: Icon, tone = "teal", help }) {
  return (
    <div className="rounded-card border border-line bg-panel px-5 py-4" title={help}>
      <div className="flex items-start justify-between">
        <span className="text-[13px] text-ink-muted">{label}</span>
        {Icon && <span className={`grid h-8 w-8 place-items-center rounded-ctl ${TONES[tone]}`}><Icon size={16} /></span>}
      </div>
      <div className="mt-2 font-mono text-[26px] font-medium leading-none text-white">{value}</div>
      {sub && <div className={`mt-2.5 text-[12px] ${TONES[tone].split(" ")[0]}`}>{sub}</div>}
    </div>
  );
}

export function Bar({ value, max = 1, color = "#2DD4BF", h = "h-2" }) {
  const w = Math.max(0, Math.min(100, (100 * value) / (max || 1)));
  return (
    <div className={`w-full overflow-hidden rounded-full bg-line/70 ${h}`}>
      <div className="h-full rounded-full transition-all duration-500" style={{ width: `${w}%`, background: color }} />
    </div>
  );
}

export function Pill({ children, tone = "teal", className = "" }) {
  const t = {
    teal: "border-teal/50 text-teal", amber: "border-fallback/50 text-fallback", blue: "border-retrieval/50 text-retrieval",
    violet: "border-memory/50 text-memory-soft", red: "border-rose-400/50 text-rose-300", gray: "border-line text-ink-muted",
  }[tone];
  return <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide ${t} ${className}`}>{children}</span>;
}

export function Button({ children, onClick, tone = "teal", disabled, className = "", type = "button", title }) {
  const t = {
    teal: "bg-teal text-app hover:bg-teal-soft", ghost: "border border-line bg-card text-ink hover:border-teal/60",
    amber: "border border-fallback/50 bg-fallback/10 text-fallback hover:bg-fallback/20",
  }[tone];
  return (
    <button type={type} title={title} disabled={disabled} onClick={onClick}
      className={`inline-flex items-center justify-center gap-2 rounded-ctl px-3.5 py-2 text-[13px] font-semibold transition disabled:cursor-not-allowed disabled:opacity-40 ${t} ${className}`}>
      {children}
    </button>
  );
}

export function Note({ children, tone = "info" }) {
  const cls = tone === "warn" ? "border-fallback/40 bg-fallback/10 text-fallback-soft" : "border-line bg-card text-ink-muted";
  return (
    <div className={`flex gap-2 rounded-ctl border px-3 py-2 text-[12.5px] ${cls}`}>
      <Info size={15} className="mt-0.5 shrink-0" /> <div>{children}</div>
    </div>
  );
}

export function KindDot({ kind, color }) {
  return <span className="inline-block h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: color }} title={kind} />;
}

export function Offline() {
  return (
    <Panel title="Backend not reachable">
      <p className="text-[14px] text-ink-muted">
        Start it with <code className="rounded bg-card px-1.5 py-0.5 font-mono text-teal">python demo/agent/server.py</code> in the
        project folder. Nothing on this page is shown until it answers a real health check.
      </p>
    </Panel>
  );
}

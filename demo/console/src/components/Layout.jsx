import { useState } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import {
  Activity, BarChart3, Boxes, LayoutDashboard, Menu, MessageSquareText, Search, Settings2, Sparkles, Waves, X,
} from "lucide-react";
import { useLive, useQueries } from "../lib/store";
import { fmt } from "../lib/api";

export const NAV = [
  { group: "Workspace", items: [
    { to: "/", label: "Overview", icon: LayoutDashboard },
    { to: "/stream", label: "Live stream", icon: Waves },
    { to: "/memory", label: "Memory explorer", icon: Boxes },
    { to: "/query", label: "Query playground", icon: MessageSquareText },
  ] },
  { group: "Evaluation", items: [{ to: "/evaluation", label: "Evaluation lab", icon: BarChart3 }] },
  { group: "System", items: [{ to: "/config", label: "Configuration & logs", icon: Settings2 }] },
];

function Logo() {
  return (
    <div className="flex items-center gap-3 px-5 py-5">
      <div className="grid h-9 w-9 place-items-center rounded-[10px] bg-teal">
        <svg viewBox="0 0 32 32" className="h-6 w-6"><circle cx="10" cy="10" r="3.2" fill="#0B1120" /><circle cx="22" cy="22" r="3.2" fill="#0B1120" />
          <circle cx="22" cy="10" r="2.2" fill="#0B1120" /><circle cx="10" cy="22" r="2.2" fill="#0B1120" />
          <path d="M10 10 22 22M22 10 10 22" stroke="#0B1120" strokeWidth="2.2" /></svg>
      </div>
      <div>
        <div className="text-[18px] font-bold leading-none text-white">SketchRAG</div>
        <div className="mt-1 text-[10.5px] font-semibold uppercase tracking-[0.12em] text-ink-muted">Agent console</div>
      </div>
    </div>
  );
}

function EnvCard() {
  const { online, health, status } = useLive();
  const llm = health?.llm?.ok;
  return (
    <div className="mx-4 rounded-card border border-line bg-panel px-4 py-3">
      <div className="flex items-center gap-2 text-[13.5px] font-semibold text-white">
        <span className={`h-2 w-2 rounded-full ${online ? "bg-teal animate-pulse" : "bg-fallback"}`} />
        {online ? "Live backend" : "Backend offline"}
      </div>
      <div className="mt-1 text-[11.5px] leading-snug text-ink-muted">
        {online ? <>Agent {status.state} · {fmt.pct(status.progress)} of feed<br />LLM {llm ? "ready" : "unavailable"}</> : "python demo/agent/server.py"}
      </div>
    </div>
  );
}

function SideNav({ onPick }) {
  const { busy } = useQueries();
  return (
    <nav className="mt-4 flex-1 space-y-5 px-3">
      {NAV.map((g) => (
        <div key={g.group}>
          <div className="px-3 pb-2 text-[10.5px] font-semibold uppercase tracking-[0.12em] text-ink-dim">{g.group}</div>
          {g.items.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} end={to === "/"} onClick={onPick}
              className={({ isActive }) => `mb-1 flex items-center gap-3 rounded-ctl border-l-2 px-3 py-2.5 text-[14px] transition ${
                isActive ? "border-teal bg-teal/10 font-semibold text-white" : "border-transparent text-ink-muted hover:bg-card hover:text-ink"}`}>
              <Icon size={17} /> {label}
              {to === "/query" && busy && <span className="ml-auto h-2 w-2 animate-ping rounded-full bg-fallback" title="query running" />}
            </NavLink>
          ))}
        </div>
      ))}
    </nav>
  );
}

function Sidebar({ open, onClose }) {
  const body = (
    <div className="flex h-full flex-col">
      <Logo />
      <EnvCard />
      <SideNav onPick={onClose} />
      <div className="m-4 rounded-card border border-teal/30 bg-teal/5 px-4 py-3">
        <div className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-teal"><Sparkles size={12} /> Bounded by design</div>
        <p className="mt-1.5 text-[11.5px] leading-relaxed text-ink-muted">Compact summary for a growing stream. Raw evidence stays available on demand.</p>
      </div>
    </div>
  );
  return (
    <>
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-[248px] border-r border-line bg-deep lg:block">{body}</aside>
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/60" onClick={onClose} />
          <aside className="absolute inset-y-0 left-0 w-[260px] border-r border-line bg-deep">
            <button className="absolute right-3 top-5 text-ink-muted" onClick={onClose}><X size={20} /></button>
            {body}
          </aside>
        </div>
      )}
    </>
  );
}

function TopBar({ onMenu }) {
  const { online, status } = useLive();
  const loc = useLocation();
  const nav = useNavigate();
  const [q, setQ] = useState("");
  const page = NAV.flatMap((g) => g.items).find((i) => (i.to === "/" ? loc.pathname === "/" : loc.pathname.startsWith(i.to)));
  return (
    <header className="sticky top-0 z-20 flex h-[66px] items-center gap-4 border-b border-line bg-app/95 px-5 backdrop-blur lg:px-8">
      <button className="text-ink-muted lg:hidden" onClick={onMenu}><Menu size={22} /></button>
      <div className="text-[13px] text-ink-muted">SketchRAG <span className="mx-2 text-ink-dim">/</span>
        <span className="font-semibold text-white">{page?.label || "Overview"}</span></div>
      <form className="ml-auto hidden w-[300px] items-center gap-2 rounded-ctl border border-line bg-panel px-3 py-2 md:flex"
        onSubmit={(e) => { e.preventDefault(); nav(`/stream?search=${encodeURIComponent(q)}`); }}>
        <Search size={15} className="text-ink-dim" />
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search documents in the feed…"
          className="w-full bg-transparent text-[13px] text-ink outline-none placeholder:text-ink-dim" />
      </form>
      <div className={`flex items-center gap-2 rounded-full border px-3 py-1.5 text-[11px] font-bold uppercase tracking-wider ${
        online ? "border-teal/40 bg-teal/10 text-teal" : "border-fallback/40 bg-fallback/10 text-fallback"}`}>
        <Activity size={13} /> {online ? (status.state === "thinking" ? "Agent thinking" : "Live") : "Offline"}
      </div>
    </header>
  );
}

export default function Layout({ children }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="min-h-screen bg-app text-ink">
      <Sidebar open={open} onClose={() => setOpen(false)} />
      <div className="lg:pl-[248px]">
        <TopBar onMenu={() => setOpen(true)} />
        <main className="mx-auto max-w-[1500px] px-5 py-7 lg:px-8">{children}</main>
      </div>
    </div>
  );
}


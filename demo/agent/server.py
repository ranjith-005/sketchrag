"""SketchRAG console backend: one process serving the live feed, the agent, the results and the UI.

  python demo/agent/server.py            # http://localhost:8600

The agent monitors a feed of documents (feed.py). /api/ask answers a question as a stream of
Server-Sent Events, one per step of the real pipeline (embed -> rank cells -> error-bound gate
-> raw fetch -> context -> LLM tokens). Everything under /api/stream, /api/memory and /api/events
is live telemetry from that process; /api/evaluation is computed from results/ (report results).
The React console (demo/console) is served from demo/console/dist.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # bge-small is in the local cache; skip hub checks
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import demo_core as dc  # noqa: E402
from feed import ROOT, Feed  # noqa: E402
from sketchrag.eval.metrics import answer_in_context, score  # noqa: E402
from sketchrag.memory import SketchConfig  # noqa: E402
from sketchrag.prompts import MAX_TOKENS, SYSTEM, user_prompt  # noqa: E402
from sketchrag.retrieve import RetrieveConfig, build_context, fetch_raw  # noqa: E402
from sketchrag.text import est_tokens, terms  # noqa: E402

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").replace("//localhost", "//127.0.0.1")
MODEL = os.environ.get("SKETCHRAG_MODEL", "llama3.2:3b")
PRESETS = json.loads((HERE / "presets.json").read_text(encoding="utf-8"))
PRESET_BY_Q = {p["question"]: p for p in PRESETS["presets"]}
DIST = HERE.parent / "console" / "dist"

feed = Feed(max_cells=PRESETS["max_cells"])
ask_lock = threading.Lock()
last_query: dict = {}
app = FastAPI(title="SketchRAG console API")


# ---------------- health ----------------
_llm_health = {"checked": 0.0, "ok": False, "detail": "not checked yet"}


def llm_health() -> dict:
    if time.time() - _llm_health["checked"] > 10:
        try:
            with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=2) as r:
                names = [m["name"] for m in json.load(r)["models"]]
            ok = MODEL in names
            _llm_health.update(ok=ok, detail=f"{MODEL} ready" if ok else f"{MODEL} not pulled (have: {', '.join(names)})")
        except Exception as e:  # noqa: BLE001
            _llm_health.update(ok=False, detail=f"Ollama unreachable at {OLLAMA}: {type(e).__name__}")
        _llm_health["checked"] = time.time()
    return {"ok": _llm_health["ok"], "detail": _llm_health["detail"]}


@app.get("/api/health")
def health():
    st = feed.status()
    ingest_ok = time.time() - feed.last_tick < 2
    return {
        "server": {"ok": True, "detail": "FastAPI on this machine"},
        "ingestion": {"ok": ingest_ok, "detail": f"{st['state']} · {st['pos']:,}/{st['total']:,} chunks" if ingest_ok
                      else "ingest thread stalled"},
        "retrieval": {"ok": st["cells"] > 0,
                      "detail": f"{st['cells']} cells searchable · bge-small "
                                + ("loaded" if feed.embedder._model is not None else "loading (typed questions wait for it)")},
        "llm": llm_health(),
        "time": time.time(),
    }


# ---------------- stream ----------------
@app.get("/api/stream/status")
def stream_status():
    return feed.status()


@app.get("/api/stream/documents")
def stream_documents():
    return feed.documents()


@app.post("/api/stream/control")
def stream_control(cmd: dict):
    action = cmd.get("action")
    with feed.lock:
        if action == "play":
            feed.playing = True
            feed.event("info", "control", "ingestion resumed")
        elif action == "pause":
            feed.playing = False
            feed.event("info", "control", "ingestion paused")
        elif action == "speed":
            feed.speed = max(1, min(64, int(cmd.get("value", 1))))
            feed.event("info", "control", f"ingestion speed set to {feed.speed}x")
        elif action == "restart":
            feed._reset_memory()
            feed.playing = True
    return feed.status()


# ---------------- memory ----------------
@app.get("/api/memory/summary")
def memory_summary():
    st = feed.status()
    return {k: st[k] for k in ("cells", "max_cells", "merges", "hot_bytes", "cold_bytes", "dense_bytes", "text_bytes",
                               "pos", "words")} | {"digest_words": feed.cfg.digest_words, "cms_width": feed.cfg.cms_width,
                                                   "cms_depth": feed.cfg.cms_depth, "topics": feed.topics(15)}


@app.get("/api/memory/history")
def memory_history():
    with feed.lock:
        return list(feed.history)


@app.get("/api/memory/cells")
def memory_cells():
    return feed.cells()


@app.get("/api/memory/cells/{cell_id}")
def memory_cell(cell_id: int):
    d = feed.cell_detail(cell_id)
    return d if d else JSONResponse({"error": f"cell {cell_id} no longer exists (merged)"}, status_code=404)


# ---------------- config & events ----------------
@app.get("/api/config")
def get_config():
    c, d = feed.cfg, feed.default_cfg
    return {"max_cells": c.max_cells, "cms_width": c.cms_width, "join_threshold": c.join_threshold,
            "digest_words": c.digest_words, "defaults": {"max_cells": d.max_cells, "cms_width": d.cms_width,
                                                        "join_threshold": d.join_threshold},
            "presets_rehearsed_with": PRESETS["max_cells"], "model": MODEL}


@app.post("/api/config")
def set_config(body: dict):
    """Apply a new memory configuration. This really rebuilds the agent's memory and restarts the feed."""
    cfg = SketchConfig(max_cells=int(body.get("max_cells", feed.cfg.max_cells)),
                       cms_width=int(body.get("cms_width", feed.cfg.cms_width)),
                       join_threshold=float(body.get("join_threshold", feed.cfg.join_threshold)))
    if not (8 <= cfg.max_cells <= 1024 and 16 <= cfg.cms_width <= 8192 and 0.5 <= cfg.join_threshold <= 0.99):
        return JSONResponse({"error": "value out of range"}, status_code=400)
    feed._reset_memory(cfg)
    feed.playing = True
    return get_config()


@app.get("/api/events")
def events(limit: int = 60):
    with feed.lock:
        return list(feed.events)[:limit]


# ---------------- evaluation (report results, from results/) ----------------
def _rows(run: str) -> list[dict]:
    out = []
    for p in sorted((ROOT / "results" / "runs" / run).glob("*.jsonl")):
        out += [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]
    return out


def _evaluation() -> dict:
    by = defaultdict(lambda: defaultdict(list))
    for r in _rows("main"):
        by[r["system"]][r["task"]].append(r)
    systems = []
    for s, tasks in by.items():
        allr = [r for rs in tasks.values() for r in rs]
        systems.append({"key": s, "label": dc.SYSTEM_LABELS.get(s, s),
                        "score": round(100 * sum(sum(r["score"] for r in rs) / len(rs) for rs in tasks.values()) / len(tasks), 1),
                        "context_tokens": round(sum(r["context_tokens"] for r in allr) / len(allr)),
                        "llm_s": round(sum(r["llm_s"] for r in allr) / len(allr), 1),
                        "fallback_rate": round(sum(bool(r.get("fallback")) for r in allr) / len(allr), 2),
                        "questions": len(allr)})
    systems.sort(key=lambda x: -x["score"])
    cal = dc.gate_calibration(dc.load_run("main"))
    stress = {}
    for cells in (64, 256):
        p = ROOT / "results" / "stress" / f"cells_{cells}" / "curve_8488kw.json"
        stress[str(cells)] = json.loads(p.read_text())
    summary = json.loads((ROOT / "results" / "stress" / "cells_256" / "summary_8488kw.json").read_text())
    return {"provenance": "report", "source": "results/runs/main (qwen2.5:7b-instruct, 12 LongBench / InfiniteBench tasks)",
            "systems": systems, "calibration": cal, "stress_curves": stress,
            "stress_summary": {k: summary[k] for k in summary if k.startswith("recall_") or k in ("n_words", "n_chunks", "gate_fallback_rate")}}


EVALUATION = _evaluation()


@app.get("/api/evaluation/results")
def evaluation():
    return EVALUATION


# ---------------- query ----------------
@app.get("/api/presets")
def presets():
    return {**{k: v for k, v in PRESETS.items() if k != "presets"}, "model": MODEL,
            "presets": [{k: p.get(k) for k in ("question", "answers", "doc_icon", "doc_kind", "doc_title",
                                                "available_at", "gate_fired", "rehearsal")} for p in PRESETS["presets"]]}


@app.get("/api/query/last")
def query_last():
    return last_query


def _sse(kind: str, **data) -> str:
    return "data: " + json.dumps({"type": kind, **data}) + "\n\n"


def _doc_card(chunk_id: int) -> dict:
    d = feed.doc_of[feed.chunks[chunk_id].doc_id]
    return {"kind": d.kind, "title": d.title}


def _stream_llm(prompt: str, max_tokens: int):
    body = json.dumps({"model": MODEL, "stream": True, "keep_alive": "60m",
                       "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                       "options": {"num_ctx": 8192, "num_predict": max_tokens, "temperature": 0, "seed": 0}}).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.strip():
                yield json.loads(line)


def run_agent(question: str, policy: str):
    global last_query
    preset = PRESET_BY_Q.get(question.strip())
    kind = preset["kind"] if preset else "qa"
    answers = preset["answers"] if preset else []
    t_all = time.perf_counter()
    feed.thinking = True
    try:
        words_seen = int(feed.words[feed.pos - 1]) if feed.pos else 0
        yield _sse("start", question=question, policy=policy, pos=feed.pos, words=words_seen)
        with feed.lock:  # barrier: wait for an in-flight ingest batch; thinking=True keeps the feed paused
            pass
        mem = feed.mem
        if not mem.cells:
            yield _sse("error", message="The agent has not read anything yet. Resume the stream first.")
            return
        feed.event("info", "query", f"query received: {question[:80]}", f"policy {policy}")
        t0 = time.perf_counter()
        q_emb = feed.embedder.embed_queries([question])[0]
        yield _sse("step", step="embed", ms=round(1000 * (time.perf_counter() - t0), 1),
                   text="Embedded the question with bge-small (384 dimensions)")

        t0 = time.perf_counter()
        g = dc.explain_gate(mem, question, q_emb)
        kinds = feed.cell_kinds()
        hits = [{"cell": h.cell.id, "sim": round(h.dense, 3), "n": h.cell.n, "span": [h.cell.t_first, h.cell.t_last],
                 "kind": kinds.get(h.cell.id, "?"), "digest": h.cell.digest_text()[:260]} for h in g["hits"]]
        yield _sse("step", step="rank", ms=round(1000 * (time.perf_counter() - t0), 1), hits=hits,
                   text=f"Ranked {len(mem.cells)} memory cells by centroid and exemplar similarity; kept {len(hits)}")

        dec = g["decision"]
        verdicts = {str(h["cell"]): dec.reasons.get(h["cell"], "trusted") for h in hits}
        reasons = [{"cell": c, "why": w, "text": dc.REASON_TEXT[w]} for c, w in dec.reasons.items()]
        fired = bool(dec.fallback_cells) and policy == "gate"
        yield _sse("step", step="gate", rare=g["rare"], verdicts=verdicts, reasons=reasons, fired=fired,
                   terms=[{"term": r["term"], "cell": r["cell"], "est": r["CMS estimate"], "above_noise": r["above noise"],
                           "above_bound": r["above εN"], "in_digest": r["in digest"]} for r in g["term_rows"]],
                   text=("No rare query terms, so every digest is trusted" if not g["rare"] else
                         (f"Escalate {len(dec.fallback_cells)} of {len(hits)} cells: their digests cannot vouch for the rare terms"
                          if dec.fallback_cells else "Every digest vouches for the rare query terms"))
                   + (" (digest-only policy: escalation skipped)" if policy == "sketch" and dec.fallback_cells else ""))
        if fired:
            feed.event("warn", "gate", f"error-bound gate escalated {len(dec.fallback_cells)} cell(s)",
                       ", ".join(f"cell {r['cell']}: {r['why']}" for r in reasons[:4]))

        cfg = RetrieveConfig()
        raw = []
        if policy == "gate" and dec.fallback_cells:
            t0 = time.perf_counter()
            for cid in dec.fallback_cells:
                for chunk_id, text in fetch_raw(mem, cid, question, cfg.raw_per_cell, q_emb):
                    raw.append({"cell": cid, "chunk": chunk_id, "doc": _doc_card(chunk_id), "text": text[:900]})
            yield _sse("step", step="raw", ms=round(1000 * (time.perf_counter() - t0), 1), chunks=raw,
                       text=f"Read {len(raw)} raw chunk(s) from cold storage")
        else:
            yield _sse("step", step="raw", skipped=True, chunks=[],
                       text="No raw retrieval: answering from the compact digests")
        ctx, info = build_context(mem, question, q_emb, cfg, policy)
        ctx_tokens = est_tokens(ctx)
        full_tokens = int(round(words_seen * 1.33))
        yield _sse("step", step="context", words=len(ctx.split()), tokens=ctx_tokens, full_tokens=full_tokens,
                   has_answer=answer_in_context(ctx, answers) if answers else None,
                   text=f"Assembled a {ctx_tokens:,}-token context (the whole feed so far is {full_tokens:,})")
        last_query = {"question": question, "policy": policy, "hits": [h["cell"] for h in hits], "verdicts": verdicts,
                      "raw_cells": [c for c in dec.fallback_cells] if policy == "gate" else [], "ts": time.time()}

        t0 = time.perf_counter()
        out, final = [], {}
        try:
            for part in _stream_llm(user_prompt(kind, ctx, question), MAX_TOKENS[kind]):
                tok = part.get("message", {}).get("content", "")
                if tok:
                    out.append(tok)
                    yield _sse("token", text=tok)
                if part.get("done"):
                    final = part
        except Exception as e:  # noqa: BLE001
            feed.event("error", "llm", "LLM call failed", str(e)[:120])
            yield _sse("error", message=f"Ollama is not reachable ({type(e).__name__}). Start it with `ollama serve`.")
            return
        pred = "".join(out).strip()
        result = {"pred": pred, "llm_s": round(time.perf_counter() - t0, 1),
                  "prompt_tokens": final.get("prompt_eval_count"), "total_s": round(time.perf_counter() - t_all, 1),
                  "model": MODEL}
        if answers:
            result.update({"gold": answers[0], **score(kind, pred, answers)})
        feed.event("info", "answer", f"answer streamed in {result['llm_s']}s", pred[:100])
        yield _sse("done", **result, highlight=sorted(set(terms(question))))
    finally:
        feed.thinking = False


@app.get("/api/ask")
def ask(q: str, policy: str = "gate"):
    if not ask_lock.acquire(blocking=False):
        return JSONResponse({"error": "busy"}, status_code=409)

    def gen():
        try:
            yield from run_agent(q, "sketch" if policy == "sketch" else "gate")
        finally:
            ask_lock.release()
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------------- UI ----------------
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/{path:path}")
def spa(path: str):
    """Serve the React console; every non-API path falls back to index.html (client-side routing)."""
    if not DIST.exists():
        return JSONResponse({"error": "console not built: cd demo/console && npm install && npm run build"}, status_code=503)
    f = DIST / path
    return FileResponse(f if path and f.is_file() else DIST / "index.html")


def warm_up():
    """Load bge-small and the Ollama model up front so the first typed question does not pay ~30 s of load time."""
    feed.embedder._load()  # embed_queries alone would hit the cache and never load the model
    if os.environ.get("SKETCHRAG_WARM_LLM", "1") == "0":  # low-memory machines: let the first question load the model
        return
    try:
        body = json.dumps({"model": MODEL, "prompt": "", "keep_alive": "60m"}).encode()
        urllib.request.urlopen(urllib.request.Request(f"{OLLAMA}/api/generate", body,
                                                      {"Content-Type": "application/json"}), timeout=120).read()
    except Exception as e:  # noqa: BLE001 - /api/health reports the LLM as down instead
        print("ollama warm-up failed:", e, flush=True)


if __name__ == "__main__":
    threading.Thread(target=warm_up, daemon=True).start()
    print(f"SketchRAG console on http://localhost:{os.environ.get('PORT', 8600)}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8600)), log_level="warning")

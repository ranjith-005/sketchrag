"""Run systems end to end with the shared Ollama model, score them, and write one JSONL row
per (system, example) to results/runs/<run>/<system>.jsonl. Runs are resumable.

  python scripts/run_system.py --run pilot --systems dense_rag sketchrag_gate --tasks hotpotqa --n 5
  OLLAMA_HOST=http://host:11434 python scripts/run_system.py --run main --model qwen2.5:7b-instruct --num-ctx 32768 ...

Every system uses the same model, num_ctx, greedy decoding, and prompt template; only the
context differs. full_context is middle-truncated to fit num_ctx (LongBench protocol).
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sketchrag.data.loaders import INFBENCH_TASKS, LONGBENCH_TASKS, load_task  # noqa: E402
from sketchrag.embed import Embedder  # noqa: E402
from sketchrag.eval.metrics import answer_in_context, score  # noqa: E402
from sketchrag.llm import DEFAULT_MODEL, LLM, Completion  # noqa: E402
from sketchrag.prompts import MAX_TOKENS, SYSTEM, user_prompt  # noqa: E402
from sketchrag.registry import make_system  # noqa: E402
from sketchrag.systems import DocIndex  # noqa: E402

INFO_KEYS = ("hot_bytes", "stored_tokens", "n_cells", "n_chunks", "ingest_s", "ingest_ms_per_chunk", "retrieve_s",
             "fallback_cells", "gate_reasons", "fetched_cells", "llm_fetches", "searches", "rounds", "cold_bytes",
             "truncated")


def make_row(sys_name, ex, comp: Completion, info: dict, context: str | None, model: str) -> dict:
    row = {"system": sys_name, "id": ex.id, "task": ex.task, "kind": ex.kind, "pred": comp.text,
           "answers": ex.answers[:3], "stop_reason": comp.stop_reason, **score(ex.kind, comp.text, ex.answers),
           "context_tokens": comp.usage.input_tokens, "output_tokens": comp.usage.output_tokens,
           "llm_calls": comp.usage.calls, "llm_s": comp.usage.llm_s, "latency_s": comp.latency_s, "model": model}
    if context is not None:
        row["answer_recall"] = answer_in_context(context, ex.answers)
        row["ctx_words"] = len(context.split())
    for k in INFO_KEYS:
        if k in info:
            row[k] = info[k]
    row["fallback"] = bool(info.get("fallback_cells") or info.get("fetched_cells"))
    return row


def file_name(system_name: str) -> str:
    return system_name.replace("@", "__").replace(",", "_").replace("=", "-") + ".jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=LONGBENCH_TASKS + list(INFBENCH_TASKS))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--chunk-words", type=int, default=150)
    ap.add_argument("--workers", type=int, default=1, help="parallel requests (match OLLAMA_NUM_PARALLEL)")
    args = ap.parse_args()

    out_dir = ROOT / "results" / "runs" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=1))
    emb = Embedder(cache_path=ROOT / "cache" / "emb.sqlite")
    llm = LLM(args.model, num_ctx=args.num_ctx)
    # full context gets the whole window minus room for instructions and the answer. 3 chars/token is
    # conservative for prose (~4) and for UUID-heavy kv_retrieval text (~2.5-3).
    full_ctx_chars = int((args.num_ctx - 1024) * 3)

    examples = []
    for t in args.tasks:
        examples += load_task(t, args.n, args.seed)
    print(f"{len(examples)} examples over {len(args.tasks)} tasks; model={args.model} num_ctx={args.num_ctx}", flush=True)
    idx_cache: dict[str, DocIndex] = {}

    def index(ex):
        if ex.id not in idx_cache:
            idx_cache[ex.id] = DocIndex.build(ex, emb, args.chunk_words)
        return idx_cache[ex.id]

    for sys_name in args.systems:
        overrides = list(args.set)
        if sys_name == "full_context":
            overrides.append(f"x.max_chars={full_ctx_chars}")
        system = make_system(sys_name, overrides, llm)
        path = out_dir / file_name(system.name)
        done = set()
        if path.exists():
            done = {json.loads(l)["id"] for l in path.open(encoding="utf-8") if l.strip()}
        todo = [ex for ex in examples if ex.id not in done]
        print(f"\n== {system.name}: {len(todo)} to run ({len(done)} done) -> {path.name}", flush=True)
        if not todo:
            continue
        for ex in todo:  # embeddings are looked up serially (the cache is not thread-safe)
            index(ex)

        def run_one(ex):
            idx = index(ex)
            if getattr(system, "agentic", False):
                comp, info = system.run_live(ex, idx, emb)
                return make_row(system.name, ex, comp, info, info.get("context"), args.model)
            p = system.prepare(ex, idx)
            comp = llm.complete(SYSTEM, user_prompt(ex.kind, p.context, ex.question), MAX_TOKENS[ex.kind])
            row = make_row(system.name, ex, comp, p.info, p.context, args.model)
            # Ollama silently drops the start of prompts longer than num_ctx; flag any row where that happened
            row["prompt_truncated"] = comp.usage.input_tokens >= args.num_ctx - MAX_TOKENS[ex.kind] - 8
            return row

        with ThreadPoolExecutor(args.workers) as pool, path.open("a", encoding="utf-8") as f:
            for i, row in enumerate(pool.map(run_one, todo), 1):
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"  [{i}/{len(todo)}] {row['task']:20s} score={row['score']:.2f} ctx_tok={row['context_tokens']} "
                      f"{row['latency_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

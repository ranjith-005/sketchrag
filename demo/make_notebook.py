"""Write notebooks/SketchRAG_Demo.ipynb. Execute it with:

  python -m jupyter nbconvert --to notebook --execute --inplace notebooks/SketchRAG_Demo.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# SketchRAG demo: streaming sketches as retrieval memory

**Team:** Ranjith Kumar (2023BCD0030), Bhuvan S M (2023BCS0198), Nooh K (2023BCD0003), Hariprasanth (2023BCS0036)

An agent reading an unbounded stream cannot keep and search every chunk. SketchRAG keeps a **bounded hot memory**
(semantic cells, per-cell Count-Min Sketches, Misra-Gries heavy hitters, short extractive digests) and puts the raw chunks in
**cold storage**. At question time, an **error-bound gate** decides which cells' digests can be trusted and which must be
read raw.

This notebook walks through it on real test questions:

1. The two sketches on their own, and their error guarantees
2. Streaming a long document into bounded memory
3. Looking inside the memory
4. Three questions: the gate stays closed, fires on a lossy digest, fires on an exact key
5. The full results (350 questions, qwen2.5:7b-instruct)

Steps 1 to 4 need no LLM. Answers shown as *saved* are the recorded qwen2.5:7b-instruct answers from the main Kaggle run.
Set `RUN_LIVE = True` to also ask a local Ollama model.
""")

code("""
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "demo"))

import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

import demo_core as d
from sketchrag.sketches.countmin import CountMinSketch
from sketchrag.sketches.misra_gries import MisraGries

pd.set_option("display.max_colwidth", 120)
RUN_LIVE = False          # True: also ask the local Ollama model (about 40 s per answer on this CPU)
LIVE_MODEL = "llama3.2:3b"
""")

md("""
## 1. The sketches on their own

**Count-Min Sketch** (width *w*, depth *d*) never under-counts, and over-counts by at most ε·N with ε = e/w,
with probability at least 1 − e^(−d). **Misra-Gries** with *k* counters finds every term with frequency above N/(k+1).
Both are hand-written in `src/sketchrag/sketches/` and unit-tested. Here they run on a Zipf-distributed stream of 200,000 tokens.
""")

code("""
rng = np.random.default_rng(0)
stream_tokens = [f"t{x}" for x in rng.zipf(1.3, 200_000) if x < 50_000]
true = Counter(stream_tokens)

cms = CountMinSketch(width=512, depth=3)
mg = MisraGries(k=64)
for t in stream_tokens:
    cms.update(t)
    mg.update(t)

N = len(stream_tokens)
over = np.array([cms.query(t) - c for t, c in true.items()])
print(f"stream length N = {N:,}, distinct terms = {len(true):,}")
print(f"CMS memory: {cms.nbytes / 1024:.0f} KB for {len(true):,} distinct terms")
print(f"never under-counts: {bool((over >= 0).all())}")
print(f"eps*N bound = {cms.error_bound():.0f};  terms over the bound: {(over > cms.error_bound()).mean():.2%} "
      f"(guarantee: at most {cms.delta:.1%})")

top_true = [t for t, _ in true.most_common(10)]
top_mg = [t for t, _ in mg.top(10)]
print("\\ntrue top 10:        ", top_true)
print("Misra-Gries top 10: ", top_mg)
print("overlap:", len(set(top_true) & set(top_mg)), "/ 10")
""")

md("""
## 2. Stream a long document into bounded memory

A NarrativeQA story (about 28k words) arrives as 150-word chunks. Each chunk joins its nearest cell or opens a new one.
When there are more than 64 cells, the closest pair is merged (Ward linkage). The hot memory stops growing once
the cap is reached; cold storage keeps every raw chunk.
""")

code("""
ex = d.find_example("narrativeqa/a5e44b1e2b98f80aabab5f1b8296d0fe66d53b31c1f3e332")
chunks, embs = d.chunk_and_embed(ex)          # bge-small embeddings, cached from the experiments
mem, trace = d.stream(chunks, embs, every=2)
tr = pd.DataFrame(trace)

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].plot(tr.chunks, tr.cells, color="#2a6fdb")
ax[0].axhline(64, ls="--", color="grey", lw=1)
ax[0].set(title="Cells (cap = 64)", xlabel="chunks streamed", ylabel="cells")
ax[1].plot(tr.chunks, tr.hot_kb, label="hot memory (searched)", color="#2a6fdb")
ax[1].plot(tr.chunks, tr.cold_kb, label="cold storage (raw chunks)", color="#d9822b")
ax[1].set(title="Memory size", xlabel="chunks streamed", ylabel="KB")
ax[1].legend()
plt.tight_layout()
plt.show()
s = mem.stats()
print(f"{s['n_chunks']} chunks -> {s['n_cells']} cells after {s['n_merges']} merges; "
      f"ingest {tr.ms.iloc[-1] / s['n_chunks']:.1f} ms/chunk")
""")

md("""
After the cap, hot memory only creeps up until every cell's digest (120 words) and exemplar set (8 embeddings)
is full; that ceiling is fixed by the configuration, not by the stream length. On one short document it is
still larger than the raw text, because sketch tables and exemplar embeddings dominate. The bound pays off on
long streams. In our 8.5M-word stress test, hot memory
stayed at 1 to 4 MB while a dense index needed 146 MB:
""")

code("""
display(Image(filename=str(d.ROOT / "report" / "figures" / "stress_memory.png"), width=720))
""")

md("""
## 3. Inside the memory

Each cell keeps its time span, size, radius (how spread out its chunks are) and a digest of at most 120 words.
The global Misra-Gries table gives the stream's topics.
""")

code("""
print("Stream topics (Misra-Gries heavy hitters):", ", ".join(t for t, _ in mem.mg.top(15)))
cells = pd.DataFrame(d.cells_table(mem))
cells["digest"] = cells["digest"].str.slice(0, 160) + "..."
cells.head(8)
""")

md("""
## 4. The gate, question by question

For every retrieved cell, the gate checks each **rare** query term (idf ≥ 2.5) against the cell's Count-Min Sketch:

* **lossy**: the sketch says the term is in the cell (above the noise floor) but the digest dropped it
* **unresolved**: no retrieved cell holds the rare terms above the guaranteed ε·N bound, so the top two cells are read raw
* **loose**: the cell is too spread out for its digest to be faithful

Escalated cells have their digest replaced by their two most relevant raw chunks from cold storage.
""")

code('''
def show(ex_id):
    case = next(c for c in d.SHOWCASE if c["id"] == ex_id)
    ex = d.find_example(ex_id)
    chunks, embs = d.chunk_and_embed(ex)
    mem, _ = d.stream(chunks, embs, every=len(chunks))
    q = d.embed_question(ex.question)
    g = d.explain_gate(mem, ex.question, q)
    ctx = d.contexts(mem, ex.question, q)

    display(Markdown(f"### {case['title']}\\n{case['story']}\\n\\n"
                     f"**Question:** {ex.question.strip()[:300] or '(summarise the report)'}  \\n"
                     f"**Gold answer:** {ex.answers[0][:200]}  \\n"
                     f"**Document:** {len(ex.context.split()):,} words, {len(chunks)} chunks, {len(mem.cells)} cells"))
    print("rare query terms:", g["rare"] or "none (the gate has nothing to check)")
    display(pd.DataFrame(g["hit_rows"]).set_index("rank"))
    if g["term_rows"]:
        t = pd.DataFrame(g["term_rows"])
        display(t.pivot_table(index="term", columns="cell", values="CMS estimate", sort=False)
                 .style.set_caption("CMS estimate of each rare term in each retrieved cell"))
    for cid, why in g["decision"].reasons.items():
        print(f"  cell {cid}: {why} -> {d.REASON_TEXT[why]}")

    rows = []
    saved = d.saved_answers(ex.id)
    for policy, system in (("sketch", "sketchrag_sketch"), ("gate", "sketchrag_gate"), ("raw", "sketchrag_raw")):
        r = saved.get(system, {})
        rows.append({"system": d.SYSTEM_LABELS[system], "context words": ctx[policy]["words"],
                     "gold answer in context": bool(d.answer_in_context(ctx[policy]["context"], ex.answers)),
                     "saved answer (qwen2.5:7b)": r.get("pred", "")[:60], "score": r.get("score")})
    for system in ("full_context", "bm25_rag", "dense_rag"):
        r = saved.get(system, {})
        rows.append({"system": d.SYSTEM_LABELS[system], "context words": None, "gold answer in context": None,
                     "saved answer (qwen2.5:7b)": r.get("pred", "")[:60], "score": r.get("score")})
    display(pd.DataFrame(rows).set_index("system"))
    if RUN_LIVE:
        live = d.live_answer(ex, ctx["gate"]["context"], LIVE_MODEL)
        print(f"live {LIVE_MODEL} with the gated context: {live}")
    return ctx
''')

md("### 4a. Summary is enough")
code('ctx = show("hotpotqa/9336554e511e88766228ffa1acd1c6f82c9a41046514c253")')

md("### 4b. Lossy digest")
code('ctx = show("hotpotqa/1ceb1c91662d241b3421853112a91b5868d1b81fe3b88ce4")')

md("### 4c. Exact key lookup")
code('ctx = show("kv_retrieval/130")')
md("The gated context the LLM actually receives (digests for trusted cells, raw evidence for escalated ones):")
code('print(ctx["gate"]["context"][:1500])')

md("""
### 4d. More cases

Same function, other behaviours: a long story where digest-only gives up, a needle in a haystack where full context
loses the needle to truncation, and a summary request where the gate correctly never fires.
""")
code('ctx = show("narrativeqa/a5e44b1e2b98f80aabab5f1b8296d0fe66d53b31c1f3e332")')
code('ctx = show("passkey/62")')
code('ctx = show("gov_report/81e845f35acd55c1f6b4edda428a0a52fd22bade5be9cf13")')

md("""
## 5. Results over all 350 test questions

12 LongBench / ∞-Bench tasks, the same qwen2.5:7b-instruct model and prompt for every system.
""")

code("""
run = d.load_run("main")
rows = []
for name, recs in run.items():
    df = pd.DataFrame(recs)
    per_task = df.groupby("task")["score"].mean() * 100
    rows.append({"system": d.SYSTEM_LABELS.get(name, name), "avg score": per_task.mean(),
                 "LLM context tokens": df.context_tokens.mean(), "LLM s / question": df.llm_s.mean(),
                 "fallback rate": df.get("fallback", pd.Series([False] * len(df))).fillna(False).astype(float).mean()})
pd.DataFrame(rows).set_index("system").sort_values("avg score", ascending=False).round(1)
""")

code("""
display(Image(filename=str(d.ROOT / "report" / "figures" / "main_pareto.png"), width=720))
""")

md("### Is the gate calibrated?")
code("""
c = d.gate_calibration(run)
print(f"gate fired on {c['fired']} of {c['n']} questions ({c['fired'] / c['n']:.0%})")
print(f"  on those:  digest only {c['fired_sketch']:.1f}  ->  with raw evidence {c['fired_gate']:.1f}")
print(f"gate trusted the digest on {c['trusted']} questions")
print(f"  on those:  gate {c['trusted_gate']:.1f}  vs  always raw {c['trusted_raw']:.1f}  (little left to gain)")
""")

md("""
## Takeaways

* The digest alone (29.9) loses exact facts. The Count-Min error bound tells us exactly when, and reading raw evidence
  only then gives 41.7, beating full context, dense RAG, SARA-lite and AMBER-lite with the fewest LLM tokens (1,213 / question).
* The gate's signal comes from the sketch guarantee, not from asking the LLM: the LLM-driven agent called a tool on
  only 30 of 350 questions and stayed at the digest-only level (30.1).
* BM25 RAG is still the strongest system overall (46.0), mostly on exact-match tasks.
* Limitation: on an 8.5M-word stream the memory stays at 1 to 4 MB, but recall drops to 0.29 against 0.44 for dense RAG.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "SketchRAG_Demo.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)

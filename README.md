# SketchRAG

**Approximate streaming summaries as retrieval context for agentic LLMs.**

Mini project, IIIT Kottayam, **Group 6**. Team: R Ranjith Kumar (2023BCD0030, team lead), Bhuvan S M (2023BCS0198),
Hariprashanth M (2023BCS0036), Nooh K (2023BCD0003).

A document (or an unbounded stream) arrives as chunks. Instead of keeping every chunk in an index,
SketchRAG keeps a **bounded hot memory** and updates it online:

- **semantic cells**: streaming clustering with Ward merge-and-reduce, at most `max_cells`
- a per-cell **Count-Min Sketch** over content terms
- a global **Misra-Gries** heavy-hitter table
- **extractive digests** of at most `digest_words` per cell

Raw chunks go to append-only **cold storage**. At query time, cells are ranked by dense similarity
plus CMS lexical evidence plus temporal contiguity. An **error-bound gate** decides which cells
can't be trusted as summaries and reads raw evidence for those cells from cold storage.

```
stream ──► chunker ──► embed ──► SketchMemory (hot, bounded) ─┐
                         │        cells + CMS + MG + digests   │   query ──► rank cells ──► error-bound gate
                         └──────► ColdStore (append-only raw) ◄┘                          │
                                                                    fetch_raw for untrusted cells
                                                                                           ▼
                                                                      context ──► LLM (Ollama) ──► answer
```

## Results

Evaluation uses 12 LongBench and InfiniteBench tasks, with the same fixed-seed questions (test split, seed 0)
for every system. Answers come from `qwen2.5:7b-instruct` via Ollama. Full tables are in
[`results/tables/main.md`](results/tables/main.md) and the paper.

| System | Avg score (×100) | LLM context tokens / q | LLM s / q |
|---|---|---|---|
| bm25_rag | **46.0** | 1327 | 8.1 |
| sketchrag_raw | 42.5 | 1549 | 7.6 |
| sketchrag_gate | 41.7 | **1213** | **5.8** |
| sara_lite | 38.2 | 2435 | 12.9 |
| full_context | 38.1 | 14232 | 61.5 |
| dense_rag | 35.8 | 1361 | 6.9 |
| amber_lite | 32.0 | 2223 | 31.8 |

BM25 has the best average. `sketchrag_gate` scores 4.3 points lower but uses the least LLM context
and time of all systems, and beats dense RAG and full context. It loses the most against full context
on narrative and multi-hop QA, and matches BM25 on passkey and KV retrieval.

## Repository layout

```
src/sketchrag/sketches/        countmin.py, misra_gries.py, reservoir.py   (hand-written, unit-tested)
src/sketchrag/memory.py        SketchMemory (hot) + ColdStore
src/sketchrag/retrieve.py      sketch-aware search, fallback gate, fetch_raw, context assembly
src/sketchrag/agent.py         tool-using agent (fetch_raw / search_memory) via Ollama
src/sketchrag/systems.py       baselines: full context, BM25, dense (FAISS), SARA-lite; SketchRAG policies
src/sketchrag/baselines/       amber_lite.py (AMBER-inspired adaptive memory RAG)
src/sketchrag/llm.py           Ollama /api/chat client (greedy, seeded, token + time accounting)
src/sketchrag/data/loaders.py  LongBench / InfiniteBench loaders with fixed-seed subsampling
scripts/                       offline_eval, run_system, stress_stream, make_tables, make_figs, ...
tests/                         sketch guarantees + memory invariants (pytest)
kaggle/                        GPU kernels: embed_cache, emllm_repro (official EM-LLM), ollama_runs (bulk LLM runs)
results/                       per-run answers (runs/*/*.jsonl), tables, stress tests, dev-split tuning log
demo/                          live agent server + React console, Streamlit dashboard, notebook builder
notebooks/                     SketchRAG_Demo.ipynb (executed walkthrough)
report/paper/                  IEEE LaTeX paper (tables, figures and numbers generated from results/)
report/slides/                 presentation deck builder (pptxgenjs)
data_raw/examples/             frozen test-split questions (seed 0)
```

## Setup

Requires Python 3.10+ and, for live answers, [Ollama](https://ollama.com).

```
git clone https://github.com/ranjith-005/sketchrag.git
cd sketchrag
pip install -e .            # numpy, torch, transformers, faiss-cpu, rank_bm25, matplotlib
pip install fastapi uvicorn pytest streamlit jupyter
ollama pull llama3.2:3b     # small model for the live demo (CPU-friendly)
ollama pull qwen2.5:7b-instruct   # model used for the reported results
```

`OLLAMA_HOST` selects the Ollama server and `SKETCHRAG_MODEL` the default model.

### Data

The frozen test-split questions are already in `data_raw/examples/`. The raw benchmark files are too big
for GitHub. To draw new subsamples (for example the dev split, seed 1000), download them into `data_raw/`:

| File in `data_raw/` | Source |
|---|---|
| `longbench_data.zip` | `data.zip` from [THUDM/LongBench](https://huggingface.co/datasets/THUDM/LongBench) |
| `ib_passkey.jsonl` | `passkey.jsonl` from [xinrongzhang2022/InfiniteBench](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench) |
| `ib_kv_retrieval.jsonl` | `kv_retrieval.jsonl` from InfiniteBench |
| `ib_longbook_qa_eng.jsonl` | `longbook_qa_eng.jsonl` from InfiniteBench |

`SKETCHRAG_RAW` can point to a different data directory.

## Running

```
python -m pytest -q                                   # sketch guarantees + memory invariants
python scripts/offline_eval.py --n 10                 # LLM-free answer recall on the dev split (seed 1000)
python scripts/run_system.py --run pilot --tasks hotpotqa --n 5 --systems dense_rag sketchrag_gate
python scripts/make_tables.py --run pilot
```

Tuning uses only the dev split (seed 1000). The runner is resumable: it continues from
`results/runs/<run>/*.jsonl`. Bulk runs for the paper were done on Kaggle GPUs (`kaggle/ollama_runs`),
and the EM-LLM comparison reproduces the official repo at reduced scale (`kaggle/emllm_repro`).

## Demo

```
cd demo/console && npm install && npm run build && cd ../..   # once: build the React console
python demo/agent/server.py                                     # SketchRAG console at http://localhost:8600
```

The console has six pages: overview, live stream, memory explorer, query playground, evaluation lab, and
configuration & logs. The backend streams a 400k-word feed into bounded memory and answers questions live
with local Ollama (`llama3.2:3b`), showing each step: search, error-bound gate, raw fetch, context and
the streamed answer. Live panels read the running backend. Evaluation panels read `results/` and are
labelled as report results. `python demo/agent/pick_presets.py --llm` rebuilds the suggested questions.

Other ways to explore it:

```
python -m streamlit run demo/dashboard/Home.py              # full dashboard (6 pages) at http://localhost:8501
python -m streamlit run demo/app.py                         # simpler one-page demo
python -m jupyter notebook notebooks/SketchRAG_Demo.ipynb   # walkthrough (already executed, outputs saved)
python demo/make_notebook.py                                # regenerate the notebook from source
```

The dashboard and notebook use `demo/demo_core.py`, which calls the package code and the saved
qwen2.5:7b answers in `results/runs/main`. They need no LLM, except for the optional live answer.

## Complete pipeline

1. **Data**: fixed-seed subsamples of 9 LongBench and 3 InfiniteBench tasks (`src/sketchrag/data/loaders.py`);
   test split seed 0 (frozen in `data_raw/examples/`), dev split seed 1000 for tuning.
2. **Embeddings**: `bge-small-en-v1.5` chunk embeddings, precomputed on a Kaggle GPU (`kaggle/embed_cache`).
3. **Ingest**: each document is streamed chunk by chunk into `SketchMemory` (cells, Count-Min, Misra-Gries,
   digests), with raw chunks written to `ColdStore`.
4. **Retrieve**: cell ranking, error-bound gate and raw fetch build the LLM context (`retrieve.py`).
5. **Answer**: Ollama generates the answer, `qwen2.5:7b-instruct` for the reported runs
   (`kaggle/ollama_runs`, `scripts/run_system.py`); per-question records go to `results/runs/<run>/*.jsonl`.
6. **Score and report**: `scripts/make_tables.py` and `scripts/make_figs.py` compute the tables and figures;
   `report/paper/make_assets.py` writes them into the LaTeX paper.
7. **Baseline comparison**: EM-LLM reproduced from its official repo on Kaggle (`kaggle/emllm_repro`) and
   matched against SketchRAG on the same questions (`results/runs/emllm_match`).
8. **Demo**: `demo/agent/server.py` runs the same package code live on a 400k-word stream, shown in the
   React console.

## Tools and versions

| Tool | Version | Used for |
|---|---|---|
| Python | 3.12.10 | everything (EM-LLM reproduction: Python 3.11 venv via `uv` on Kaggle) |
| PyTorch | 2.14.0 (CPU locally, CUDA on Kaggle) | embeddings, EM-LLM |
| Transformers | 5.19.0 | embedding model |
| faiss-cpu | 1.15.1 | dense retrieval baseline |
| rank_bm25 | 0.2.2 | BM25 baseline |
| NumPy | 2.5.1 | sketches, clustering |
| Matplotlib | 3.11.0 | figures |
| pytest | 9.1.1 | unit tests |
| FastAPI / Uvicorn | 0.139.0 / 0.51.0 | demo backend |
| Streamlit | 1.66.0 | demo dashboard |
| Ollama | 0.40.5 (local) | LLM serving |
| LLMs | `qwen2.5:7b-instruct` (results), `llama3.2:3b` (live demo), `Phi-3.5-mini-instruct` (EM-LLM) | answering |
| Embedding model | `BAAI/bge-small-en-v1.5` | chunk and query embeddings |
| Node.js | 24.19.0 | demo console and slide build |
| React / Vite / Tailwind CSS / Recharts | 18.3.1 / 5.4.10 / 3.4.14 / 2.13.3 | demo console |
| pptxgenjs | 3.12.0 | slide deck |
| MiKTeX (pdfTeX) | 25.12 (4.23), `IEEEtran` class | report |
| Kaggle | 2 × Tesla T4 GPU notebooks | bulk embedding, LLM runs, EM-LLM |

## Report

- Paper: [`report/SketchRAG_Final_Report.pdf`](report/SketchRAG_Final_Report.pdf), LaTeX source in
  [`report/paper/`](report/paper/) (`python report/paper/make_assets.py` regenerates every table, figure and number)
- Slides: [`report/SketchRAG_Presentation.pptx`](report/SketchRAG_Presentation.pptx)

"""Fixed prose for the SketchRAG report (HTML fragments). Every number here comes from
results/ through build_report.py; this file only holds text that doesn't depend on results."""

TITLE = "SketchRAG: Approximate Streaming Summaries as Retrieval Context for Agentic AI"
AUTHORS = [("Ranjith Kumar", "2023BCD0030"), ("Bhuvan S M", "2023BCS0198"), ("Nooh K", "2023BCD0003"),
           ("Hariprasanth", "2023BCS0036")]
INSTITUTE = "Department of Computer Science and Engineering, Indian Institute of Information Technology Kottayam"

INTRO = """
<p>LLM agents increasingly read information that arrives continuously: logs, transcripts, documents
appended over time. Conventional retrieval-augmented generation (RAG) keeps every chunk and a vector for each
one, so its searchable index grows linearly with the stream. Long-context and episodic-memory methods
(EM-LLM) avoid re-reading everything but still keep per-event key/value state for the whole history.</p>
<p><b>SketchRAG</b> asks whether a <i>bounded</i>, continuously maintained approximate summary of the stream
can serve as the retrieval layer itself. It combines classic streaming-data synopses (Count-Min Sketch,
Misra-Gries heavy hitters, reservoir sampling, streaming clustering) with an agentic fallback, so the agent
answers from the summary when that is safe and reads raw evidence only when the sketch says it cannot vouch
for the answer.</p>
<p><b>Central research question.</b> Can approximate streaming summaries provide a memory-efficient retrieval
context for agentic AI while keeping answer quality comparable to full-context and RAG approaches?</p>
"""

NOVELTY = """
<ol>
<li><b>Summaries maintained before any query.</b> Compression is applied to the <i>stream</i> as it arrives, not to
retrieved passages after a query (SARA) and not at query time (AMBER). The hot memory has a fixed size
whatever the stream length.</li>
<li><b>An error-bound-gated fallback.</b> The Count-Min Sketch's guarantee
(<i>f(x) &le; est(x) &le; f(x) + &epsilon;N</i> with probability 1&minus;&delta;) and the digest contents decide, per
retrieved cell, whether the summary can be trusted. Cells that fail are escalated to raw evidence in cold storage.
This turns a classic sketch guarantee into a calibrated "is the summary enough?" signal for the agent (RQ4).</li>
</ol>
"""

METHOD = """
<h3>Architecture</h3>
<p>The document or stream is split into chunks of about 150 word-units, where long machine tokens such as UUIDs
are charged per 6 characters. Each chunk is embedded once with <code>bge-small-en-v1.5</code> and passed to
<code>SketchMemory.ingest</code>, which updates two stores:</p>
<ul>
<li><b>Hot memory</b> (searched at query time; size fixed by the configuration):
  <ul>
  <li><b>Semantic cells.</b> These are clustering features: embedding sum and count, from which the centroid and
  radius follow (for unit vectors E||x&minus;&mu;||&sup2; = 1 &minus; ||&mu;||&sup2;), plus a time span. A chunk joins
  the most similar cell if cosine &ge; 0.90 (with a recency bonus for the current cell, in the spirit of EM-LLM's
  event continuity). Otherwise it opens a new cell. When the cell count exceeds <code>max_cells</code>, the pair
  with the smallest <b>Ward</b> merge cost n<sub>a</sub>n<sub>b</sub>/(n<sub>a</sub>+n<sub>b</sub>)&middot;||&mu;<sub>a</sub>&minus;&mu;<sub>b</sub>||&sup2;
  is merged (merge-and-reduce).</li>
  <li><b>Per-cell Count-Min Sketch</b> over content terms (width 512, depth 3, conservative update). CMS is linear,
  so merging two cells just adds their tables.</li>
  <li><b>Extractive digest</b> per cell: at most 120 words of the most salient sentences. Salience is the mean
  streaming IDF of a sentence's terms, scaled by how typical its chunk is of the cell. No LLM is called at
  ingest time.</li>
  <li><b>Exemplars:</b> up to 8 diverse member embeddings per cell (fp16), kept by dropping the most redundant
  member. They play the role of EM-LLM's representative tokens when cells are ranked.</li>
  <li>A global <b>Count-Min Sketch of document frequencies</b> (streaming IDF) and a global <b>Misra-Gries</b>
  table of the k = 256 heaviest terms (stream topics).</li>
  </ul></li>
<li><b>Cold storage</b> (append-only, never searched; read only on fallback): raw chunk text, its embedding, and
the cell&rarr;chunk pointer lists. Cold storage grows with the stream, and the report always states which of
the two memories a number refers to.</li>
</ul>

<h3>Query time</h3>
<ol>
<li><b>Cell ranking:</b> the score is the maximum cosine similarity of the query over each cell's centroid and
exemplars. Top-k = 4 cells are taken, plus one temporal neighbour of the best cell (EM-LLM's contiguity buffer).
A Count-Min lexical term is available, but dev-split tuning set its weight to 0 for ranking (see Ablations).</li>
<li><b>Gate.</b> For the rare query terms (streaming IDF &ge; 2.5), a retrieved cell is escalated to raw evidence
if any of these hold:
  <ul>
  <li><i>lossy:</i> the CMS shows the term in the cell above its noise floor N/w, but the digest lacks it, or no
  single digest sentence contains the rare terms together;</li>
  <li><i>loose:</i> the cell radius is above 0.75;</li>
  <li><i>unresolved:</i> no retrieved cell holds any rare term above the guaranteed bound &epsilon;N. In that case
  the top two cells are read raw.</li>
  </ul>
  Questions with no rare terms (global or summary questions) are answered from the sketch.</li>
<li><b>Raw fetch:</b> inside an escalated cell, the cell's cold chunks are reranked by cosine similarity plus
IDF-weighted term overlap. The best two replace that cell's digest in the context.</li>
<li>The LLM answers from: heavy-hitter topics + digests of trusted cells + raw evidence of escalated cells.</li>
</ol>

<h3>Policies compared (RQ4)</h3>
<ul>
<li><b>sketch:</b> digests only, never fall back.</li>
<li><b>raw:</b> always fall back on every retrieved cell.</li>
<li><b>gate:</b> the deterministic gate above.</li>
<li><b>agent:</b> the LLM sees the sketch and may call <code>fetch_raw(cell_id)</code> or
<code>search_memory(query)</code> tools (at most 3 rounds).</li>
<li><b>hybrid:</b> the gate's context, plus the same tools.</li>
</ul>
"""

SETUP = """
<h3>Data</h3>
<p>The test split uses a fixed-seed sample (seed 0) of 30 questions per task:
<ul>
<li>9 English LongBench tasks: NarrativeQA, Qasper, MultiFieldQA-en, HotpotQA, 2WikiMQA, MuSiQue, GovReport,
QMSum, PassageRetrieval-en;</li>
<li>&infin;-Bench Retrieve.PassKey and Retrieve.KV;</li>
<li>20 questions from &infin;-Bench En.QA (books of about 190K tokens).</li>
</ul>
All tuning used a disjoint <b>dev split</b> (seed 1000, 10 questions per task). Test questions were never used
to choose a setting. Each document is fed to every system as a chunk stream.</p>
<p>The <b>long-stream stress test</b> concatenates every English LongBench QA and summary document (1,115
documents, 8.5M words, about 11M tokens) into one stream in a fixed shuffled order. It then asks 100 probe
questions whose source documents are spread evenly over the stream.</p>

<h3>LLM and compared systems</h3>
<p>All systems share one LLM served by <b>Ollama</b> with greedy decoding, the same prompt template, and the
same answer-format instructions. Only the context differs between systems. The main runs use
<code>qwen2.5:7b-instruct</code> (num_ctx 32,768) on a Kaggle T4. Full context is middle-truncated to fit the
window, following LongBench's protocol.</p>
<ul>
<li><b>Full context:</b> the whole document, truncated to the window.</li>
<li><b>BM25 RAG:</b> <code>rank_bm25</code> top-5 chunks. This replaces Elasticsearch BM25 with the same scoring
algorithm.</li>
<li><b>Dense RAG:</b> FAISS inner product over bge embeddings, top-5 chunks.</li>
<li><b>SARA-lite:</b> dense top-20 chunks, compressed after retrieval to a 600-word budget by sentence
selection. SARA's learned compression vectors are not reproduced.</li>
<li><b>AMBER-lite:</b> up to 3 rounds of retrieval in which the LLM updates a memory note (AMU), decides whether
to retrieve again (AIC) and filters sentences (MCF), using JSON-schema outputs. AMBER's multi-agent prompts and
trained filters are not reproduced.</li>
<li><b>EM-LLM:</b> the <i>official</i> repository and evaluation scripts with its Phi-3.5-mini configuration, run
on Kaggle 2&times;T4 on a 20/10-question subset of the same test questions. Hardware-forced changes are listed in
the reproduction section. To compare like with like, SketchRAG and the RAG baselines are also run on the same
subset with the same base model (<code>phi3.5</code> through Ollama).</li>
</ul>

<h3>Metrics</h3>
<ul>
<li><b>Answer quality:</b> LongBench's official metrics. QA uses token F1 (exact match is also reported),
summaries use ROUGE-L, passage retrieval uses its retrieval score, and PassKey/KV use exact containment.</li>
<li><b>Retrieval quality:</b> <i>answer recall</i>, whether a gold answer string appears in the context sent to
the LLM. LongBench lacks gold evidence spans, so this is a proxy, labelled as such.</li>
<li><b>Efficiency:</b> LLM context tokens and LLM calls per question; hot-memory bytes; ingest time per chunk;
retrieval latency; fallback rate.</li>
</ul>
"""

LIMITATIONS = """
<ul>
<li><b>The hot memory is bounded; cold storage is not.</b> The fallback needs raw text on disk. The
<i>sketch</i> policy is the only truly bounded configuration end to end, and it is reported separately.</li>
<li><b>Re-implementations, not reproductions.</b> AMBER and SARA are lightweight re-implementations on a shared
LLM. Only EM-LLM is run from its official code, and only at reduced scale on T4 GPUs with the changes listed.
Its &infin;-Bench runs (KV, PassKey; ~128K-token inputs) and NarrativeQA (killed after its first question, twice)
exceeded the 29 GB host-RAM limit even with disk offload, so EM-LLM is reported on 6 of the 9 planned tasks.</li>
<li><b>Answer recall is a proxy.</b> It is a string-containment proxy for retrieval quality, not Recall@K
against annotated evidence.</li>
<li><b>Single open model.</b> One 7B model (plus Phi-3.5 for the matched comparison) and 30 questions per task.
Small differences between systems (&lt;3 points) are within noise.</li>
</ul>
"""

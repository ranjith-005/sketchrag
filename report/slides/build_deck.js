// SketchRAG project deck. Every number comes from deck_data.json (python report/slides/extract_data.py).
//   node report/slides/build_deck.js  ->  report/SketchRAG_Presentation.pptx
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");
const sharp = require("sharp");
const lu = require("react-icons/lu");
const { applyTheme } = require(path.join(process.env.PPTX_SKILL_DIR, "scripts", "apply_theme.js"));

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const OUT = path.join(__dirname, "..", "SketchRAG_Presentation.pptx");

// ---------------- theme (dark console palette from the dashboard design spec) ----------------
const THEME = {
  name: "SketchRAG Console",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0B1120", lt1: "FFFFFF", dk2: "151F30", lt2: "8A9AB2",
    accent1: "2DD4BF", accent2: "60A5FA", accent3: "A78BFA", accent4: "FBBF24", accent5: "F27FBA", accent6: "233248",
    hlink: "60A5FA", folHlink: "A78BFA",
  },
};
const H = THEME.colors; // hex, only for options that need hex (chart colours, grid lines)

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "SketchRAG";
pres.author = "Ranjith Kumar, Bhuvan S M, Nooh K, Hariprashanth M";
const C = pres.SchemeColor;
const BG = C.text1, CARD = C.text2, INK = C.background1, MUTED = C.background2;
const TEAL = C.accent1, BLUE = C.accent2, VIOLET = C.accent3, AMBER = C.accent4, PINK = C.accent5, LINE = C.accent6;

// ---------------- layouts ----------------
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: BG },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.75, y: 2.35, w: 7.4, h: 1.3, fontSize: 60, bold: true, color: INK, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.75, y: 3.75, w: 7.2, h: 1.1, fontSize: 22, color: MUTED, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: BG },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "eyebrow", type: "body", x: 0.6, y: 0.38, w: 9, h: 0.3, fontSize: 12, bold: true, color: TEAL, charSpacing: 2, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.7, w: 12.1, h: 0.75, fontSize: 34, bold: true, color: INK, valign: "top", align: "left", margin: 0 }, text: "" } },
    { text: { text: "SketchRAG  ·  IIIT Kottayam", options: { x: 0.6, y: 7.02, w: 6, h: 0.3, fontSize: 10, color: MUTED, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.02, w: 0.55, h: 0.3, fontSize: 10, color: MUTED, align: "right" },
});

// ---------------- helpers ----------------
async function icon(name, hex) {
  const svg = renderToStaticMarkup(React.createElement(lu[name], { size: 256, color: `#${hex}` }));
  const buf = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
const ICONS = {};
async function loadIcons(spec) { for (const [k, n, hex] of spec) ICONS[k] = await icon(n, hex); }

function card(slide, x, y, w, h, name, fill = CARD) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: LINE, width: 0.75 }, objectName: name });
}
function iconCircle(slide, x, y, d, key, fill, name) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill, transparency: 82 }, line: { type: "none" }, objectName: `${name} circle` });
  const p = d * 0.25;
  slide.addImage({ data: ICONS[key], x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, objectName: `${name} icon` });
}
function text(slide, t, opts) { slide.addText(t, { isTextBox: true, margin: 0, color: INK, fontSize: 15, ...opts }); }
function content(section, eyebrow, title) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(eyebrow.toUpperCase(), { placeholder: "eyebrow" });
  s.addText(title, { placeholder: "title" });
  return s;
}
const chartText = { catAxisLabelColor: H.lt2, valAxisLabelColor: H.lt2, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  dataLabelFontSize: 12, dataLabelColor: H.lt1, valGridLine: { color: H.accent6, size: 0.75 }, catGridLine: { style: "none" },
  catAxisLineShow: false, valAxisLineShow: false, legendColor: H.lt2, legendFontSize: 12 };
const sys = (k) => D.systems.find((s) => s.key === k);
const fmt = (n) => n.toLocaleString("en-US");
const f1 = (x) => Number(x).toFixed(1);

// ---------------- slides ----------------
async function build() {
  await loadIcons([
    ["waves", "LuWaves", H.accent1], ["boxes", "LuBoxes", H.accent1], ["archive", "LuArchive", H.accent4], ["gate", "LuShieldCheck", H.accent3],
    ["coins", "LuCoins", H.accent4], ["clock", "LuClock", H.accent2], ["disk", "LuHardDrive", H.accent3], ["sigma", "LuSigma", H.accent3],
    ["file", "LuFileText", H.accent4], ["radar", "LuRadar", H.accent2], ["layers", "LuLayers", H.accent1], ["merge", "LuMerge", H.accent4],
    ["search", "LuSearch", H.accent2], ["brain", "LuBrain", H.accent5], ["alert", "LuAlertTriangle", H.accent4], ["check", "LuCheckCircle", H.accent1],
    ["list", "LuListChecks", H.accent1], ["cpu", "LuCpu", H.accent2], ["target", "LuTarget", H.accent3], ["scale", "LuScale", H.accent2],
    ["monitor", "LuMonitor", H.accent1], ["down", "LuTrendingDown", H.accent4], ["git", "LuGitBranch", H.accent3], ["sparkles", "LuSparkles", H.accent1],
  ]);
  const gate = sys("sketchrag_gate"), full = sys("full_context"), bm25 = sys("bm25_rag"), raw = sys("sketchrag_raw"),
    sketch = sys("sketchrag_sketch"), dense = sys("dense_rag"), sara = sys("sara_lite"), amber = sys("amber_lite"), agent = sys("sketchrag_agent");
  const cal = D.calibration, pt = D.per_task;
  const c64 = D.stress.curve_64, c256 = D.stress.curve_256;
  const MB = (b) => b / 1048576;

  // 1 ---- title
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Introduction" });
  s.addText("SketchRAG", { placeholder: "title" });
  s.addText("Approximate streaming summaries as retrieval memory for agentic LLMs", { placeholder: "body" });
  text(s, "Ranjith Kumar (2023BCD0030)  ·  Bhuvan S M (2023BCS0198)  ·  Nooh K (2023BCD0003)  ·  Hariprashanth M (2023BCS0036)",
    { x: 0.75, y: 5.35, w: 8, h: 0.35, fontSize: 14, color: INK });
  text(s, "Indian Institute of Information Technology Kottayam", { x: 0.75, y: 5.75, w: 8, h: 0.3, fontSize: 14, color: MUTED });
  // motif: a bounded grid of memory cells, a few lit by a query
  const cols = 8, size = 0.42, gap = 0.12, gx = 8.85, gy = 1.45;
  const lit = { 10: TEAL, 19: AMBER, 27: TEAL, 44: AMBER, 52: TEAL };
  const fills = [TEAL, BLUE, VIOLET, TEAL, BLUE, TEAL, VIOLET, AMBER];
  for (let i = 0; i < 64; i++) {
    const r = Math.floor(i / cols), c = i % cols;
    const f = lit[i] || fills[(i * 7 + r) % fills.length];
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: gx + c * (size + gap), y: gy + r * (size + gap), w: size, h: size, rectRadius: 0.07,
      fill: { color: f, transparency: lit[i] ? 0 : 72 }, line: { type: "none" }, objectName: `cell tile ${i}` });
  }
  text(s, "64 memory cells: the agent's whole searchable memory", { x: 8.85, y: 5.85, w: 4.3, h: 0.3, fontSize: 11, color: MUTED });
  s.addNotes("SketchRAG is our streaming-data project. The question: an agent reads a stream that never ends. Can it keep a small, fixed-size summary of everything it has read, use that summary to answer questions, and know when the summary is not good enough and it must go back to the original text? The grid on the right is the idea in one picture: a fixed number of memory cells, a few of them picked by a question.");

  // 2 ---- problem
  s = content("Introduction", "The problem", "Agents reading a stream cannot keep re-reading all of it");
  const probs = [
    ["coins", fmt(full.tokens), "LLM tokens per question", `to send the whole document (full context). SketchRAG sends ${fmt(gate.tokens)}.`, AMBER],
    ["clock", `${full.llm_s} s`, "LLM time per question", `and still only ${pt.full_context.passkey} on PassKey: truncating the middle loses the needle.`, BLUE],
    ["disk", `${Math.round(MB(c256[c256.length - 1].dense_bytes))} MB`, "dense index at 8.5M words", "a vector store grows with every chunk it ever sees.", VIOLET],
  ];
  probs.forEach(([ic, big, label, desc, col], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.95, 3.8, 3.9, `problem card ${i + 1}`);
    iconCircle(s, x + 0.35, 2.3, 0.75, ic, col, `problem ${i + 1}`);
    text(s, big, { x: x + 0.35, y: 3.25, w: 3.2, h: 0.9, fontSize: 48, bold: true, color: col });
    text(s, label, { x: x + 0.35, y: 4.2, w: 3.2, h: 0.4, fontSize: 18, bold: true });
    text(s, desc, { x: x + 0.35, y: 4.65, w: 3.15, h: 0.95, fontSize: 14, color: MUTED, valign: "top" });
  });
  text(s, "Numbers from our own runs: 350 test questions, qwen2.5:7b-instruct; long stream = 8.5M words of LongBench documents.",
    { x: 0.6, y: 6.2, w: 12, h: 0.35, fontSize: 12, color: MUTED });
  s.addNotes(`Three costs of not summarising. Full context sends ${fmt(full.tokens)} tokens per question and takes ${full.llm_s} seconds of LLM time, and it still scores only ${pt.full_context.passkey} on PassKey because the document has to be cut in the middle to fit the window. A dense vector index does not have that problem, but its size grows with every chunk: ${Math.round(MB(c256[c256.length - 1].dense_bytes))} MB after 8.5 million words in our stress test.`);

  // 3 ---- idea
  s = content("Introduction", "Our idea", "Keep a fixed-size summary, and know when not to trust it");
  const ideas = [
    ["boxes", "Bounded hot memory", "Streaming sketches summarise everything read so far in a fixed budget of cells: clustering, Count-Min Sketch, Misra-Gries, short digests.", TEAL],
    ["archive", "Cold storage", "Every raw chunk is kept on disk behind pointers. It is never searched, only read when needed.", AMBER],
    ["gate", "Error-bound gate", "At question time, each cell's Count-Min Sketch says whether its digest can vouch for the rare words in the question. If not, read the raw chunks.", VIOLET],
  ];
  ideas.forEach(([ic, head, body, col], i) => {
    const y = 1.85 + i * 1.6;
    iconCircle(s, 0.6, y, 0.95, ic, col, `idea ${i + 1}`);
    text(s, head, { x: 1.85, y: y + 0.02, w: 6.2, h: 0.45, fontSize: 22, bold: true, color: col });
    text(s, body, { x: 1.85, y: y + 0.5, w: 6.3, h: 0.95, fontSize: 15, color: INK, valign: "top" });
  });
  card(s, 8.75, 1.85, 3.95, 4.45, "idea summary card");
  text(s, "In one line", { x: 9.1, y: 2.15, w: 3.3, h: 0.35, fontSize: 13, bold: true, color: TEAL });
  text(s, "Search a small summary. Read the original only where a sketch's error bound says the summary may have dropped the fact.",
    { x: 9.1, y: 2.6, w: 3.3, h: 2.2, fontSize: 20, italic: true, color: INK, valign: "top" });
  text(s, "Memory stays bounded; context stays small; exact facts are still recoverable.", { x: 9.1, y: 5.0, w: 3.3, h: 1.0, fontSize: 13, color: MUTED, valign: "top" });
  s.addNotes("Three parts. A hot memory that is bounded no matter how long the stream gets. A cold store that keeps every raw chunk but is never searched. And the error-bound gate, which is the part we contribute: it uses the Count-Min Sketch guarantee to decide, per memory cell and per question, whether the summary is safe to use.");

  // 4 ---- architecture
  pres.addSection({ title: "Method" });
  s = content("Method", "Architecture", "From a stream of chunks to an answer");
  const box = (x, y, w, h, title, sub, col, name) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: CARD }, line: { color: col, width: 1.5 }, objectName: name });
    text(s, title, { x: x + 0.15, y: y + 0.12, w: w - 0.3, h: 0.38, fontSize: 15, bold: true, color: col, align: "center" });
    text(s, sub, { x: x + 0.15, y: y + 0.52, w: w - 0.3, h: h - 0.62, fontSize: 12.5, color: INK, align: "center", valign: "top" });
  };
  const arrow = (x1, y1, x2, y2, col, name, dash) => s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1, line: { color: col, width: 2, endArrowType: "triangle", dashType: dash || "solid" }, objectName: name });
  box(0.6, 2.15, 2.0, 1.6, "Stream", "documents arrive as 150-word chunks", BLUE, "stream box");
  box(3.15, 1.75, 3.3, 2.4, "Hot memory (bounded)", "≤ 64 cells: centroid + exemplars, Count-Min Sketch, 120-word digest\nGlobal: Misra-Gries topics, streaming IDF", TEAL, "hot memory box");
  box(3.15, 4.75, 3.3, 1.45, "Cold storage (grows)", "raw chunks + embeddings, reached through cell pointers", AMBER, "cold storage box");
  box(7.05, 2.15, 2.15, 1.6, "Rank cells", "question embedding vs centroids and exemplars; top 4 + 1 neighbour", BLUE, "rank box");
  box(9.8, 2.15, 2.95, 1.6, "Error-bound gate", "per cell: can the Count-Min Sketch vouch for the rare question terms?", VIOLET, "gate box");
  box(9.8, 4.75, 2.95, 1.45, "LLM answers", "one call, with digests + raw evidence for escalated cells", TEAL, "llm box");
  arrow(2.6, 2.95, 3.15, 2.95, BLUE, "arrow stream to hot");
  arrow(4.8, 4.15, 4.8, 4.75, AMBER, "arrow hot to cold", "dash");
  arrow(6.45, 2.95, 7.05, 2.95, BLUE, "arrow hot to rank");
  arrow(9.2, 2.95, 9.8, 2.95, BLUE, "arrow rank to gate");
  arrow(11.27, 3.75, 11.27, 4.75, TEAL, "arrow gate to llm");
  arrow(9.8, 3.6, 6.45, 5.35, AMBER, "arrow gate escalates to cold", "dash");
  arrow(6.45, 5.6, 9.8, 5.6, AMBER, "arrow cold raw to llm", "dash");
  text(s, "trusted → digest", { x: 11.35, y: 4.1, w: 1.6, h: 0.3, fontSize: 11, color: TEAL });
  text(s, "escalate → read raw chunks", { x: 7.75, y: 5.05, w: 2.0, h: 0.3, fontSize: 11, color: AMBER });
  text(s, "Single pass over the stream; ingest costs 2–26 ms per chunk on CPU.", { x: 0.6, y: 6.45, w: 12, h: 0.3, fontSize: 12, color: MUTED });
  s.addNotes("The data flow. Each chunk updates the hot memory in one pass and is written to cold storage. A question is embedded and compared with the cells; we keep the top four plus the cell that is next to the best one in time. The gate then checks each selected cell. Trusted cells contribute their digest; escalated cells are replaced by their two most relevant raw chunks from cold storage. The LLM is called once with that context.");

  // 5 ---- memory cell
  s = content("Method", "One memory cell", "A cell is a compact box for one topic in the stream");
  const parts = [
    ["radar", "Centroid + 8 exemplars", "where the topic sits in embedding space", BLUE],
    ["sigma", "Count-Min Sketch 3 × 512", "how often every word occurs in the cell", VIOLET],
    ["file", "Digest ≤ 120 words", "the most informative sentences, read by the LLM", AMBER],
    ["layers", "Span + radius", "where in the stream; how spread out (loose > 0.75)", TEAL],
    ["archive", "Raw-chunk pointers", "the way back to the original text in cold storage", AMBER],
    ["merge", "Merge when full", "over budget: merge the two closest cells (Ward)", PINK],
  ];
  parts.forEach(([ic, head, body, col], i) => {
    const x = 0.6 + (i % 2) * 4.0, y = 1.9 + Math.floor(i / 2) * 1.5;
    card(s, x, y, 3.75, 1.3, `cell part ${i + 1}`);
    iconCircle(s, x + 0.25, y + 0.3, 0.7, ic, col, `cell part ${i + 1}`);
    text(s, head, { x: x + 1.1, y: y + 0.22, w: 2.5, h: 0.4, fontSize: 15, bold: true, color: col });
    text(s, body, { x: x + 1.1, y: y + 0.62, w: 2.5, h: 0.55, fontSize: 12.5, color: INK, valign: "top" });
  });
  card(s, 8.85, 1.9, 3.85, 4.3, "cell size card");
  text(s, "≈ 14 KB", { x: 9.2, y: 2.25, w: 3.2, h: 0.9, fontSize: 52, bold: true, color: TEAL });
  text(s, "per cell, mostly sketch and exemplars", { x: 9.2, y: 3.15, w: 3.2, h: 0.4, fontSize: 14, color: MUTED });
  text(s, [
    { text: `${(MB(c64[c64.length - 1].sketch_hot_bytes)).toFixed(1)} MB`, options: { bold: true, color: TEAL, breakLine: true } },
    { text: "hot memory with 64 cells after 8.5M words", options: { color: INK, breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: `${(MB(c256[c256.length - 1].sketch_hot_bytes)).toFixed(1)} MB`, options: { bold: true, color: TEAL, breakLine: true } },
    { text: "with 256 cells", options: { color: INK } },
  ], { x: 9.2, y: 3.75, w: 3.2, h: 2.2, fontSize: 16, valign: "top" });
  s.addNotes("What one cell stores. The centroid and up to eight exemplar embeddings are used to find cells. The Count-Min Sketch counts every word in the cell in a fixed 3 by 512 table; this is what the gate checks. The digest is up to 120 words of the best sentences. Pointers lead back to the raw chunks. A chunk joins a cell if its cosine similarity is at least 0.9, otherwise it opens a new cell; when the budget is exceeded, the two closest cells are merged. About 14 KB per cell, which is why memory stays at about 1 MB for 64 cells however long the stream gets.");

  // 6 ---- the gate
  s = content("Method", "The error-bound gate", "Read raw text only where the sketch can't vouch for it");
  const rules = [
    ["Lossy", "A rare query term is in the cell (Count-Min above its noise floor) but missing from the digest: the summary dropped the fact.", AMBER],
    ["Unresolved", "No retrieved cell holds the rare terms above the guaranteed εN bound: the sketch cannot locate them, so read the top two cells raw.", VIOLET],
    ["Loose", "The cell's radius is above 0.75: it mixes too much for its digest to be faithful.", BLUE],
  ];
  rules.forEach(([head, body, col], i) => {
    const y = 1.85 + i * 1.45;
    card(s, 0.6, y, 7.3, 1.25, `gate rule ${i + 1}`);
    text(s, head, { x: 0.95, y: y + 0.18, w: 2.0, h: 0.4, fontSize: 20, bold: true, color: col });
    text(s, body, { x: 2.85, y: y + 0.18, w: 4.85, h: 0.95, fontSize: 14, color: INK, valign: "top" });
  });
  text(s, "Questions with no rare terms (summaries) are answered from the digests.", { x: 0.6, y: 6.25, w: 7.3, h: 0.35, fontSize: 13, color: MUTED });
  card(s, 8.4, 1.85, 4.3, 4.15, "kv example card");
  iconCircle(s, 8.75, 2.15, 0.7, "target", VIOLET, "kv example");
  text(s, "Exact-key lookup (∞-Bench KV)", { x: 9.6, y: 2.25, w: 3.0, h: 0.5, fontSize: 15, bold: true });
  text(s, [
    { text: `${pt.sketchrag_sketch.kv_retrieval}`, options: { fontSize: 46, bold: true, color: MUTED } },
    { text: "  →  ", options: { fontSize: 30, color: MUTED } },
    { text: `${pt.sketchrag_gate.kv_retrieval}`, options: { fontSize: 46, bold: true, color: TEAL } },
  ], { x: 8.75, y: 3.1, w: 3.8, h: 1.0 });
  text(s, "score with digest only → with the gate. A UUID key never survives summarisation; the sketch knows it cannot vouch for it.",
    { x: 8.75, y: 4.2, w: 3.65, h: 1.5, fontSize: 13.5, color: INK, valign: "top" });
  s.addNotes(`The gate is deterministic, no LLM involved. It looks at the rare terms in the question, by streaming IDF. For each retrieved cell: if the sketch says a rare term occurs in the cell but the digest does not contain it, the digest lost the fact. If no cell has the rare terms above the guaranteed error bound epsilon times N, the sketch cannot locate them at all, so the top cells are read raw. Very loose cells are always escalated. On key-value retrieval this takes the score from ${pt.sketchrag_sketch.kv_retrieval} to ${pt.sketchrag_gate.kv_retrieval}.`);

  // 7 ---- setup
  s = content("Method", "Experimental setup", "One model, one prompt; only the context changes");
  const setup = [
    ["list", "350 test questions", "12 tasks: 9 from LongBench (single- and multi-doc QA, summaries, passage retrieval) and 3 from ∞-Bench (PassKey, KV, En.QA)"],
    ["cpu", "qwen2.5:7b-instruct", "through Ollama on Kaggle GPUs; greedy decoding, fixed seed, same prompt for every system"],
    ["scale", "9 compared systems", "full context, BM25 RAG, dense RAG, SARA-lite, AMBER-lite, SketchRAG digest only / always raw / LLM agent / gate + agent"],
    ["check", "Honest protocol", "settings tuned only on a dev split (seed 1000); test split (seed 0) frozen; EM-LLM run from its official code"],
  ];
  setup.forEach(([ic, head, body], i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 1.95 + Math.floor(i / 2) * 2.25;
    card(s, x, y, 5.95, 2.0, `setup card ${i + 1}`);
    iconCircle(s, x + 0.3, y + 0.35, 0.8, ic, TEAL, `setup ${i + 1}`);
    text(s, head, { x: x + 1.35, y: y + 0.35, w: 4.4, h: 0.45, fontSize: 20, bold: true });
    text(s, body, { x: x + 1.35, y: y + 0.85, w: 4.4, h: 1.0, fontSize: 14, color: MUTED, valign: "top" });
  });
  s.addNotes("Setup. 350 test questions over 12 tasks. Every system uses the same 7B model, the same prompt and greedy decoding, so differences come only from the context each system builds. We compare against full context, BM25 and dense retrieval, our simplified versions of SARA and AMBER, and several SketchRAG variants. Tuning used a separate dev split; the test split was frozen before tuning.");

  // 8 ---- main results
  pres.addSection({ title: "Results" });
  s = content("Results", "Answer quality", "The gate beats full context and dense RAG; BM25 is still best");
  const order = D.systems.filter((x) => x.key !== "sketchrag_hybrid");
  const colorOf = (k) => (k === "sketchrag_gate" ? H.accent1 : k === "bm25_rag" ? H.accent4 : k.startsWith("sketchrag") ? "2F7F74" : "3B4A63");
  s.addChart(pres.charts.BAR, [{ name: "Average score", labels: order.map((x) => x.label), values: order.map((x) => x.score) }], {
    x: 0.6, y: 1.75, w: 7.6, h: 4.95, barDir: "bar", catAxisOrientation: "maxMin", chartColors: order.map((x) => colorOf(x.key)),
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 50, valAxisHidden: true,
    barGapWidthPct: 45, showLegend: false, showTitle: true, title: "Average task score (× 100), 350 questions", titleColor: H.lt2, titleFontSize: 13, ...chartText,
  });
  const res = [
    [f1(gate.score), "SketchRAG + gate", `beats full context (${f1(full.score)}), dense RAG (${f1(dense.score)}), SARA-lite (${f1(sara.score)}), AMBER-lite (${f1(amber.score)})`, TEAL],
    [f1(bm25.score), "BM25 RAG", "the best system overall, mostly on exact-match tasks", AMBER],
    [f1(sketch.score), "digest only", `summaries alone lose exact facts; the gate adds ${f1(gate.score - sketch.score)} points`, MUTED],
  ];
  res.forEach(([big, label, desc, col], i) => {
    const y = 1.85 + i * 1.6;
    card(s, 8.6, y, 4.1, 1.4, `result card ${i + 1}`);
    text(s, big, { x: 8.9, y: y + 0.2, w: 1.5, h: 0.8, fontSize: 36, bold: true, color: col });
    text(s, label, { x: 10.35, y: y + 0.2, w: 2.25, h: 0.35, fontSize: 15, bold: true });
    text(s, desc, { x: 10.35, y: y + 0.55, w: 2.25, h: 0.8, fontSize: 12, color: MUTED, valign: "top" });
  });
  s.addNotes(`Answer quality. SketchRAG with the gate scores ${gate.score}, ahead of full context, dense RAG, SARA-lite and AMBER-lite. We report honestly that plain BM25 retrieval is the best system at ${bm25.score}, mostly because several tasks reward exact string matching. The digest alone scores ${sketch.score}; the gate is worth almost 12 points.`);

  // 9 ---- cost
  s = content("Results", "Context cost", `${fmt(gate.tokens)} tokens per question: the cheapest context`);
  const byTok = [...D.systems].filter((x) => x.key !== "sketchrag_hybrid").sort((a, b) => a.tokens - b.tokens);
  s.addChart(pres.charts.BAR, [{ name: "Tokens", labels: byTok.map((x) => x.label), values: byTok.map((x) => x.tokens) }], {
    x: 0.6, y: 1.75, w: 7.6, h: 4.95, barDir: "bar", catAxisOrientation: "maxMin", chartColors: byTok.map((x) => colorOf(x.key)),
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "#,##0", valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 17000,
    barGapWidthPct: 45, showLegend: false, showTitle: true, title: "Mean LLM context tokens per question", titleColor: H.lt2, titleFontSize: 13, ...chartText,
  });
  const cost = [
    [`${(100 * gate.tokens / full.tokens).toFixed(1)}%`, "of full context's tokens", TEAL],
    [`${gate.llm_s} s`, `LLM time per question, vs ${full.llm_s} s for full context`, BLUE],
    ["1 call", `per question; AMBER-lite needs ${amber.calls.toFixed(1)}`, VIOLET],
  ];
  cost.forEach(([big, label, col], i) => {
    const y = 1.85 + i * 1.6;
    card(s, 8.6, y, 4.1, 1.4, `cost card ${i + 1}`);
    text(s, big, { x: 8.9, y: y + 0.2, w: 3.6, h: 0.65, fontSize: 34, bold: true, color: col });
    text(s, label, { x: 8.9, y: y + 0.85, w: 3.6, h: 0.45, fontSize: 13.5, color: INK });
  });
  s.addNotes(`Cost. The gated system sends ${fmt(gate.tokens)} tokens per question, the fewest of any system and ${(100 * gate.tokens / full.tokens).toFixed(1)} percent of full context, and needs ${gate.llm_s} seconds of LLM time against ${full.llm_s}. It makes one LLM call per question; AMBER-lite, which lets the LLM manage memory, makes about three.`);

  // 10 ---- calibration
  s = content("Results", "Is the gate calibrated?", "It escalates exactly where raw evidence matters");
  s.addChart(pres.charts.BAR, [
    { name: "Digest only", labels: [`Gate fired (${cal.fired} q)`, `Gate trusted (${cal.trusted} q)`], values: [cal.fired_sketch, cal.trusted_gate] },
    { name: "With raw evidence", labels: [`Gate fired (${cal.fired} q)`, `Gate trusted (${cal.trusted} q)`], values: [cal.fired_gate, cal.trusted_raw] },
  ], {
    x: 0.6, y: 1.75, w: 6.6, h: 4.95, barDir: "col", chartColors: ["3B4A63", H.accent1], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 60, valAxisHidden: true, barGapWidthPct: 60, showLegend: true, legendPos: "b",
    showTitle: true, title: "Average score on each group of questions", titleColor: H.lt2, titleFontSize: 13, ...chartText,
  });
  const gainGate = gate.score - sketch.score, gainRaw = raw.score - sketch.score;
  const calib = [
    [`${Math.round(100 * cal.fired / cal.n)}%`, "of questions escalated", `${cal.fired} of ${cal.n}; there raw text lifts the score ${f1(cal.fired_sketch)} → ${f1(cal.fired_gate)}`, AMBER],
    [`${Math.round(100 * gainGate / gainRaw)}%`, "of the always-raw gain", `with ${Math.round(100 * (1 - gate.tokens / raw.tokens))}% fewer tokens than always reading raw`, TEAL],
    [`${cal.agent_tool_calls} / ${cal.n}`, "tool calls by an LLM agent", `left to decide itself, a 7B model rarely fetches raw text and scores ${agent.score}`, PINK],
  ];
  calib.forEach(([big, label, desc, col], i) => {
    const y = 1.85 + i * 1.6;
    card(s, 7.6, y, 5.1, 1.4, `calibration card ${i + 1}`);
    text(s, big, { x: 7.9, y: y + 0.2, w: 1.75, h: 0.75, fontSize: 30, bold: true, color: col });
    text(s, label, { x: 9.7, y: y + 0.2, w: 2.85, h: 0.35, fontSize: 14.5, bold: true });
    text(s, desc, { x: 9.7, y: y + 0.56, w: 2.85, h: 0.8, fontSize: 12, color: MUTED, valign: "top" });
  });
  s.addNotes(`Calibration. The gate fires on ${cal.fired} of ${cal.n} questions. On those, reading raw evidence lifts the score from ${cal.fired_sketch} to ${cal.fired_gate}. On the ${cal.trusted} questions it trusts, always reading raw would add less than one point. So it recovers ${Math.round(100 * gainGate / gainRaw)} percent of the always-raw gain with ${Math.round(100 * (1 - gate.tokens / raw.tokens))} percent fewer tokens. When we let the LLM decide instead, through tool calls, it fetched raw text on only ${cal.agent_tool_calls} questions and scored ${agent.score}: the useful signal is the sketch's error bound, not the model's self-assessment.`);

  // 11 ---- what matters
  s = content("Results", "Ablations and memory budget", "Raw pointers matter most; cell budget barely does");
  const ab = D.ablations;
  s.addChart(pres.charts.BAR, [{ name: "Score", labels: ab.map((a) => a.label), values: ab.map((a) => a.score) }], {
    x: 0.6, y: 1.75, w: 6.2, h: 4.95, barDir: "bar", catAxisOrientation: "maxMin",
    chartColors: ab.map((a, i) => (i === 0 ? H.accent1 : a.score < ab[0].score - 5 ? H.accent4 : "3B4A63")),
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 50, valAxisHidden: true,
    barGapWidthPct: 45, showLegend: false, showTitle: true, title: "Gated score per ablation (110 questions)", titleColor: H.lt2, titleFontSize: 13, ...chartText,
  });
  const sw = D.sweep_cells;
  s.addChart(pres.charts.LINE, [
    { name: "SketchRAG + gate", labels: sw["SketchRAG + gate"].map((p) => String(p.max_cells)), values: sw["SketchRAG + gate"].map((p) => p.score) },
    { name: "Digest only", labels: sw["SketchRAG digest only"].map((p) => String(p.max_cells)), values: sw["SketchRAG digest only"].map((p) => p.score) },
  ], {
    x: 7.2, y: 1.75, w: 5.5, h: 3.75, chartColors: [H.accent1, "8A9AB2"], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 7,
    valAxisMinVal: 20, valAxisMaxVal: 50, showLegend: true, legendPos: "b", showTitle: true, title: "Score vs max cells (8 → 256)", titleColor: H.lt2,
    titleFontSize: 13, ...chartText,
  });
  const g = sw["SketchRAG + gate"];
  text(s, `Fewer cells → coarser digests → the gate escalates more (fire rate ${g[g.length - 1].fallback} → ${g[0].fallback}) and holds the score at ${Math.min(...g.map((p) => p.score))}–${Math.max(...g.map((p) => p.score))}. Hot memory: ${g[0].hot_kb}–${g[g.length - 1].hot_kb} KB.`,
    { x: 7.2, y: 5.6, w: 5.5, h: 1.0, fontSize: 13, color: MUTED, valign: "top" });
  s.addNotes(`Ablations on 110 questions. Capping each cell's raw pointers at a 16-item reservoir drops the score from ${ab[0].score} to ${ab[1].score}: the chunk with the answer is often no longer reachable. Removing the temporal neighbour costs about 1.7 points; the rest is within run-to-run noise of about 1 point. On the right, shrinking memory from 256 to 8 cells: the digest-only score falls, but the gate compensates by escalating more often, so the gated score stays between ${Math.min(...g.map((p) => p.score))} and ${Math.max(...g.map((p) => p.score))}.`);

  // 12 ---- stress test
  s = content("Results", "Long-stream stress test", "8.5M words: memory stays flat, but recall drops");
  s.addChart(pres.charts.LINE, [
    { name: "Dense RAG index", labels: c256.map((c) => (c.words / 1e6).toFixed(1)), values: c256.map((c) => +MB(c.dense_bytes).toFixed(1)) },
    { name: "Raw text", labels: c256.map((c) => (c.words / 1e6).toFixed(1)), values: c256.map((c) => +MB(c.text_bytes).toFixed(1)) },
    { name: "SketchRAG, 256 cells", labels: c256.map((c) => (c.words / 1e6).toFixed(1)), values: c256.map((c) => +MB(c.sketch_hot_bytes).toFixed(2)) },
    { name: "SketchRAG, 64 cells", labels: c64.map((c) => (c.words / 1e6).toFixed(1)), values: c64.map((c) => +MB(c.sketch_hot_bytes).toFixed(2)) },
  ], {
    x: 0.6, y: 1.75, w: 7.4, h: 4.8, chartColors: [H.accent2, H.accent4, H.accent3, H.accent1], lineSize: 2.5, lineDataSymbol: "none",
    valAxisMinVal: 0, valAxisTitle: "MB", showValAxisTitle: true, valAxisTitleColor: H.lt2, valAxisTitleFontSize: 12, catAxisTitle: "million words streamed",
    showCatAxisTitle: true, catAxisTitleColor: H.lt2, catAxisTitleFontSize: 12, showLegend: true, legendPos: "b", showTitle: true,
    title: "Searchable memory vs stream length", titleColor: H.lt2, titleFontSize: 13, ...chartText,
  });
  text(s, "Dense index size estimated as raw text + one 384-d float32 vector per chunk (same accounting as the stress-test script).", { x: 0.6, y: 6.72, w: 7.4, h: 0.28, fontSize: 10, color: MUTED });
  const rc = D.stress.recall;
  card(s, 8.4, 1.85, 4.3, 2.0, "stress memory card");
  text(s, `${MB(c64[c64.length - 1].sketch_hot_bytes).toFixed(1)} MB vs ${Math.round(MB(c256[c256.length - 1].dense_bytes))} MB`,
    { x: 8.7, y: 2.1, w: 3.8, h: 0.7, fontSize: 30, bold: true, color: TEAL });
  text(s, "SketchRAG hot memory (64 cells) vs a dense index after 8.5M words", { x: 8.7, y: 2.85, w: 3.8, h: 0.8, fontSize: 13, color: INK, valign: "top" });
  card(s, 8.4, 4.1, 4.3, 2.6, "stress recall card");
  iconCircle(s, 8.7, 4.35, 0.6, "alert", AMBER, "stress limitation");
  text(s, "Main limitation", { x: 9.45, y: 4.43, w: 3.0, h: 0.4, fontSize: 16, bold: true, color: AMBER });
  text(s, `Answer recall: dense RAG ${rc.dense_rag}, BM25 ${rc.bm25_rag}, SketchRAG gate ${rc.sketchrag_gate} (256 cells). With thousands of unrelated documents sharing a few hundred cells, a centroid is too blunt a key.`,
    { x: 8.7, y: 5.05, w: 3.85, h: 1.55, fontSize: 13, color: INK, valign: "top" });
  s.addNotes(`The stress test streams all English LongBench documents, ${fmt(D.stress.n_words)} words, through the system. Hot memory stays at about ${MB(c64[c64.length - 1].sketch_hot_bytes).toFixed(1)} MB with 64 cells, while a dense index grows to about ${Math.round(MB(c256[c256.length - 1].dense_bytes))} MB. But we have to be honest about recall: at this scale SketchRAG finds the evidence for ${rc.sketchrag_gate} of probe questions against ${rc.dense_rag} for dense retrieval. Every probe escalates, so it behaves like a two-level router over cold storage. This is the main limitation we report.`);

  // 13 ---- EM-LLM
  s = content("Results", "Against EM-LLM", "Same model, same 120 questions: within a point of EM-LLM");
  const em = D.emllm.systems;
  s.addChart(pres.charts.BAR, [{ name: "Average", labels: em.map((e) => e.label), values: em.map((e) => e.avg) }], {
    x: 0.6, y: 1.75, w: 7.4, h: 4.95, barDir: "bar", catAxisOrientation: "maxMin",
    chartColors: em.map((e) => (e.label === "SketchRAG + gate" ? H.accent1 : e.label.startsWith("EM-LLM") ? H.accent5 : "3B4A63")),
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 45, valAxisHidden: true,
    barGapWidthPct: 45, showLegend: false, showTitle: true, title: "Average over 6 LongBench tasks · Phi-3.5-mini for every system", titleColor: H.lt2, titleFontSize: 13, ...chartText,
  });
  const emGate = em.find((e) => e.label === "SketchRAG + gate").avg, emEm = em.find((e) => e.label.startsWith("EM-LLM")).avg;
  card(s, 8.4, 1.85, 4.3, 2.1, "emllm compare card");
  text(s, [{ text: `${emGate}`, options: { color: TEAL } }, { text: "  vs  ", options: { color: MUTED, fontSize: 24 } }, { text: `${emEm}`, options: { color: PINK } }],
    { x: 8.7, y: 2.1, w: 3.8, h: 0.8, fontSize: 40, bold: true });
  text(s, "SketchRAG gate vs EM-LLM's official code. SketchRAG needs one generation per question instead of streaming the whole document through the model.",
    { x: 8.7, y: 2.9, w: 3.8, h: 1.0, fontSize: 13, color: INK, valign: "top" });
  card(s, 8.4, 4.2, 4.3, 2.5, "emllm caveat card");
  text(s, "Caveats", { x: 8.7, y: 4.4, w: 3.8, h: 0.4, fontSize: 16, bold: true, color: AMBER });
  text(s, [
    { text: "EM-LLM ran in fp16; our Phi-3.5 is Ollama's quantised build", options: { bullet: true, breakLine: true } },
    { text: "Reduced scale on Kaggle 2×T4: 6 of 9 planned tasks", options: { bullet: true, breakLine: true } },
    { text: "Full context is highest here (37.7)", options: { bullet: true } },
  ], { x: 8.7, y: 4.85, w: 3.85, h: 1.75, fontSize: 13, color: INK, paraSpaceAfter: 4, valign: "top" });
  s.addNotes(`We reproduced EM-LLM, an episodic-memory method, from its official repository, at reduced scale on Kaggle's two T4 GPUs. With the same base model and the same ${D.emllm.questions} questions, SketchRAG with the gate scores ${emGate} and EM-LLM ${emEm}. The caveats: EM-LLM ran in fp16 while our Phi-3.5 is a quantised build, so it is the same questions but not an identical model, and only six tasks finished within the hardware limits.`);

  // 14 ---- demo
  pres.addSection({ title: "Demo and conclusion" });
  s = content("Demo and conclusion", "Live demo", "An agent monitoring a feed, answering live on a laptop");
  s.addImage({ path: path.join(__dirname, "img", "demo_query.png"), x: 0.6, y: 1.7, w: 7.04, h: 4.4, objectName: "demo query screenshot" });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.7, w: 7.04, h: 4.4, fill: { type: "none" }, line: { color: LINE, width: 1 }, objectName: "demo screenshot frame" });
  const demo = [
    ["waves", "Feed", "58 LongBench documents (≈ 400k words) stream into a 256-cell memory"],
    ["search", "Ask", "the trace shows ranking, the gate's verdict per cell and the raw chunks it read"],
    ["sparkles", "Answer", "llama3.2:3b via Ollama on the CPU, streamed in ≈ 30–60 s, checked against the known answer"],
  ];
  demo.forEach(([ic, head, body], i) => {
    const y = 1.8 + i * 1.5;
    iconCircle(s, 8.2, y, 0.7, ic, TEAL, `demo ${i + 1}`);
    text(s, head, { x: 9.1, y: y + 0.02, w: 3.6, h: 0.38, fontSize: 17, bold: true, color: TEAL });
    text(s, body, { x: 9.1, y: y + 0.42, w: 3.6, h: 0.95, fontSize: 13, color: INK, valign: "top" });
  });
  text(s, `Live run shown: "Where was the father of Teobaldo II Ordelaffi born?" → Italy (matches the known answer). Feed-wide, the gate's context holds the answer for ${Math.round(100 * D.presets_feed.gate)}% of questions (dense ${Math.round(100 * D.presets_feed.dense)}%).`,
    { x: 0.6, y: 6.3, w: 12.1, h: 0.55, fontSize: 12, color: MUTED, valign: "top" });
  s.addNotes("Live demo: python demo/agent/server.py, then open localhost:8600. Show the Live stream page filling, the Memory explorer grid, then ask the Teobaldo question in the Query playground and click Compare to show the digest-only answer. Note: the demo answers with a 3B model on this laptop, not the 7B model used for the benchmark numbers.");

  // 15 ---- limitations
  s = content("Demo and conclusion", "Limitations", "What SketchRAG does not solve yet");
  const lim = [
    ["scale", "BM25 is still stronger", `A plain lexical baseline scores ${f1(bm25.score)} vs ${f1(gate.score)}; these benchmarks reward exact string matches.`],
    ["git", "Multi-hop questions", `On 2WikiMQA the gate scores ${pt.sketchrag_gate["2wikimqa"]} vs ${pt.sara_lite["2wikimqa"]} (SARA-lite) and ${pt.full_context["2wikimqa"]} (full context): evidence spread over many chunks does not fit in 4 cells.`],
    ["down", "Recall at stream scale", `At 8.5M words recall falls to ${rc.sketchrag_gate} vs ${rc.dense_rag} for dense RAG; cell centroids are too coarse.`],
    ["cpu", "Reduced-scale baselines", "SARA-lite and AMBER-lite are our re-implementations; EM-LLM ran on 6 of 9 tasks."],
  ];
  lim.forEach(([ic, head, body], i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 1.95 + Math.floor(i / 2) * 2.25;
    card(s, x, y, 5.95, 2.0, `limitation card ${i + 1}`);
    iconCircle(s, x + 0.3, y + 0.35, 0.8, ic, AMBER, `limitation ${i + 1}`);
    text(s, head, { x: x + 1.35, y: y + 0.32, w: 4.4, h: 0.45, fontSize: 19, bold: true, color: AMBER });
    text(s, body, { x: x + 1.35, y: y + 0.82, w: 4.4, h: 1.05, fontSize: 13.5, color: INK, valign: "top" });
  });
  s.addNotes("Limitations, stated plainly. BM25 remains the strongest system. Multi-hop QA suffers because the evidence is spread across more chunks than four cells hold. At stream scale, recall drops well below dense retrieval. And two of our baselines are simplified re-implementations, while EM-LLM ran at reduced scale.");

  // 16 ---- conclusion
  s = content("Demo and conclusion", "Conclusion", "Bounded memory, with a sketch deciding what to re-read");
  const take = [
    ["boxes", "Bounded", `Hot memory stays ≈ ${MB(c64[c64.length - 1].sketch_hot_bytes).toFixed(1)} MB at 8.5M words, against a ${Math.round(MB(c256[c256.length - 1].dense_bytes))} MB dense index.`, TEAL],
    ["coins", "Cheap", `${fmt(gate.tokens)} tokens and ${gate.llm_s} s per question; a score of ${gate.score}, ahead of full context, dense RAG, SARA-lite and AMBER-lite.`, BLUE],
    ["gate", "Calibrated", `The Count-Min bound picks where raw text matters (${f1(cal.fired_sketch)} → ${f1(cal.fired_gate)}); an LLM left to decide does not.`, VIOLET],
  ];
  take.forEach(([ic, head, body, col], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.95, 3.8, 3.3, `takeaway card ${i + 1}`);
    iconCircle(s, x + 0.35, 2.3, 0.85, ic, col, `takeaway ${i + 1}`);
    text(s, head, { x: x + 0.35, y: 3.35, w: 3.1, h: 0.5, fontSize: 24, bold: true, color: col });
    text(s, body, { x: x + 0.35, y: 3.9, w: 3.15, h: 1.25, fontSize: 14, color: INK, valign: "top" });
  });
  text(s, "Possible next steps: finer-grained keys for long streams (sub-cell exemplars or a small lexical index over digests), and a multi-hop-aware gate.",
    { x: 0.6, y: 5.6, w: 12.1, h: 0.45, fontSize: 15, color: MUTED });
  text(s, "Thank you · Questions?", { x: 0.6, y: 6.15, w: 12.1, h: 0.6, fontSize: 26, bold: true, color: INK });
  s.addNotes("Three takeaways: memory is bounded, context is cheap, and the gate is calibrated because it comes from the sketch's error bound. The obvious next steps target the stream-scale recall problem and multi-hop questions. Thank you; happy to take questions or show the demo again.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
}

build().catch((e) => { console.error(e); process.exit(1); });

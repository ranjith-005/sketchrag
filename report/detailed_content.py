"""Plain-English theory and explanation text for the detailed SketchRAG report.

Only fixed text lives here. Every result number in the detailed report is computed from
results/ by build_detailed.py.
"""

READING_GUIDE = """
<div class="note"><b>How to read this report.</b> It is the long, explained version of our project report.
Sections 2&ndash;4 teach the background ideas from scratch, in simple language, with small worked examples.
Sections 5&ndash;7 explain what we built and how we tested it. Sections 8&ndash;10 give every result, each
followed by a short &ldquo;what this means&rdquo; paragraph. A glossary of all technical words is at the end.
</div>
"""

# ---------------------------------------------------------------- 2. background
BACKGROUND = """
<h3>2.1 Large language models, tokens and the context window</h3>
<p>A <b>large language model (LLM)</b>, such as Qwen or Phi, is a program that reads text and writes text. It does
not read letters or whole words. It reads <b>tokens</b>: small pieces of text, roughly three quarters of an
English word each. &ldquo;Streaming&rdquo; might be one token; a long ID such as
<code>9f2c-41aa-...</code> may be ten.</p>
<p>Everything the model is allowed to look at when answering (the instructions, the question and any documents)
must fit inside its <b>context window</b>. The model we used, <code>qwen2.5:7b-instruct</code>, was given a window
of 32,768 tokens. A long novel is about 190,000 tokens, so it simply does not fit.</p>
<p>Even when a document does fit, putting all of it in the prompt has two costs:</p>
<ul>
<li><b>Time and money.</b> The work the model does grows with the number of tokens it reads. Ten times more
context means roughly ten times more computing.</li>
<li><b>Lost in the middle.</b> Models are good at using text at the start and end of a long prompt, but often
miss facts buried in the middle.</li>
</ul>

<h3>2.2 Retrieval-augmented generation (RAG)</h3>
<p><b>RAG</b> is the standard fix. Instead of giving the model the whole document, we:</p>
<ol>
<li>cut the document into small <b>chunks</b> (in our project about 150 words each);</li>
<li>when a question arrives, <b>retrieve</b> the few chunks most related to it;</li>
<li>give only those chunks to the model, which then <b>generates</b> the answer.</li>
</ol>
<p>Think of an open-book exam. You cannot read the whole textbook during the exam, so you use the index to find
the right two pages and read only those. Retrieval is the index.</p>
<p>There are two common ways to decide which chunks are &ldquo;related&rdquo;:</p>
<ul>
<li><b>Lexical (word-matching) retrieval: BM25.</b> A chunk scores high if it contains the question's words,
especially <i>rare</i> words. Matching the word &ldquo;the&rdquo; means nothing; matching &ldquo;Ravenna&rdquo;
means a lot. How rare a word is gets measured by <b>IDF</b> (inverse document frequency):
<div class="formula">IDF(word) = log( N / df(word) )</div>
where N is the number of chunks and df is how many chunks contain the word. A word found in every chunk has
IDF = log(1) = 0. A word found in one chunk out of 1,000 has IDF = log(1000) &asymp; 6.9.</li>
<li><b>Dense (meaning-based) retrieval.</b> An <b>embedding model</b> (we used <code>bge-small-en-v1.5</code>) turns
any piece of text into a list of 384 numbers, called a <b>vector</b> or <b>embedding</b>. Texts with similar
meaning get vectors that point in similar directions, even when they share no words (&ldquo;car&rdquo; and
&ldquo;automobile&rdquo;). Similarity is measured by <b>cosine similarity</b>: the cosine of the angle between two
vectors, from &minus;1 (opposite) to 1 (same direction). Retrieval means picking the chunks whose vectors have
the highest cosine similarity with the question's vector. A library called <b>FAISS</b> does this search
quickly.</li>
</ul>

<h3>2.3 Why RAG struggles with streams</h3>
<p>Normal RAG keeps <i>every</i> chunk and <i>every</i> vector forever. One 384-number vector in 32-bit floats takes
1.5 KB, plus the chunk's text. For a single document that is nothing. But an <b>agent</b> that reads a
continuous stream (chat logs, news, sensor reports, meeting transcripts, months of emails) keeps adding
chunks. Its searchable index grows <b>linearly</b>: twice the stream, twice the memory, and slower searches. In our
stress test of 8.5 million words, the dense-RAG index grew to about 146 MB, and it would keep growing.</p>

<h3>2.4 The data-stream model</h3>
<p>Computer science has studied this exact problem for decades, under the name <b>streaming algorithms</b>. The
rules of the streaming model are strict:</p>
<ul>
<li>items arrive one at a time, and the stream may never end;</li>
<li>each item can be looked at <b>once</b> (one pass), and cannot be fetched again later;</li>
<li>the memory allowed is <b>much smaller</b> than the stream: ideally a fixed size, or growing very slowly.</li>
</ul>
<p>Under these rules you cannot keep exact answers to most questions (&ldquo;how many times did word X
appear?&rdquo;). Streaming algorithms instead keep a small <b>sketch</b> (a summary) that gives
<b>approximate</b> answers with a <b>proven error guarantee</b>, for example &ldquo;the answer is off by at most
0.5% of the stream length, 95% of the time&rdquo;. Section 3 explains the sketches we used.</p>

<h3>2.5 Agentic AI</h3>
<p>An <b>AI agent</b> is an LLM that can take actions instead of just answering once. It can call <b>tools</b> (search
a database, open a file, fetch more evidence), look at the result and decide what to do next. In our project,
the key decision an agent faces is: <i>&ldquo;Is my summary good enough to answer this question, or do I need to
go back and read the original text?&rdquo;</i> Reading the original text is more accurate but costs more. Making
that decision well is the central idea of SketchRAG.</p>
"""

# ---------------------------------------------------------------- 3. streaming theory
SKETCH_THEORY = """
<p>This section explains, with small examples, the five streaming tools SketchRAG is built from.</p>

<h3>3.1 Count-Min Sketch: approximate counting</h3>
<p><b>Problem.</b> We want to know how often each word appeared in a stream, but there may be millions of
different words and we cannot keep a counter for each one.</p>
<p><b>Idea.</b> Keep a small table of counters with <b>d rows</b> and <b>w columns</b>. Each row has its own
<b>hash function</b>: a fixed, random-looking rule that maps any word to one column of that row. To <b>add</b> a
word, go to each row, find the word's column and add 1. To <b>ask</b> how often a word appeared, look at the word's
column in every row and take the <b>smallest</b> value.</p>
<p>Different words can land in the same column (a <b>collision</b>), so a counter may include other words' counts
too. Collisions can only <i>add</i>, never subtract. So every counter is <b>at least</b> the true count, and the
smallest one is the most honest guess.</p>

<div class="eg"><b>Worked example</b> (w = 4 columns, d = 2 rows). Stream: <i>apple, banana, apple, cherry</i>.
Suppose the hash functions send the words here:
<table class="small"><tr><th>word</th><th>row 1 column</th><th>row 2 column</th><th>true count</th></tr>
<tr><td>apple</td><td>1</td><td>2</td><td>2</td></tr>
<tr><td>banana</td><td>3</td><td>2</td><td>1</td></tr>
<tr><td>cherry</td><td>1</td><td>4</td><td>1</td></tr></table>
After the four words, the table is:
<table class="small"><tr><th></th><th>col 1</th><th>col 2</th><th>col 3</th><th>col 4</th></tr>
<tr><td>row 1</td><td>3 <span class="dim">(apple 2 + cherry 1)</span></td><td>0</td><td>1</td><td>0</td></tr>
<tr><td>row 2</td><td>0</td><td>3 <span class="dim">(apple 2 + banana 1)</span></td><td>0</td><td>1</td></tr></table>
Queries: banana = min(1, 3) = <b>1</b> (correct). cherry = min(3, 1) = <b>1</b> (correct).
apple = min(3, 3) = <b>3</b>, but the truth is 2: both of apple's counters had a collision.
The answer is too high but <b>never too low</b>, and with more columns collisions become rare.</div>

<p><b>The guarantee.</b> Let N be the total number of items added. If the table has
w = &lceil;e/&epsilon;&rceil; columns and d = &lceil;ln(1/&delta;)&rceil; rows (e &asymp; 2.718), then for every word x:</p>
<div class="formula">true(x) &le; estimate(x) &le; true(x) + &epsilon;&middot;N &nbsp;&nbsp; with probability at least 1 &minus; &delta;</div>
<p>In words: the estimate is never below the truth, and is above it by at most a fraction &epsilon; of the whole
stream, except with a small failure chance &delta;. SketchRAG uses w = 512 and d = 3, so
&epsilon; = e/512 &asymp; 0.0053 and &delta; = e<sup>&minus;3</sup> &asymp; 0.05. A sketch that has seen 10,000 words
over-counts any word by at most about 53, with 95% confidence. The memory is fixed at 512 &times; 3 counters,
no matter how many different words arrive.</p>
<p><b>Conservative update</b> (which we use) is a small improvement: when adding, a counter is raised only as far
as needed (to the current minimum plus one), not blindly incremented. This keeps the same guarantee and gives
smaller over-counts in practice.</p>
<p><b>Merging.</b> Two Count-Min tables with the same hash functions can be merged by simply adding them cell by
cell. The result is exactly the table you would get from the combined stream. SketchRAG relies on this when two
memory cells are merged.</p>
<p><b>Why it matters for SketchRAG.</b> The guarantee tells us when a count is <i>meaningful</i>. If the sketch says
a rare word appears 40 times in a cell, but the error bound &epsilon;N is 53, that 40 could be pure collision noise:
the sketch <b>cannot vouch</b> for it. If the estimate is clearly above &epsilon;N, the word really is in that cell.
This is the signal our gate uses (Section 5.4).</p>

<h3>3.2 Misra&ndash;Gries: finding the most frequent items</h3>
<p><b>Problem.</b> Find the <b>heavy hitters</b>, the items that appear most often, using only k counters.</p>
<p><b>Algorithm.</b> Keep at most k (item, count) pairs. For each new item:</p>
<ol><li>if it already has a counter, add 1 to it;</li>
<li>else, if fewer than k counters are in use, start a new counter at 1;</li>
<li>else, subtract 1 from <i>every</i> counter and delete counters that reach 0. The new item is not stored.</li></ol>
<div class="eg"><b>Worked example</b> (k = 2). Stream: a b a c a b d a (a appears 4 times out of 8).
<table class="small"><tr><th>item</th><th>a</th><th>b</th><th>a</th><th>c</th><th>a</th><th>b</th><th>d</th><th>a</th></tr>
<tr><td>counters after</td><td>a1</td><td>a1 b1</td><td>a2 b1</td><td>a1 <span class="dim">(b removed)</span></td><td>a2</td>
<td>a2 b1</td><td>a1 <span class="dim">(b removed)</span></td><td>a2</td></tr></table>
Final answer: <b>a</b>, with count 2 (true count 4).</div>
<p><b>Guarantee.</b> For every item, true(x) &minus; N/(k+1) &le; estimate(x) &le; true(x). Above, N/(k+1) = 8/3 &asymp; 2.7,
and indeed 4 &minus; 2 = 2 &le; 2.7. Any item that makes up more than 1/(k+1) of the stream is guaranteed to
survive. SketchRAG keeps a global Misra&ndash;Gries table with k = 256 over content words. Its top entries are
the <b>topics</b> of the stream, and they are shown to the LLM as a one-line &ldquo;what is this stream about&rdquo;
header.</p>

<h3>3.3 Reservoir sampling: a fair sample of an endless stream</h3>
<p><b>Problem.</b> Keep a uniformly random sample of R items from a stream whose length you don't know.</p>
<p><b>Algorithm R.</b> Keep the first R items. When item number t arrives (t &gt; R), pick a random number j between 1
and t. If j &le; R, the new item replaces slot j; otherwise it is ignored.</p>
<p><b>Why it is fair.</b> Item t gets in with probability R/t. Later items push it out a little at a time, and the
two effects cancel exactly. At any moment, every item seen so far has the same chance R/t of being in the sample.
We implemented it, with a size-weighted merge, as the &ldquo;bounded pointers&rdquo; option of SketchRAG: each
cell keeps at most R = 16 pointers to raw chunks instead of all of them. The ablation in Section 8.5 shows this
costs too much accuracy, so it is <b>off</b> in our final configuration.</p>

<h3>3.4 Streaming clustering: grouping similar chunks into cells</h3>
<p><b>Clustering</b> means grouping similar things. Here the things are chunk embeddings, and similar means a high
cosine similarity. Each group is a <b>cell</b>.</p>
<p><b>Clustering features (from the BIRCH algorithm).</b> A cell does not store its members' vectors. It stores only
their <b>sum</b> S and their <b>count</b> n, which is enough to know:</p>
<ul><li>the <b>centroid</b> (centre) &mu; = S / n;</li>
<li>the <b>radius</b>, how spread out the members are. Because our vectors have length 1, the average squared
distance from the centre is simply 1 &minus; ||&mu;||&sup2;. A tight cell has a centroid of length close to 1 and a
radius near 0.</li></ul>
<p>Adding a chunk is a cheap update (S += x, n += 1), and so is merging two cells (add the sums and counts).</p>
<p><b>Keeping the number of cells bounded.</b> Each new chunk joins the most similar cell if their cosine similarity is
at least 0.90; otherwise it opens a new cell. If that pushes the number of cells over the budget
(<code>max_cells</code>, 64 by default), the two cells that are cheapest to merge are merged. &ldquo;Cheapest&rdquo;
uses <b>Ward's criterion</b>:</p>
<div class="formula">cost(a, b) = n<sub>a</sub>&middot;n<sub>b</sub> / (n<sub>a</sub> + n<sub>b</sub>) &middot; ||&mu;<sub>a</sub> &minus; &mu;<sub>b</sub>||&sup2;</div>
<p>This is how much the total spread of the clustering grows if we merge them. Two small cells with nearby centres
cost little; two big cells far apart cost a lot. This &ldquo;merge-and-reduce&rdquo; step is what keeps memory
fixed: the cell count can never exceed the budget, however long the stream gets.</p>

<h3>3.5 Streaming IDF</h3>
<p>IDF (Section 2.2) needs df(word), the number of chunks containing each word, which is again an unbounded set of
counters. We keep it in one more global Count-Min Sketch: each chunk adds 1 for every distinct word it contains.
That gives an approximate, fixed-memory IDF that updates as the stream flows, and SketchRAG uses it everywhere it
needs to know whether a word is rare.</p>
"""

# ---------------------------------------------------------------- 4. related work
RELATED = """
<p>Our proposal named three papers to build on and compare with. Here is what each does, in simple terms.</p>
<ul>
<li><b>EM-LLM (ICLR 2025): episodic memory inside the model.</b> Instead of retrieving text, EM-LLM works on the
model's internal memory (the &ldquo;key/value cache&rdquo;). It reads a long input once, splits it into
<b>events</b> wherever the model is surprised (a sudden jump in how unexpected the next token is), and stores
each event's internal state. When answering, it brings back the most relevant events plus their neighbours in
time (<b>contiguity</b>, the way remembering one moment of a day helps you recall what came right after). It
works well, but needs the model's internals, a big GPU, and memory that grows with the input. We ran
EM-LLM's <b>official code</b> to compare against.</li>
<li><b>AMBER (ACL Findings 2025): memory built at question time by agents.</b> Several LLM roles cooperate. One
retrieves, one keeps a running <b>memory note</b> of useful facts, one decides whether to search again, and one
filters out irrelevant sentences. It is accurate but makes several LLM calls per question. We built a
lightweight re-implementation, <b>AMBER-lite</b>.</li>
<li><b>SARA (ACL 2026): compress after retrieving.</b> Retrieve many chunks, then compress them down to a fixed
budget so the model sees more evidence in fewer tokens. SARA uses learned compression vectors. Our
<b>SARA-lite</b> uses sentence selection instead.</li>
</ul>
<p><b>Where SketchRAG differs.</b> SARA compresses <i>after</i> a question arrives; AMBER builds its memory
<i>during</i> each question. SketchRAG maintains its compressed summary <b>continuously, before any question</b>,
with a fixed size, using classical streaming sketches. It then uses the sketch's own <b>error guarantee</b> to
decide when the summary cannot be trusted. EM-LLM's event segmentation and contiguity inspired our cells and
our temporal-neighbour rule.</p>
"""

# ---------------------------------------------------------------- 5. design
DESIGN = """
<h3>5.1 The big picture</h3>
<p>SketchRAG splits memory into two parts, like a person with a notebook of summaries and a filing cabinet of
original papers:</p>
<ul>
<li><b>Hot memory: the notebook.</b> Small, fixed size, searched for every question. It holds the cells, their
short digests, their Count-Min sketches, a few example vectors per cell, and the global topic and IDF
sketches.</li>
<li><b>Cold storage: the filing cabinet.</b> Every original chunk, written to disk in arrival order. It is
<b>never searched</b>. It is only opened at a known location (&ldquo;cell 12's chunks&rdquo;) when the gate
says the notebook is not enough.</li>
</ul>
{diagram}

<h3>5.2 What happens when a chunk arrives (ingest)</h3>
<ol>
<li><b>Chunk and embed.</b> The text is cut into ~150-word chunks. Each is turned into a 384-number vector once.</li>
<li><b>Store the original.</b> The raw chunk is appended to cold storage.</li>
<li><b>Assign to a cell.</b> Join the most similar cell (cosine &ge; 0.90, with a small bonus for the most recent cell,
because neighbouring text is usually on the same topic), or open a new cell.</li>
<li><b>Update the sketches.</b> Add the chunk's content words to that cell's Count-Min Sketch, to the global IDF
sketch and to the global Misra&ndash;Gries topic table.</li>
<li><b>Update the digest.</b> Each cell keeps an <b>extractive digest</b> of at most 120 words: real sentences copied
from its chunks (no LLM is used, so ingest is cheap). A sentence's priority is the average IDF of its words
(rare, specific words = more informative) times how typical its chunk is of the cell.</li>
<li><b>Update exemplars.</b> Each cell keeps up to 8 diverse member vectors. When full, the most redundant one is
dropped. They help match questions about any part of a cell, not just its centre.</li>
<li><b>Enforce the budget.</b> If there are now more cells than <code>max_cells</code>, merge the cheapest pair
(Ward). Their sketches, digests and pointer lists are combined.</li>
</ol>
<p>No step depends on the stream's length, so each chunk costs about the same no matter how much came before.</p>

<h3>5.3 What happens when a question arrives</h3>
<ol>
<li><b>Rank cells.</b> Embed the question. A cell's score is its best cosine similarity across its centroid and
exemplars. Take the top 4 cells, plus the time-neighbour of the best one (the EM-LLM contiguity idea).</li>
<li><b>Run the gate</b> (next section) on the retrieved cells.</li>
<li><b>Fetch raw evidence</b> for every escalated cell: open its chunks in cold storage, rerank them against the
question, and keep the best 2. They <b>replace</b> that cell's digest.</li>
<li><b>Build the prompt</b>: topic line + digests of trusted cells + raw chunks of escalated cells + question.
One LLM call produces the answer.</li>
</ol>

<h3>5.4 The error-bound gate: our main idea</h3>
<p>The gate looks only at the <b>rare words</b> of the question (streaming IDF &ge; 2.5), because those are the facts a
summary is likely to lose: names, numbers, keys, IDs. For each retrieved cell it asks three questions. If any
answer is &ldquo;yes&rdquo;, the cell is <b>escalated</b> to raw evidence.</p>
<table><tr><th>Reason</th><th>Plain-English meaning</th></tr>
<tr><td><b>Lossy</b></td><td>The cell's Count-Min Sketch says the rare word really occurs in this cell (above its noise
level), but the digest does not contain it, or no single digest sentence holds the rare words together. The
summary dropped the fact.</td></tr>
<tr><td><b>Loose</b></td><td>The cell is spread out (radius &gt; 0.75). Its members are about different things, so one
120-word digest cannot represent them faithfully.</td></tr>
<tr><td><b>Unresolved</b></td><td>No retrieved cell has the rare word clearly above the guaranteed error bound &epsilon;N.
The sketch cannot vouch for <i>any</i> location, so the top 2 cells are read raw to be safe.</td></tr></table>
<p>If the question has no rare words (e.g. &ldquo;Summarise the meeting&rdquo;), the digests are used directly. This
check is <b>deterministic</b>: no LLM is involved, it costs microseconds, and it is grounded in a mathematical
guarantee rather than in the model's opinion of itself.</p>
<div class="eg"><b>Example.</b> A KV-retrieval question asks for the value of the key
<code>a3f9e2...</code>. That key is a unique rare token. The digests are made of a few &ldquo;typical&rdquo;
sentences and almost never contain this exact key, so the lossy (or unresolved) rule fires, the raw chunk
holding the key is fetched, and the model can copy the value exactly. A GovReport question (&ldquo;summarise the
report&rdquo;) has no rare words, so the gate never fires and the cheap digests are used.</div>

<h3>5.5 The five answering policies we compared</h3>
<ul>
<li><b>sketch</b>: digests only, never read raw text. Truly bounded memory, end to end.</li>
<li><b>raw</b>: always read raw text for every retrieved cell (most evidence, most tokens).</li>
<li><b>gate</b>: the deterministic error-bound gate above. <b>Our main system.</b></li>
<li><b>agent</b>: the LLM sees the digests and decides itself whether to call
<code>fetch_raw(cell_id)</code> or <code>search_memory(query)</code> tools (up to 3 rounds).</li>
<li><b>hybrid</b>: the gate's context plus the same tools.</li>
</ul>
"""

DIAGRAM = """
<svg viewBox="0 0 700 250" width="100%" style="max-height:72mm;font-family:Segoe UI,Arial;font-size:11px">
<defs><marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#444"/></marker></defs>
<rect x="5" y="95" width="105" height="50" rx="6" fill="#e8eef9" stroke="#557"/><text x="57" y="117" text-anchor="middle">Text stream</text><text x="57" y="132" text-anchor="middle">(chunks)</text>
<rect x="150" y="95" width="105" height="50" rx="6" fill="#e8eef9" stroke="#557"/><text x="202" y="117" text-anchor="middle">Embed once</text><text x="202" y="132" text-anchor="middle">(bge-small)</text>
<line x1="110" y1="120" x2="148" y2="120" stroke="#444" marker-end="url(#ar)"/>
<rect x="300" y="10" width="230" height="120" rx="8" fill="#eaf6ee" stroke="#486"/>
<text x="415" y="28" text-anchor="middle" font-weight="bold">HOT MEMORY (fixed size)</text>
<text x="312" y="48">&#8226; &#8804; 64 cells: centroid, radius</text><text x="312" y="64">&#8226; per-cell Count-Min Sketch</text>
<text x="312" y="80">&#8226; per-cell digest (&#8804; 120 words)</text><text x="312" y="96">&#8226; 8 exemplar vectors per cell</text>
<text x="312" y="112">&#8226; global IDF sketch + top-256 topics</text>
<rect x="300" y="160" width="230" height="60" rx="8" fill="#f6efe4" stroke="#a84"/>
<text x="415" y="180" text-anchor="middle" font-weight="bold">COLD STORAGE (disk, grows)</text>
<text x="415" y="198" text-anchor="middle">raw chunks + pointers per cell</text><text x="415" y="212" text-anchor="middle">never searched</text>
<line x1="255" y1="115" x2="298" y2="80" stroke="#444" marker-end="url(#ar)"/>
<line x1="255" y1="125" x2="298" y2="185" stroke="#444" marker-end="url(#ar)"/>
<rect x="570" y="10" width="120" height="50" rx="6" fill="#fdf3d6" stroke="#a80"/><text x="630" y="32" text-anchor="middle">Question</text><text x="630" y="47" text-anchor="middle">rank cells</text>
<rect x="570" y="85" width="120" height="50" rx="6" fill="#fde2e2" stroke="#a44"/><text x="630" y="107" text-anchor="middle" font-weight="bold">Error-bound gate</text><text x="630" y="122" text-anchor="middle">trust or escalate?</text>
<rect x="570" y="180" width="120" height="50" rx="6" fill="#e8eef9" stroke="#557"/><text x="630" y="202" text-anchor="middle">LLM answers</text><text x="630" y="217" text-anchor="middle">(1 call)</text>
<line x1="630" y1="60" x2="630" y2="83" stroke="#444" marker-end="url(#ar)"/>
<line x1="530" y1="60" x2="568" y2="40" stroke="#486" marker-end="url(#ar)"/><text x="549" y="72" fill="#486" text-anchor="middle">search</text>
<line x1="630" y1="135" x2="630" y2="178" stroke="#444" marker-end="url(#ar)"/><text x="624" y="160" fill="#444" text-anchor="end">trusted: digest</text>
<path d="M570,120 C540,140 545,175 532,185" fill="none" stroke="#a44" stroke-dasharray="4,3" marker-end="url(#ar)"/>
<text x="455" y="150" fill="#a44">escalated: fetch raw</text>
<line x1="530" y1="200" x2="568" y2="205" stroke="#a84" marker-end="url(#ar)"/>
</svg>
<p class="cap">Figure 1. SketchRAG architecture. Only the green box is searched; the brown box is read only at a known
location when the gate escalates a cell.</p>
"""

# ---------------------------------------------------------------- 6. implementation
IMPLEMENTATION = """
<h3>6.1 Code</h3>
<p>Everything is written in Python in a small research codebase (<code>sketchrag/</code>):</p>
<table><tr><th>File / folder</th><th>What it does</th></tr>
<tr><td><code>sketches/countmin.py</code>, <code>misra_gries.py</code>, <code>reservoir.py</code></td><td>The three
streaming sketches, hand-written in NumPy with their guarantees in the docstrings.</td></tr>
<tr><td><code>memory.py</code></td><td><code>SketchMemory</code>: cells, digests, exemplars, merging, cold storage and
memory accounting.</td></tr>
<tr><td><code>retrieve.py</code></td><td>Cell ranking, the error-bound gate, raw fetch and context building.</td></tr>
<tr><td><code>agent.py</code></td><td>The tool-calling loop for the <i>agent</i> and <i>hybrid</i> policies.</td></tr>
<tr><td><code>systems.py</code>, <code>baselines/</code></td><td>All compared systems behind one interface (full context,
BM25, dense, SARA-lite, AMBER-lite).</td></tr>
<tr><td><code>llm.py</code></td><td>The Ollama client: greedy decoding, token and time logging.</td></tr>
<tr><td><code>eval/metrics.py</code></td><td>LongBench's official scoring functions (F1, ROUGE-L, retrieval score).</td></tr>
<tr><td><code>scripts/</code>, <code>kaggle/</code></td><td>Run scripts, figure scripts and the Kaggle notebooks that ran the
experiments on GPUs.</td></tr>
<tr><td><code>tests/</code></td><td>Unit tests: the Count-Min bound holds on synthetic streams, Misra&ndash;Gries keeps
its guarantee, reservoir samples are uniform, and the cell budget is never exceeded. All 12 pass.</td></tr></table>

<h3>6.2 Tools and hardware</h3>
<ul>
<li><b>LLM:</b> <code>qwen2.5:7b-instruct</code> (7 billion parameters) served by <b>Ollama</b>, a program that runs
open models locally. Every system uses the same model, the same prompt template and greedy decoding (always pick
the most likely next token), so the only difference between systems is the context they give the model.</li>
<li><b>Embeddings:</b> <code>bge-small-en-v1.5</code>, computed once per chunk and cached.</li>
<li><b>Hardware:</b> no local GPU was available, so all LLM runs used free <b>Kaggle</b> notebooks with NVIDIA T4
GPUs. Tuning and retrieval-only checks ran on a laptop CPU.</li>
<li><b>Comparison model for EM-LLM:</b> <code>Phi-3.5-mini</code>, the model EM-LLM's official configuration
supports.</li>
</ul>

<h3>6.3 Engineering problems we hit (and how we solved them)</h3>
<ul>
<li><b>Blocked libraries.</b> Windows policy on our machine blocked SciPy and scikit-learn, so all sketches and
clustering were written from scratch in NumPy. This was also a good way to learn them.</li>
<li><b>Run time.</b> The full set of experiments was estimated at about 11 GPU-hours, longer than one Kaggle session.
We split it into two notebooks (main runs, then ablations and sweeps) that ran one after the other and finished
in 6.4 h and 1.3 h. The runner can resume, so a crash never loses finished questions.</li>
<li><b>EM-LLM memory limits.</b> EM-LLM's official code had to be adapted to fit 2&times;T4 GPUs and 29 GB of host
RAM. It was offloaded to disk, and we fixed a key-parsing assumption in its KV-retrieval loader. Even so, its
&infin;-Bench runs (~128K-token inputs) and NarrativeQA were killed for running out of memory, so we report
EM-LLM on 6 of 9 tasks.</li>
<li><b>Honest tuning.</b> Every setting (thresholds, cell count, weights) was chosen on a separate <b>dev split</b>.
The test questions were only used once, for the final numbers.</li>
</ul>
"""

# ---------------------------------------------------------------- 7. setup
SETUP = """
<h3>7.1 Datasets</h3>
<p>We used two standard long-text benchmarks. Each task tests a different skill:</p>
<table><tr><th>Task</th><th>What it tests</th></tr>
<tr><td>NarrativeQA</td><td>Questions about a whole story or film script.</td></tr>
<tr><td>Qasper</td><td>Questions about a scientific paper.</td></tr>
<tr><td>MultiFieldQA-en</td><td>Questions about documents from many fields (legal, government, science...).</td></tr>
<tr><td>HotpotQA, 2WikiMQA, MuSiQue</td><td><b>Multi-hop</b> questions: the answer needs two or more facts from different
places (&ldquo;Where was the director of film X born?&rdquo;).</td></tr>
<tr><td>GovReport, QMSum</td><td><b>Summaries</b> of a long government report or meeting transcript (global questions).</td></tr>
<tr><td>PassageRetrieval-en</td><td>Given a summary, name which of 30 numbered paragraphs it came from.</td></tr>
<tr><td>&infin;-Bench PassKey</td><td>Find a number hidden in a huge amount of repeated filler text (a needle in a haystack).</td></tr>
<tr><td>&infin;-Bench KV retrieval</td><td>A long list of random key&ndash;value ID pairs; return the value of one key. Pure
exact-string lookup: the hardest case for any summary.</td></tr>
<tr><td>&infin;-Bench En.QA</td><td>Questions about whole novels (~190K tokens).</td></tr></table>
<p><b>Test split:</b> 30 questions per task, chosen at random with a fixed seed (0), and 20 for En.QA: 350 questions in total.
<b>Dev split:</b> a separate set of 10 questions per task (seed 1000), used for all tuning. Every document is fed to
every system as a stream of chunks.</p>
<p><b>Long-stream stress test:</b> all English LongBench QA and summary documents (1,115 documents, 8.5 million words,
~11 million tokens) concatenated into one stream in a fixed shuffled order, with 100 probe questions whose
source documents are spread evenly through the stream.</p>

<h3>7.2 Metrics</h3>
<ul>
<li><b>Token F1</b> (QA tasks). Compare the words of the predicted answer with the correct answer.
<i>Precision</i> = what fraction of predicted words are correct; <i>recall</i> = what fraction of the correct
words were predicted; <b>F1</b> = 2&middot;P&middot;R / (P + R).<br>
<span class="dim">Example: correct answer &ldquo;Paris, France&rdquo;, prediction &ldquo;Paris&rdquo;. P = 1/1, R = 1/2,
F1 = 2&middot;1&middot;0.5 / 1.5 = 0.67.</span></li>
<li><b>ROUGE-L</b> (summaries). How long the <i>longest common subsequence</i> of words between the generated and
reference summaries is, turned into an F-score.</li>
<li><b>Exact containment</b> (PassKey, KV, PassageRetrieval): 1 if the right number, key or paragraph label is in the
answer, else 0.</li>
<li>All scores are shown &times;100 (0 = always wrong, 100 = perfect). <b>Avg</b> is the average over tasks.</li>
<li><b>Answer recall</b> (retrieval quality): 1 if a correct answer string appears anywhere in the context given to the
LLM. It shows whether retrieval found the evidence at all. It is a proxy, since LongBench has no marked evidence spans.</li>
<li><b>Cost:</b> LLM context tokens per question, LLM calls, LLM seconds, and <b>hot memory</b> (bytes of searchable
index).</li>
<li><b>Fallback rate:</b> the fraction of questions where the gate read raw evidence.</li>
</ul>

<h3>7.3 Systems compared</h3>
<ul>
<li><b>Full context:</b> the whole document (cut in the middle if longer than the window). The &ldquo;read
everything&rdquo; option.</li>
<li><b>BM25 RAG:</b> top-5 chunks by word matching.</li>
<li><b>Dense RAG:</b> top-5 chunks by embedding similarity (FAISS).</li>
<li><b>SARA-lite:</b> top-20 dense chunks, compressed to 600 words by sentence selection.</li>
<li><b>AMBER-lite:</b> up to 3 rounds of retrieve, update memory note, decide whether to continue, filter sentences.</li>
<li><b>EM-LLM:</b> the official code with Phi-3.5-mini, on a subset of the same questions.</li>
<li><b>SketchRAG</b> with the five policies (sketch, raw, gate, agent, hybrid).</li>
</ul>
"""

FUTURE = """
<ul>
<li><b>Better keys at stream scale.</b> With thousands of unrelated documents in 64&ndash;256 cells, the cell centroid is
too blunt. A two-level scheme (cells of cells), or keeping a small sketch-based lexical index per cell, could
restore recall.</li>
<li><b>LLM-written digests.</b> Our digests copy sentences. An occasional LLM-written summary per cell might hold
more facts in the same 120 words.</li>
<li><b>Learned gate threshold.</b> The rare-word cut-off (IDF &ge; 2.5) and radius limit (0.75) were tuned by hand on
the dev split. They could be learned.</li>
<li><b>Bigger models and more questions.</b> 30 questions per task and one 7B model leave about &plusmn;2 points of
noise. Larger runs would separate close systems.</li>
<li><b>Bounding cold storage.</b> Cold storage still grows. Smarter pointer sampling than a plain reservoir (which hurt
accuracy here) is needed for a fully bounded system.</li>
</ul>
"""

GLOSSARY = [
    ("Ablation", "Removing or changing one part of a system to see how much that part matters."),
    ("Agent", "An LLM that can call tools and decide its next step, not just answer once."),
    ("Answer recall", "Whether a correct answer string appears in the context given to the LLM."),
    ("BM25", "A classic word-matching ranking formula that rewards rare shared words."),
    ("Cell", "A group of similar chunks in SketchRAG's hot memory, with a digest and a sketch."),
    ("Centroid", "The average (centre) of a group of vectors."),
    ("Chunk", "A small piece of a document (~150 words here)."),
    ("Cold storage", "Disk storage of all raw chunks. Never searched; read only on fallback."),
    ("Context window", "The maximum number of tokens an LLM can read at once."),
    ("Cosine similarity", "How closely two vectors point in the same direction (1 = same)."),
    ("Count-Min Sketch", "A fixed-size table of counters that estimates item counts, never under-counting."),
    ("Digest", "A short extractive summary (real sentences, at most 120 words) kept for each cell."),
    ("Embedding", "A list of numbers representing the meaning of a text."),
    ("Escalate / fallback", "Replacing a cell's digest with its raw chunks from cold storage."),
    ("Extractive", "Built by copying existing sentences, not by writing new ones."),
    ("Gate", "SketchRAG's rule that decides, per cell, whether the summary can be trusted."),
    ("Hash function", "A fixed rule that maps any input to a number in a range, looking random."),
    ("Heavy hitter", "An item that appears very often in a stream."),
    ("Hot memory", "The small, fixed-size memory that is searched for each question."),
    ("IDF", "Inverse document frequency: how rare (and so how informative) a word is."),
    ("LLM", "Large language model: a neural network that reads and writes text."),
    ("Misra-Gries", "A streaming algorithm that finds frequent items using k counters."),
    ("Multi-hop question", "A question that needs facts from two or more places combined."),
    ("Ollama", "Software that runs open LLMs locally behind a simple API."),
    ("RAG", "Retrieval-augmented generation: find relevant text first, then let the LLM answer from it."),
    ("Reservoir sampling", "Keeping a uniform random sample of fixed size from a stream of unknown length."),
    ("Sketch", "A small data structure that summarises a stream with a guaranteed error."),
    ("Token", "The unit of text an LLM reads; roughly 3/4 of a word."),
    ("Ward's criterion", "Merge cost = how much merging two clusters increases total spread."),
]

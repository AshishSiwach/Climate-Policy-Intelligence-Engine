# CPIE — Climate Policy Intelligence Engine

AI Engineering Buildcamp Capstone | Ashish Siwach

---

## Project Summary

Domain-aware RAG system that ingests UK and global climate policy PDFs and returns
structured analyst briefs for climate finance and policy researchers.

**The problem:** Analysts miss or delay acting on regulatory signals buried in hundreds
of pages of dense documentation from Ofgem, FCA, DESNZ, IPCC, IEA.

**The output:** Structured JSON brief — direct answer, grounding citations,
confidence level, flagged contradictions.

---

## Stack

| Component | Decision |
|---|---|
| Dependency management | `uv` |
| PDF extraction | PyMuPDF (fitz) |
| BM25 | `rank-bm25` |
| Dense embeddings | TBD — all-MiniLM-L6-v2 vs BAAI/bge-base-en-v1.5 or nomic-embed-text. Lock after Week 2 notebook validation. |
| Fusion | RRF — k value TBD. Test k=10, 30, 60 in notebook. Lock before building hybrid retriever. |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`. Lazy loading. Separate latency logging. Top-20 candidates. |
| Vector store | Chroma (v1). FAISS / Pinecone / Qdrant / pgvector are upgrade paths only. |
| LLM synthesis | TBD — Haiku 3.5 vs GPT-4o mini. Week 2 comparison on quality, latency, cost. Winner in model_selection.md. |
| Output schema | Pydantic: `answer`, `citations[]`, `confidence` (0–1), `contradictions[]` |
| Verification | Three-check loop: relevance → confidence → contradiction |
| Monitoring | JSON lines logging + Streamlit dashboard |
| UI | Streamlit |
| Containerisation | Docker + docker-compose |
| Build tooling | Makefile targets: install, run, test, eval, build, docker-build, docker-run |

---

## Corpus

- **12 PDFs** from Ofgem, FCA, DESNZ, IPCC, IEA
- All documents audited before ingestion (see Week 2 tasks)
- Estimated corpus size: 800–2500 chunks

### Chunking strategy (document-type-aware)

| Document type | Strategy |
|---|---|
| Ofgem consultations | Chunk by numbered paragraph (3.14, 3.15 etc.) — canonical citation unit |
| FCA papers | Chunk by section heading. Footnotes attached to paragraph above, not chunked independently. |
| DESNZ strategy docs | Chunk by page with 80-token overlap. Headers preserved as metadata. |
| IPCC / IEA reports | Chunk by section. Executive summary as separate high-priority chunk. |

**Chunk size:** Start exploration at 400 tokens / 80-token overlap.
Hard ceiling: 512 tokens — reranker degrades above this.
Lock after notebook validation. Do not build embedder before this is confirmed.

### Metadata schema (per chunk)

```
doc_id, institution, doc_type, jurisdiction, publication_date,
page_number, chunk_index, token_count
```

---

## Output Schema

```json
{
  "answer": "Direct response to the query",
  "citations": [
    {
      "doc_id": "OFGEM_2024_CONSULTATION_RETAILMARKET",
      "passage": "Relevant excerpt...",
      "page": 14
    }
  ],
  "confidence": 0.82,
  "contradictions": [
    { "doc_a": "...", "doc_b": "...", "summary": "..." }
  ]
}
```

---

## Evaluation

- **Ground truth dataset:** 35–50 hand-crafted QA pairs across 12 documents
- **Query types:** 3–4 factual per document + 5–8 negatives (out-of-corpus) + 3–5 cross-document synthesis
- **Metric:** LLM-as-judge score (1–5) against reference answers
- **Critical rule:** Write all QA pairs BEFORE running the system on them

---

## Sprint Timeline

| Week | Dates | Key tasks | Course deadline |
|---|---|---|---|
| Week 2 | Mon 18 – Fri 23 May | 1→ scaffold, 2→ PDF audit, 3→ lock open decisions, 4→ retrieval pipeline, 5→ model comparison | Project Card — Fri 23 May |
| Week 3 | Mon 26 – Fri 30 May | E2E pipeline, synthesis layer, CLI demo, logging active | E2E version — Fri 30 May |
| Week 4 | Mon 2 – Fri 6 Jun | Unit tests, LLM-as-judge tests, ground truth QA dataset | — |
| Week 5 | Mon 9 – Fri 13 Jun | Eval runner, monitoring dashboard, user feedback signal | Tests + Monitoring + Eval — Fri 13 Jun |
| Week 6 | Mon 16 – Fri 20 Jun | Docker, README, deploy, walkthrough | Deployed demo — Fri 20 Jun |

**Time commitment:** 2 hrs/weekday, 3–4 hrs/weekend (~16–18 hrs/week)

---

## Week 2 Task Sequence (strict order — do not skip ahead)

**Step 1 — Project scaffold** *(do this first, before any notebooks or code)*
- *Current state: repo has CLAUDE.md and data/raw/ with 12 PDFs. Nothing else.*
- *Target state: full folder structure in place, all config files created, first git commit done.*
- `git init`, create full folder structure from Project Structure section below
- Create all directories including `data/raw/`, `data/processed/`, `data/eval/results/`, `notebooks/`, `logs/` — add `.gitkeep` to each empty folder so git tracks them
- `pyproject.toml` with initial dependencies: pymupdf, rank-bm25, sentence-transformers, chromadb, anthropic, openai, streamlit, pydantic, pytest, jupyter
- `configs/config.yaml` with placeholders for all TBD values
- `Makefile` with targets: install, run, test, lint
- `.gitignore` covering: data/raw/, data/processed/, .env, __pycache__, .ipynb_checkpoints, logs/
- `.env.example` with ANTHROPIC_API_KEY and OPENAI_API_KEY placeholders
- First commit: `chore: initialise project scaffold`

**Step 2 — Add PDFs to data/raw/** *(manually — do not automate this)*
- *Current state: data/raw/ folder exists but is empty.*
- *Target state: all 12 PDFs present in data/raw/, confirmed before audit starts.*
- Copy all 12 PDFs into `data/raw/` — this folder exists from Step 1 but is gitignored
- Confirm all 12 files are present before running the audit
- No commit needed — data/raw/ is gitignored

**Step 3 — PDF quality audit** *(before any ingestion code)*
- *Current state: 12 PDFs in data/raw/, no ingestion code written.*
- *Target state: audit table complete, every document has a confirmed action, open decisions locked.*
- Confirm all 12 PDFs are in data/raw/
- Run `notebooks/pdf_quality_audit.ipynb` — 6 checks per document:

  **Check 1 — Extractability:** Can text be extracted? Flag any document where >10% of pages return empty strings — likely scanned, needs OCR fallback (pytesseract) or drop from v1.

  **Check 2 — Layout quality:** Spot-check 3 pages per document. Does extracted text read coherently? Are footnotes or headers bleeding into body paragraphs?

  **Check 3 — Structural markers:** Are numbered paragraphs (Ofgem), section headings (FCA), or chapter titles (DESNZ/IPCC) preserved in extraction? These are chunking anchors — if lost, a different strategy is needed.

  **Check 4 — Table detection:** Do any pages contain tables? Does the extractor produce coherent output or garbage? Tables may need special handling or skipping in v1.

  **Check 5 — Relevance and worth:** (a) Can you write at least 3 decision-framed queries this document alone can answer? (b) Would a climate finance analyst cite this document in their work? (c) Is this the primary source or a summary of a better document you should have instead? (d) Is the content current enough to be actionable — outdated policy positions actively mislead.

  **Check 6 — Embedding model + chunk size validation:** Index 3 representative documents with both all-MiniLM-L6-v2 and BAAI/bge-base-en-v1.5. Run 5 queries (factual, synthesis, negative). Compare retrieval relevance (manual inspection of top-5 chunks), cosine similarity distribution, and indexing time. Also validate chunk size starting at 400 tokens / 80-token overlap — adjust if chunks are too coarse or too fine.

- Produce audit table — one row per document:
  `Document | Extractable | Layout clean | Structural markers | Tables | Relevant | Worth including | Action`
  Actions: Include / Include with fixes / Replace with primary source / Drop from v1
- **No pipeline code until every document has a confirmed action.**
- Third commit: `chore: pdf quality audit complete`

**Step 4 — Lock open decisions** *(before building any pipeline modules)*
- *Current state: audit complete, open decisions still marked TBD in config.yaml.*
- *Target state: embedding model, chunk size, RRF k all locked and written into config.yaml.*
- Embedding model comparison (in audit notebook or separate notebook)
- Chunk size validation at 400/80 starting point
- RRF k value test (k=10, 30, 60)
- Update open decisions checklist below — tick off each one as locked
- Update config.yaml with locked values
- Fourth commit: `chore: lock open decisions — embedding model, chunk size, RRF k`

**Step 5 — Retrieval pipeline**
- *Current state: open decisions locked, no retrieval modules written.*
- *Target state: BM25, dense, RRF, reranker all working and manually validated on 3 queries.*
- Build `src/retrieval/bm25_retriever.py`, `dense_retriever.py`, `hybrid_retriever.py`, `reranker.py`
- Manual validation: run 3 queries, inspect top-5 chunks for relevance
- Fifth commit: `feat: retrieval pipeline`

**Step 6 — Model selection comparison**
- *Current state: retrieval pipeline working, LLM model still TBD.*
- *Target state: winning model locked, results documented in model_selection.md.*
- `notebooks/model_selection.ipynb` — Haiku 3.5 vs GPT-4o mini on 10 queries
- Write model_selection.md with results table and 2–3 sentence reasoning
- Sixth commit: `chore: model selection complete`

**Step 7 — Project Card** *(course deadline Fri 23 May — do in Claude.ai Projects)*
- *Current state: all decisions made, no written deliverable yet.*
- *Target state: one paragraph submitted to course covering problem, user, input, processing, output, success metric.*
- Open Claude.ai Projects, use Project Card prompt from the prompts list
- Submit to course

---

## Week 3 Task Sequence (strict order)

**Step 1 — PDF ingestion pipeline**
- *Current state: PDFs audited, chunking strategy confirmed, no ingestion code.*
- *Target state: all audited documents chunked with metadata, saved to data/processed/ as JSON.*
- Build `src/ingestion/pdf_loader.py` and `src/ingestion/chunker.py`
- Use locked chunk size and document-type-aware strategy from audit
- Validate: chunk count, avg token length, no chunks above 512 tokens
- Commit: `feat: pdf ingestion pipeline`

**Step 2 — Embed and index**
- *Current state: chunks in data/processed/, no vector store.*
- *Target state: Chroma index populated, BM25 index built, both queryable.*
- Build `src/retrieval/dense_retriever.py` using locked embedding model
- Initialise Chroma vector store on all audited documents
- Commit: `feat: chroma index built`

**Step 3 — Wire full retrieval pipeline**
- *Current state: individual retrieval modules exist, not wired together.*
- *Target state: BM25 + dense + RRF + reranker returning ranked chunks end-to-end on a real query.*
- Confirm BM25, dense, RRF (locked k), reranker working end-to-end
- Manual inspection: run 3 real queries, check top-5 chunks are relevant
- Commit: `feat: hybrid retrieval pipeline end-to-end`

**Step 4 — Synthesis layer**
- *Current state: retrieval pipeline working, no LLM synthesis.*
- *Target state: structured JSON brief returned for a real query.*
- Build `src/synthesis/output_schema.py` (Pydantic) and `src/synthesis/synthesiser.py`
- Wire locked LLM model, three-check verification loop
- Commit: `feat: synthesis layer`

**Step 5 — CLI end-to-end demo + logging**
- *Current state: all modules built but not wired into a single runnable pipeline.*
- *Target state: `python main.py "query"` returns a structured JSON brief. Logging active from this point.*
- Build `main.py` — accepts query string, runs full pipeline, prints JSON brief
- Add `src/monitoring/logger.py` — JSON lines, logs every query from day one
- Run 5 real queries manually, inspect output quality
- Commit: `feat: end-to-end CLI pipeline with logging`

**Step 6 — E2E version submission** *(course deadline Fri 30 May)*
- *Current state: pipeline works locally.*
- *Target state: pipeline confirmed running cleanly on a fresh install, submitted to course.*
- Test clean run from scratch: `git clone → uv install → python main.py "query"`
- Submit to course

---

## Week 4 Task Sequence (strict order)

**Step 1 — Unit tests**
- *Current state: pipeline working, no tests.*
- *Target state: unit test suite passing across ingestion, retrieval, synthesis.*
- Build `tests/test_ingestion.py`, `tests/test_retrieval.py`, `tests/test_synthesis.py`
- Cover: chunk bounds, metadata presence, schema validation, retrieval count, out-of-corpus handling
- All tests passing before moving to next step
- Commit: `test: unit tests passing`

**Step 2 — LLM-as-judge test harness**
- *Current state: unit tests passing, no judge harness.*
- *Target state: judge scoring function working, test_judge.py passing.*
- Build `src/evaluation/judge.py` with scoring rubric (1–5)
- Build `tests/test_judge.py`
- Commit: `test: llm-as-judge harness`

**Step 3 — Ground truth QA dataset** *(checkpoint Wed 4 Jun — do in Claude.ai Projects)*
- *Current state: judge harness working, no ground truth data.*
- *Target state: 35–50 QA pairs written from genuine reading, saved to data/eval/ground_truth.json.*
- Write 35–50 QA pairs from genuine reading of source documents
- **CRITICAL: write questions BEFORE running the system on them**
- Include 5–8 negatives and 3–5 cross-document synthesis questions
- Save to `data/eval/ground_truth.json`
- Commit: `data: ground truth dataset complete`

---

## Week 5 Task Sequence (strict order)

**Step 1 — Evaluation runner**
- *Current state: ground truth dataset complete, no eval pipeline.*
- *Target state: eval runner scoring all 35–50 pairs, results saved to data/eval/results/.*
- Build `src/evaluation/eval_runner.py`
- Run full eval on ground truth dataset
- Document results: mean score, score distribution, failure rate by query type
- Commit: `feat: evaluation runner`

**Step 2 — Inspect failures** *(do in Claude.ai Projects)*
- *Current state: eval results available, failures not yet analysed.*
- *Target state: failure patterns identified, issues logged, at least one fix applied.*
- Read every failing query manually
- Identify patterns: retrieval misses, synthesis errors, confidence miscalibration
- Log issues as GitHub issues or failures.md
- Commit: `chore: eval failure analysis`

**Step 3 — Monitoring dashboard**
- *Current state: query logs accumulating in logs/, no dashboard.*
- *Target state: Streamlit dashboard live showing query volume, confidence, latency, feedback.*
- Build `src/monitoring/dashboard.py` — Streamlit app
- Charts: query volume, confidence distribution, latency per stage, top retrieved docs
- User feedback widget in `app.py` — thumbs up / thumbs down per response
- Commit: `feat: monitoring dashboard`

**Step 4 — Tests + Monitoring + Eval submission** *(course deadline Fri 13 Jun)*
- *Current state: all three components built.*
- *Target state: all confirmed working, submitted to course.*
- Confirm tests pass, eval results documented, dashboard running
- Submit to course

---

## Week 6 Task Sequence (strict order)

**Step 1 — Docker + Makefile** *(checkpoint Mon 16 Jun)*
- *Current state: working local pipeline, no containerisation.*
- *Target state: Docker build succeeds, app runs in container, Makefile targets complete.*
- Write `Dockerfile` and `docker-compose.yml`
- Update Makefile with: docker-build, docker-run targets
- Test clean install from scratch inside Docker
- Commit: `chore: docker and makefile complete`

**Step 2 — README** *(do in Claude.ai Projects)*
- *Current state: no README beyond placeholder.*
- *Target state: complete README covering problem, architecture, setup, eval results, design decisions.*
- Problem statement (use Project Card text)
- Architecture diagram (ASCII)
- Setup instructions (3 steps: clone, add PDFs, make run)
- Model selection decision summary
- Eval results table
- Design decisions section
- Known limitations and v2 extensions
- Commit: `docs: readme complete`

**Step 3 — Deploy demo**
- *Current state: app runs locally, not publicly accessible.*
- *Target state: live URL accessible, confirmed working end-to-end in browser.*
- Deploy to Streamlit Cloud, Hugging Face Spaces, or equivalent
- Confirm live URL accessible
- Commit: `chore: deploy demo`

**Step 4 — Walkthrough** *(do in Claude.ai Projects)*
- *Current state: demo deployed, no walkthrough.*
- *Target state: short walkthrough document or screen recording covering the full story.*
- Cover: problem, example query end-to-end, one thing that worked, one thing that was hard, v2 roadmap

**Step 5 — Final submission** *(course deadline Fri 20 Jun)*
- *Current state: everything built and deployed.*
- *Target state: submitted to course with deployed demo, README, and walkthrough.*
- Confirm deployed demo accessible, README complete, walkthrough attached
- Submit to course



1. **Current state → target state.** Every session starts with: where is the project now, what is the target by end of session.
2. **Ship E2E early.** Working simple pipeline by end of Week 3 is the anchor. Tests, monitoring, eval grow on top.
3. **Eval follows business goals.** Metrics are: does the answer correctly reflect the document, does the system know when it can't answer, does the citation support the claim. Not generic AI benchmarks.
4. **Monitoring is part of building.** Logging goes in during Week 3 alongside the E2E pipeline — not bolted on in Week 5.

---

## Locked Decisions — Do Not Revisit Unless Explicitly Asked

- No LangChain. No LlamaIndex. No fine-tuning. No training.
- Fresh codebase — do not inherit QueryLens structure directly, only adapt retrieval modules.
- Chroma is the v1 vector store. Do not suggest FAISS as the primary store.
- One conversation per task. Do not mix concerns across sessions.
- Simple RAG for v1. Agentic RAG (query decomposition, monitor_corpus, assess_materiality) is v2.
- Contradiction detection is experimental in v1 — do not treat as a core feature.

---

## Open Decisions (resolve in Week 2 notebook before writing pipeline code)

- [ ] Embedding model — run comparison, lock winner
- [ ] Chunk size — validate 400/80 starting point, lock before embedder
- [ ] RRF k value — test 10, 30, 60, lock before hybrid retriever
- [ ] LLM synthesis model — Haiku 3.5 vs GPT-4o mini comparison, document in model_selection.md

---

## Prior Project Reference

QueryLens — hybrid retrieval pipeline on MS MARCO (164k passages, general domain).
Retrieval modules (BM25, dense, RRF, reranker) are adapted from there.

**Factual constraints to carry forward:**
- Never cite ~1ms FAISS query time — never measured. Only the ~11-minute FAISS HNSW build time on Colab T4 is real.
- Recall@10 is the QueryLens metric. CPIE's primary metric is LLM-as-judge score on ground truth dataset.
- MS MARCO is general domain. CPIE corpus is domain-specific PDFs — different chunking, different evaluation.

---

## Project Structure

```
cpie/
  CLAUDE.md                      # This file
  configs/
    config.yaml                  # All settings: models, chunking, retrieval params
  data/
    raw/                         # Original PDFs (gitignored)
    processed/                   # Chunks as JSON (gitignored)
    eval/
      ground_truth.json          # Hand-crafted QA pairs
      results/                   # Eval run outputs
  notebooks/
    pdf_quality_audit.ipynb      # Week 2 — audit before any ingestion code
    model_selection.ipynb        # Week 2 — Haiku 3.5 vs GPT-4o mini
  src/
    ingestion/
      pdf_loader.py              # PDF -> chunks with metadata
      chunker.py                 # Document-type-aware chunking
    retrieval/
      bm25_retriever.py
      dense_retriever.py
      hybrid_retriever.py        # RRF fusion
      reranker.py                # Cross-encoder, lazy loading
    synthesis/
      synthesiser.py             # LLM prompt + structured output
      output_schema.py           # Pydantic schema
    evaluation/
      judge.py                   # LLM-as-judge scoring
      eval_runner.py             # Run eval on ground truth dataset
    monitoring/
      logger.py                  # JSON lines query logging
      dashboard.py               # Streamlit monitoring view
  tests/
    test_ingestion.py
    test_retrieval.py
    test_synthesis.py
    test_judge.py
  app.py                         # Streamlit UI entry point
  main.py                        # CLI entry point
  model_selection.md             # Model comparison results and decision
  Makefile
  Dockerfile
  docker-compose.yml
  pyproject.toml
  .env.example
  README.md
```

---

## Tool Routing — Claude Code vs Claude.ai Projects

**Claude Code (this tool — use for everything that touches files or the terminal):**
- Project scaffold, folder structure, config files
- PDF quality audit notebook
- Ingestion, retrieval, synthesis pipeline code
- Unit tests and judge tests
- Evaluation runner
- Monitoring dashboard
- Docker, Makefile, README
- Debugging — paste actual errors here

**Claude.ai Projects (open a separate browser session — use for writing and thinking):**
- Project Card — Week 2 course deadline (Fri 23 May)
- Ground truth QA dataset — writing and validating 35–50 pairs (Week 4)
- model_selection.md write-up — reasoning behind model choice
- Eval failure analysis — interpreting results, deciding what to fix
- Walkthrough document — Week 6 course deadline (Fri 20 Jun)
- Architecture questions, planning, anything open-ended

**Rule of thumb:** If the output is a file of code or data — Claude Code. If the output is a document a human reads or submits — Claude.ai Projects.

---

## How to Work With Me (Claude Code)

- State current state and target state at the start of every session
- Paste actual code and actual error messages — not descriptions of them
- One task per session — do not mix ingestion debugging with synthesis design
- All decisions above are locked unless you explicitly say "I want to revisit X"
- If I suggest LangChain, LlamaIndex, or FAISS as primary store — push back

**First session prompt (copy and paste this tomorrow):**
```
Current state: repo contains CLAUDE.md and data/raw/ with 12 PDFs already copied in.
No other files or folders exist yet.
Target state: full project scaffold in place — all folders created with .gitkeep, 
pyproject.toml, config.yaml, Makefile, .gitignore, .env.example.
First git commit ready.
Let's start with Step 1 from the Week 2 task sequence.
```


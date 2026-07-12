# CPIE — Climate Policy Intelligence Engine

AI Engineering Buildcamp Capstone + LLM Zoomcamp Capstone | Ashish Siwach

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
| Dense embeddings | **LOCKED: BAAI/bge-base-en-v1.5** — beat all-MiniLM-L6-v2 on top-5 relevance and cosine sim distribution (mean 0.543 vs 0.278). 768-dim. device=cuda (RTX 4050). |
| Fusion | **LOCKED: RRF k=60** — Cormack et al. 2009 default. k=10/30/60 indistinguishable on test query (retrievers agreed). Conservative fusion correct for small single-domain corpus. |
| Reranker | **v1: NOT USED** — 3-query ablation showed reranker adds 172ms per query with zero hit-rate improvement over hybrid (3/3 top-1 on both). Code preserved in `src/retrieval/reranker.py` for Week 5 re-evaluation with full ground truth dataset. If Week 5 LLM-as-judge shows meaningful synthesis quality gain, add back. Model when re-enabled: `cross-encoder/ms-marco-MiniLM-L-6-v2`. |
| Vector store | Chroma (v1). FAISS / Pinecone / Qdrant / pgvector are upgrade paths only. |
| LLM synthesis | **LOCKED: GPT-4o mini** — beat Haiku 4.5 on quality (4.0 vs 2.7 avg) and cost (6x cheaper per query). Streaming planned to reduce perceived latency to ~800ms first token. Decision documented in model_selection.md. |
| Output schema | Pydantic: `answer`, `citations[]`, `confidence` (0–1), `contradictions[]` |
| Verification | Three-check loop: relevance → confidence → contradiction |
| Monitoring | JSON lines logging (Week 3, raw data layer) + Logfire (Week 5, observability — course specified) |
| UI | Streamlit |
| Containerisation | Docker + docker-compose |
| Build tooling | Makefile targets: install, run, test, eval, build, docker-build, docker-run |

---

## Corpus

- **12 PDFs** from Ofgem, FCA, DESNZ, IPCC, IEA
- All documents audited before ingestion (see Week 2 tasks)
- Estimated corpus size: 800–2500 chunks

### Chunking strategy

**v1: sliding window across all documents.** Simple, standard baseline.

- 400 tokens per chunk, 80-token overlap (locked)
- 50-token floor, discard fragments below
- 512-token ceiling assert fires after Tier 2 heading injection

**Why sliding window as v1:** Every chunking-strategy claim needs empirical validation. Shipping document-aware chunking without measuring it against a baseline would be "changed stance without evidence." Sliding window is cheap, standard, and the correct scientific baseline.

**Watch during eval failure analysis:** Ofgem chunks may split paragraph citations across boundaries. Retrieval will still find the right chunk; the paragraph number stays inside the chunk text; `page_number` metadata preserves the citation. Not a v1 dealbreaker — flag if it shows up as a failure pattern.

**v2 experiments (measured against v1 in Week 5):**
1. Document-aware chunking — Ofgem by paragraph, CCC/IEA/ESO by section, others sliding window
2. Universal heading injection — inject `[Section: ...]` into every chunk in every doc
3. Chunk size sweep — test 200, 300, 400, 480 against retrieval metrics

Keep whichever wins. See v2 Roadmap for detail.

### Table handling — three tiers (ALL documents, locked in audit)

**Tier 1 — Active stripping before chunking**
Apply document-specific regex to remove layout noise before text reaches the chunker.

| Document | What to strip |
|---|---|
| ESO Beyond 2030 | Interactive PDF nav elements ("Navigation", "Download a pdf", "Text Links", "Return to contents"), duplicated map page headers, fragmented social handles |
| Ofgem Smart Secure | Running header ("Consultation Smart Secure Electricity Systems...NN"), `OFFICIAL OFFICIAL` security stamps |

**Tier 2 — Heading injection on table pages**
For pages detected as table pages (`page.find_tables()` returns results), prepend nearest section heading
(walk back up to 3 pages). Preserves numeric context and makes table chunks retrievable by BM25 + dense.

| Document | Why |
|---|---|
| WEO 2025 (122 real table pages) | Dense numeric cost/demand tables — LCOE by technology, demand by scenario. Analysts will query these figures. |
| Seventh Carbon Budget (28 real table pages) | Sector emissions trajectories, policy cost estimates. Good heading structure in source. |
| CCC Progress 2024 (20 real table pages) | RAG status tables — traffic-light indicators don't extract. Prose surrounds carry signal; heading injection makes page findable. |
| CCC Progress 2025 (13 real table pages) | Same as above. |
| BoE Disclosure 2024 (7 real table pages) | Real financial data — portfolio exposures, carbon intensities, compositions. Analysts will query these. |

**Tier 3 — Keep as-is**
Table content is descriptive prose or benign layout artefacts. No special handling needed.

| Document | Why |
|---|---|
| CBES Results 2021 (11 table pages) | Prose cells — participant lists, loss estimate tables. Reads fine flat. |
| CBES Key Elements (8 table pages) | Prose cells — participant lists, scenario design. |
| Measuring Climate Risks (3 table pages) | Scenario description tables — descriptive text in cells. |
| ZEV Mandate (5 real table pages) | Layout boxes with text repeated across columns. Artefacts merge or fall under min-token floor. |
| BoE Macro Implications (0 table pages) | No tables — no action needed. |

**Note on table page counts:** `page.find_tables()` raw hit counts significantly overcount real tables on
several documents — it misdetects recurring page headers/footers and body prose split into columns as tables.
Verified via multi-sample audit (`notebooks/pdf_quality_audit.ipynb`, Check 4) with a fill-ratio filter
(discards detections where >70% of data cells are empty). Raw vs. real: WEO 2025 338→122, ESO Beyond2030
118→53 (not table-tiered — nav elements already stripped in Tier 1), Ofgem Smart Secure 19→7 (not table-tiered
— chunked by numbered paragraph, false positives don't affect pipeline), Seventh Carbon Budget 47→28, CCC
Progress 2024 24→20, CCC Progress 2025 27→13, BoE Disclosure 2024 9→7, ZEV Mandate 6→5. CBES Results/Key
Elements and Measuring Climate Risks had zero false positives (raw count = real count). Counts above are the
filtered "real" figures — use these for Tier 2 planning, not the raw `find_tables()` count.

**Note on CCC Progress RAG tables:** Traffic-light status indicators (On Track / Insufficient Action symbols) do not
extract as text. This is a known limitation. The prose paragraphs surrounding each table restate the assessment
in text form — these carry the retrieval signal. Log as known limitation in README.

### Metadata schema (per chunk)

```
doc_id, institution, doc_type, jurisdiction, publication_date,
page_number, chunk_index, token_count, chunk_type
```

`chunk_type`: `"prose"` (default) or `"table"` (set when page.find_tables() detected a table).
Used at synthesis time — LLM instructed to extract specific values from table chunks rather than paraphrase.

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
| Week 5 | Mon 9 – Fri 13 Jun | Eval runner, Logfire monitoring, user feedback signal | Tests + Monitoring + Eval — Fri 13 Jun |
| Week 6 | Mon 16 – Fri 20 Jun | Docker, README, deploy, walkthrough | Deployed demo — Fri 20 Jun |
| Week 7+ | Mon 23 Jun – Fri 25 Jul | LLM Zoomcamp gap analysis — add whatever CPIE doesn't already cover | Project Attempt 1 — Mon 28 Jul |

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
- *Current state: PDFs audited, all decisions locked, no ingestion code.*
- *Target state: all 12 audited documents chunked with metadata, saved to data/processed/ as JSON.*

Build `src/ingestion/pdf_loader.py`:
- `load_pdf(path) -> list[dict]` — extract text per page using PyMuPDF `page.get_text("text")`
- `clean_text(text, doc_id) -> str` — apply Tier 1 strip rules for ESO and Ofgem only:
  - ESO: strip nav elements ("Navigation", "Download a pdf", "Text Links", "Return to contents"), duplicated map headers, fragmented social handles
  - Ofgem: strip running header pattern + `OFFICIAL OFFICIAL` stamps
- `detect_table_page(page) -> bool` — use `page.find_tables().tables` (PyMuPDF)
- `inject_heading(doc, page_idx) -> str` — walk back up to 3 pages, find nearest section heading via regex `r'(\d+[\.\d]*\s+[A-Z][^\n]{5,60}|Table \d+[\.\d]*[^\n]{5,60})'`, prepend as `[Section: ...]`
- Apply heading injection to Tier 2 documents: WEO2025, Seventh Carbon Budget, CCC Progress 2024/2025, BoE Disclosure 2024
- Attach metadata per page: `doc_id`, `institution`, `doc_type`, `jurisdiction`, `publication_date`, `page_number`, `chunk_type` ("prose"/"table")

Build `src/ingestion/chunker.py`:
- `chunk_page(text, chunk_size=400, overlap=80) -> list[str]` — tiktoken cl100k_base tokeniser
- Minimum token floor: 50 tokens — discard fragments below this
- Hard ceiling: 512 tokens — assert no chunk exceeds this
- Add `chunk_index` and `token_count` to metadata per chunk
- Save output: `data/processed/<doc_id>.json` — list of chunk dicts

> **Implementation note — 512-token assert placement:** The ceiling assert must fire in `pdf_loader.py` *after* heading injection, not inside `chunk_page()`. The chunker sees raw text (≤400 tokens); heading injection in `pdf_loader.py` prepends `[Section: ...]` (~15–18 tokens) afterwards. Assert on the final combined token count so the guard checks what the reranker actually receives:
> ```python
> final_tokens = len(tokenizer.encode(chunk_with_heading))
> assert final_tokens <= 512, f"Chunk exceeds ceiling after heading injection: {final_tokens}"
> ```
> Risk is low in practice (400 + ~18 = 418, well under 512) but the assert is in the wrong place until this is fixed.

Validate before moving on:
- Total chunk count across all 12 docs (expect 800–2500)
- Avg token length per doc
- Zero chunks above 512 tokens
- Spot-check: 3 chunks from ESO (confirm nav stripped), 3 from Ofgem (confirm OFFICIAL stripped), 3 table chunks from WEO2025 (confirm heading prepended)

- Commit: `feat: pdf ingestion pipeline`

**Step 2 — Embed and index**
- *Current state: chunks in data/processed/, no vector store.*
- *Target state: Chroma index populated, BM25 index built, both queryable.*

Build `src/retrieval/dense_retriever.py`:
- Model: `BAAI/bge-base-en-v1.5`, device=cuda (RTX 4050 confirmed)
- `build_index(chunks) -> None` — encode all chunks, persist to Chroma
- `query(text, top_k=20) -> list[dict]` — encode query, return top-k with scores and metadata
- Chroma collection: `cpie_v1`, persist to `data/processed/chroma_db/`
- Store all metadata fields in Chroma document metadata (enables filtered retrieval later)

BM25 index lives in `src/retrieval/bm25_retriever.py` (built in Week 2 Step 5 — wire here if not done):
- `build_index(chunks) -> BM25Okapi`
- `query(text, top_k=20) -> list[dict]`
- Serialise index to `data/processed/bm25_index.pkl`

- Commit: `feat: chroma index and bm25 index built`

**Step 3 — Wire hybrid retrieval pipeline (no reranker in v1)**
- *Current state: BM25 and dense indices built, not fused.*
- *Target state: BM25 + dense + RRF returning ranked top-5 chunks end-to-end on real queries.*

`src/retrieval/hybrid_retriever.py`:
- RRF fusion with k=60 (locked)
- `retrieve(query, top_k=5) -> list[dict]` — calls BM25 + dense (each top-10), fuses with RRF, returns top-5 for synthesiser

`src/retrieval/reranker.py`:
- Preserved but NOT wired into the v1 pipeline. See ablation reasoning below.

**Retrieval ablation (run in Week 3, on 3 pilot queries):**
Compared 4 configs — BM25 only, Dense only, Hybrid, Full pipeline (hybrid + rerank).
All 4 got 3/3 top-1 hits. Reranker added 172ms per query with zero hit-rate improvement.
Decision: **drop reranker for v1**, revisit in Week 5 with 35–50 ground truth queries + LLM-as-judge.
Hybrid stays because it costs only 14ms over BM25 alone and preserves semantic coverage for queries that don't name the institution. Full ablation script: `scripts/ablation_retrieval.py`.

Manual validation queries (used in `scripts/validate_pipeline_e2e.py`):
1. *"What load control licensing requirements does Ofgem propose?"* — expect Ofgem Smart Secure chunks
2. *"What aggregate losses did UK banks face under the CBES early action scenario?"* — expect CBES Results chunks
3. *"What does the IEA project for peak fossil fuel demand?"* — expect WEO 2025 chunks

- Commit: `feat: hybrid retrieval pipeline end-to-end`

**Step 4 — Synthesis layer**
- *Current state: retrieval pipeline working, LLM model locked from Step 6 (Week 2).*
- *Target state: structured JSON brief returned for a real query.*

`src/synthesis/output_schema.py`:
```python
class Citation(BaseModel):
    doc_id: str
    passage: str
    page: int

class AnalystBrief(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float          # 0.0–1.0
    contradictions: list[dict] # experimental — list of {doc_a, doc_b, summary}
```

`src/synthesis/synthesiser.py`:
- Three-check verification loop: relevance → confidence → contradiction
- `chunk_type: table` handling — instruct LLM to extract specific values, not paraphrase
- Handle out-of-corpus queries explicitly: if no chunk scores above confidence threshold, return `confidence=0.0` with `answer="The corpus does not contain sufficient information to answer this query."`
- Note: contradiction detection is experimental in v1 — implement but do not treat as core feature
- **Confidence must come from the pipeline, not the LLM.** Combine: top RRF score + retrieval score spread (are top chunks clustered or scattered?) + citation count (how many chunks support the answer). Map these signals to 0.0–1.0. Do not ask the LLM to self-assess confidence — it is unreliable.
- **Confidence weights are v1 placeholders (0.5 score / 0.3 spread / 0.2 citation) — not empirically calibrated.** Individual signals (`score_signal`, `spread_signal`, `citation_signal`) are returned by `Synthesiser.synthesise()` and must be logged per query. Week 5 calibrates real weights by fitting logistic regression against LLM-as-judge scores on the ground truth dataset.
- **Citation verification:** after synthesis, check every cited passage against the actual retrieved chunks. If a cited passage does not appear in any retrieved chunk, flag or remove it. Prevents fabricated citations.

- Commit: `feat: synthesis layer`

**Step 5 — CLI end-to-end demo + logging**
- *Current state: all modules built but not wired into a single runnable entry point.*
- *Target state: `python main.py "query"` returns a structured JSON brief. Logging active from this point.*

`main.py`:
- Accept query string as CLI argument
- Run pipeline: clean query → BM25 + dense retrieve → RRF fuse → top-5 → synthesise (no rerank in v1)
- Print JSON brief to stdout

`src/monitoring/logger.py`:
- JSON lines format, one record per query
- Log: `timestamp`, `query`, `retrieved_doc_ids`, `retrieval_scores`, `rrf_scores`, `retrieval_latency_ms`, `synthesis_latency_ms`, `confidence`, `model_used`, `prompt_tokens`, `completion_tokens`, `cost_usd`, `failure_reason`
- Write to `logs/queries.jsonl`
- Note: `rerank_scores` / `rerank_latency_ms` will be added back when reranker is re-enabled.
- Logging goes in NOW — not in Week 5. Every query logged from day one.

Run 5 real queries manually, inspect output quality before committing:
1. Factual — Ofgem licensing timeline
2. Factual — CBES loss estimates
3. Factual — IEA fossil fuel demand peak
4. Cross-document — CCC vs IEA on net zero pathways
5. Out-of-corpus negative — confirm system returns low confidence, not a hallucination

- Commit: `feat: end-to-end CLI pipeline with logging`

**Step 6 — E2E version submission** *(course deadline Fri 30 May)*
- *Current state: pipeline works locally.*
- *Target state: pipeline confirmed running cleanly on a fresh install, submitted to course.*
- Test clean run: `git clone → uv sync → cp .env.example .env → add API key → python main.py "query"`
- Confirm logs/queries.jsonl is being written
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

**Step 3 — Logfire monitoring**
- *Current state: query logs accumulating in logs/queries.jsonl, no observability layer.*
- *Target state: Logfire integrated, failures and unusual outputs visible, user feedback captured.*
- Add `logfire` to pyproject.toml dependencies
- Instrument `src/monitoring/logger.py` to emit to Logfire alongside existing JSONL logging
- Log spans for: retrieval latency, reranker latency, synthesis latency, confidence score per query
- Surface failures — low confidence responses, synthesis errors, out-of-corpus queries
- User feedback widget in `app.py` — thumbs up / thumbs down per response, logged to Logfire
- Commit: `feat: logfire monitoring`

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



## Week 7+ — LLM Zoomcamp Integration (after Jun 20)

**Rule: do not touch this until CPIE Week 6 is fully complete and submitted.**

After the 6-week CPIE build is done, do one gap analysis pass:
- Compare LLM Zoomcamp homework topics against what CPIE already has
- Identify anything missing from the intersection
- Add only what's missing — do not rebuild things that already exist

**Known topics to check at that point:**

| LLM Zoomcamp topic | CPIE coverage | Gap? |
|---|---|---|
| Agentic RAG | Simple RAG only — agentic is v2 | Likely yes — assess HW1 requirements |
| Vector Search | Chroma + BAAI/bge-base-en-v1.5 | No gap |
| Orchestration | main.py full pipeline | No gap |
| Evaluation | LLM-as-judge + eval runner | No gap |
| Monitoring | Logfire | No gap |

**Submission deadline:** Project Attempt 1 — Mon 28 Jul 2026. Project Attempt 2 (buffer) — Tue 11 Aug 2026.

---

1. **Current state → target state.** Every session starts with: where is the project now, what is the target by end of session.
2. **Ship E2E early.** Working simple pipeline by end of Week 3 is the anchor. Tests, monitoring, eval grow on top.
3. **Eval follows business goals.** Metrics are: does the answer correctly reflect the document, does the system know when it can't answer, does the citation support the claim. Not generic AI benchmarks.
4. **Monitoring is part of building.** Logging goes in during Week 3 alongside the E2E pipeline — not bolted on in Week 5.

---

## v2 Roadmap — Do Not Build Until v1 Is Shipped

These are validated improvements deferred deliberately. Add them after Week 6 submission.

| Item | Why deferred | Value |
|---|---|---|
| Contradiction detector redesign | Current `contradictions[]` is LLM self-report — unreliable. v2: cluster by source → summarise each → compare claims → LLM explains differences. Rename to `conflicting_sources`. | High |
| Metadata filtering | Institution/jurisdiction/date stored but unused in retrieval. Add pre-retrieval filter: if query names DESNZ, filter `institution=DESNZ` before BM25+dense. Improves precision significantly. | High |
| Query classification | Classify query type (factual / comparison / summarisation / cross-document / numeric) before retrieval. Choose strategy per type. Lightweight classifier, big retrieval gain. | High |
| Retrieval metrics (Recall@5, MRR, nDCG@10) | Requires labeled relevance judgments per query. Add after ground truth dataset exists. Shows whether retrieval is improving independently of the LLM. | Medium |
| Universal heading injection | v1 injects section headings for CCC/IEA/ESO (section strategy) and Tier 2 table pages only. BoE prose docs and DESNZ have no section context on their chunks. v2: extend heading injection to every chunk in every document, or move to hierarchical trail (`[Chapter 3 > Transport > Cars & vans]`). Run A/B with retrieval metrics to confirm gain before committing. | Medium |
| Reranker re-evaluation | Cross-encoder rerank dropped from v1 pipeline after 3-query ablation showed no hit-rate gain over hybrid (all 3 configs got 3/3 top-1) at cost of 172ms/query. Test unrepresentative — too few queries, all named the institution. Week 5: re-run ablation with 35–50 ground truth queries + LLM-as-judge scoring. If rerank meaningfully improves top-5 relevance or synthesis quality, re-enable in `main.py` (one-line change; module preserved in `src/retrieval/reranker.py`). | Medium |
| Prompt versioning | Version prompts (v1/v2/v3) and log which version generated each response. Required for A/B testing. Add when you have multiple prompt variants to test. | Medium |
| Incremental indexing | Add/update documents without rebuilding Chroma from scratch. Needed when corpus grows. | Medium |
| Async retrieval | Run BM25 and dense retrieval in parallel. Reduces retrieval latency by ~30–40%. | Medium |
| Embedding + query caching | Cache embeddings for repeated queries. Reduces cost and latency at scale. | Low |
| Automatic document ingestion | Ingest from Ofgem/FCA/DESNZ RSS feeds or GOV.UK APIs. Keeps corpus current. | Low |
| Agentic workflow | Query decomposition, multi-step retrieval, tool use for complex cross-document questions. | v3 |

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

- [x] Embedding model — BAAI/bge-base-en-v1.5. Beat all-MiniLM-L6-v2 on top-5 relevance and cosine sim distribution (mean 0.543 vs 0.278). Locked in config.yaml.
- [x] Chunk size — 400 tokens / 80-token overlap. Validated: avg=397-400, max=400, zero chunks over 512 ceiling. Locked in config.yaml.
- [x] RRF k value — k=60. Literature default (Cormack et al. 2009); k=10/30/60 identical on test query. Locked in config.yaml.
- [x] LLM synthesis model — GPT-4o mini. Beat Haiku 4.5 on quality (4.0 vs 2.7) and cost (6x cheaper). Locked in config.yaml, documented in model_selection.md.

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
      dashboard.py               # Logfire monitoring integration
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


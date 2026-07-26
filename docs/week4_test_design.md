# Week 4 Step 1 — Unit Test Design & Execution

**Date:** 2026-07-13
**Author:** Ashish Siwach
**Scope:** How I decided what to unit-test, how I planned each test, and how it was executed.

---

## 1. Purpose

Document the reasoning behind the Week 4 unit-test suite so future contributors (and future me) can extend it without re-deriving the design principles. Also serves as evidence for the "Testing" rubric item in the LLM Zoomcamp submission.

---

## 2. Testing philosophy (three principles)

Every scope decision below derives from these three principles.

### Principle 1 — Test your code, not third-party libraries

BAAI/bge-base-en-v1.5, Chroma, OpenAI SDK, PyMuPDF, tiktoken, rank-bm25 all have their own test suites. Writing tests like "does BGE produce good embeddings" or "does Chroma order by cosine similarity" would be testing them, not us. Waste of time and brittle to their releases.

What CPIE *owns* and therefore tests:
- Doc registry integrity (12 PDFs, required metadata fields, unique doc_ids)
- Chunker bounds and overlap logic
- Tier 1 stripping regexes (ESO, Ofgem)
- RRF fusion math
- HybridRetriever dedup logic
- Pydantic schema constraints (LLMCitation vs Citation split)
- Citation verification (drops fabricated, injects publication_date)
- Confidence formula (equal weights across 4 signals)
- Out-of-corpus short-circuit branch
- LLM refusal branch
- Guardrails (query length, cost circuit breaker)
- Locked-decision constants (chunk size, RRF k, guardrail limits)

### Principle 2 — Fast + hermetic > slow + comprehensive

Unit tests must run on every save without pain. Concrete bar:
- **Total suite < 5 seconds** (achieved: 1.66s for 58 tests)
- **No network calls** (OpenAI mocked, no HuggingFace downloads)
- **No model loads** (no torch, no sentence-transformers)
- **No real corpus dependency** (PDFs are gitignored — tests use inline text)

Trade-off: this bar means unit tests catch *logic bugs*, not *data quality bugs*. Data quality is the job of the integration tests (`scripts/validate_pipeline_e2e.py`) and Week 5's LLM-as-judge eval on ground truth.

### Principle 3 — Test invariants and locked decisions, not implementation details

Two kinds of tests worth writing:

- **Invariants**: things that must always be true (Pydantic bounds, schema field presence, RRF monotonicity)
- **Locked-decision guardrails**: values that should never drift without a deliberate decision (CHUNK_SIZE=400, RRF k=60, MAX_QUERY_CHARS=500, DAILY_COST_LIMIT_USD=5.00)

What NOT to test:
- Exact intermediate values that could reasonably change (specific RRF scores of specific chunks)
- Internal helper function signatures
- How many times a function calls another function (unless mocking demands it)

---

## 3. Scope decisions — what to unit test vs what to defer

Coverage matrix:

| Module | Test approach | Reasoning |
|---|---|---|
| `ingestion/pdf_loader.py` — `clean_text` | Unit-tested with inline text | Pure function, no I/O — perfect unit test target |
| `ingestion/pdf_loader.py` — `load_pdf` | NOT unit-tested | Requires PDFs (gitignored) — covered by e2e script |
| `ingestion/pdf_loader.py` — `DOC_REGISTRY` | Unit-tested (integrity checks) | Static config — easy invariants to check |
| `ingestion/chunker.py` — `chunk_page` | Unit-tested with synthetic text | Pure function, bounds are contract-critical |
| `retrieval/bm25_retriever.py` | Fully unit-tested (build, query, save/load, edge cases) | Fast, in-memory, no external deps |
| `retrieval/dense_retriever.py` | NOT unit-tested | Requires BGE model + Chroma — covered by e2e |
| `retrieval/hybrid_retriever.py` | Unit-tested with MOCK dense retriever | Test fusion logic, not the retrievers themselves |
| `retrieval/reranker.py` | NOT unit-tested | Not in v1 pipeline — deferred until Week 5 re-eval |
| `synthesis/output_schema.py` | Fully unit-tested (Pydantic bounds, schema separation) | Cheap, catches schema drift |
| `synthesis/synthesiser.py` — helpers | Fully unit-tested (`_verify_citations`, `_compute_confidence`) | Pure functions |
| `synthesis/synthesiser.py` — `synthesise()` | Unit-tested with MOCKED OpenAI client | Test branching (out-of-corpus, refusal, normal), not the LLM |
| `main.py` — `run_query()` guardrails | Unit-tested (length limit, cost breaker) | Safety-critical, easy to test |
| `main.py` — `build_pipeline()` | NOT unit-tested | Requires real indices — covered by e2e |
| `monitoring/logger.py` | Not directly tested; tested transitively via test_main | Very thin — writes JSONL lines, trivial |

**Roughly 60% of the codebase's LOC is exercised.** The uncovered LOC is deliberately deferred to integration testing.

---

## 4. Design pattern — mocks at seams, real objects elsewhere

The most important test-design decision: **where to put the mock boundary**.

For each test file, I asked "what is this file testing?" and mocked everything outside that boundary:

### `test_retrieval.py` seam
Tests HybridRetriever fusion logic. Real BM25 (fast, in-memory). Mocked dense retriever (a `MagicMock` returning fake chunks with fake ranks). The mock proves that the fusion doesn't care about the source — just the ranks. That's exactly what RRF is designed to do.

Consequence for the codebase: this test exposed that `hybrid_retriever.py` had a hard import of `DenseRetriever` at module-load time. Fixed via `if TYPE_CHECKING` (see section 6 below). The fix made the code better AND made the test fast.

### `test_synthesis.py` seam
Tests synthesis logic and refusal branches. Everything real except the OpenAI client, which is patched with `unittest.mock.patch.object`. The mock returns a fake `response.choices[0].message.parsed` object shaped like a real Structured Output. Lets us prove:
- Out-of-corpus branch is taken when top RRF < threshold (`.parse` never called)
- LLM refusal branch triggers the canonical answer (`.parse` called, `refusal` set)
- Normal branch enriches citations with publication_date

### `test_main.py` seam
Tests guardrail branching in `run_query()`. Both `hybrid` and `synth` are `MagicMock` objects with `side_effect=AssertionError` — meaning if the guardrail fails to short-circuit and the pipeline gets called, the test fails loudly. Positive control.

---

## 5. Coverage plan — per test file

For each file I chose tests along three axes:
1. **Happy path** — does the thing work at all?
2. **Edge cases** — empty input, boundary values, malformed input
3. **Contract enforcement** — invariants that other modules depend on

### `test_ingestion.py` (13 tests)

**Doc registry integrity (3 tests)** — invariant checks:
- 12 entries (locked decision)
- All required metadata fields present per entry
- doc_ids are unique (retrieval would break otherwise)

**`clean_text` — Tier 1 stripping (3 tests)** — regex correctness:
- ESO nav elements removed
- Ofgem "OFFICIAL OFFICIAL" removed
- Tier 3 doc (BoE CBES) passes through unchanged

**`chunk_page` — bounds and overlap (6 tests)** — most safety-critical section:
- Long input produces multiple chunks
- Every chunk within CHUNK_SIZE token limit
- Fragments below MIN_TOKENS discarded
- OVERLAP tokens shared between adjacent chunks
- Empty string returns empty list (edge case)
- Non-default chunk_size and overlap args honoured

**Constants match locked decisions (1 test)** — anti-drift guardrail:
- CHUNK_SIZE == 400, OVERLAP == 80, MIN_TOKENS == 50, MAX_TOKENS == 512

### `test_retrieval.py` (13 tests)

**BM25Retriever (7 tests)** — the whole class exercised end-to-end:
- Build then query returns results with `bm25_score` and `bm25_rank`
- Scores strictly positive (zero-score chunks excluded)
- Ranks are 1-indexed and ordered
- Metadata carried through unchanged
- `.query()` before `.build()` raises
- Empty chunk list to `.build()` raises
- Save-and-load roundtrip preserves query results exactly

**RRF fusion math (2 tests)** — pure function:
- `1/(k + rank)` formula matches reference values
- Higher rank yields lower score (monotonic)

**HybridRetriever with mocked dense (4 tests)** — fusion logic:
- Chunks from both retrievers accumulate RRF scores
- Duplicate chunk_ids dedupe correctly
- Returns at most top_k
- RRF k=60 is the locked default

### `test_synthesis.py` (22 tests)

**Pydantic schema (6 tests)** — invariants:
- Confidence bounds enforced (must be in [0, 1])
- Confidence required (no default)
- Citation `page` must be >= 1
- Citation `publication_date` defaults to None
- LLMCitation has no `publication_date` field (LLM must not fabricate it)
- LLMResponse uses LLMCitation not Citation (correct schema for LLM output)

**Citation verification (5 tests)** — the fact-check layer:
- Genuine citation matched and kept
- Fabricated citation dropped
- Match is case-insensitive
- publication_date correctly injected from matched chunk
- Empty input returns empty output

**Confidence computation (7 tests)** — formula correctness:
- All 4 signals returned + `out_of_corpus` + `llm_refusal` flags
- Empty chunks returns 0.0 confidence and 0.0 signals
- Equal weights formula verified against known-signal input
- Single-chunk edge case: margin defaults to 1.0
- margin_signal uses MARGIN_NORMALISER constant
- citation_signal saturates at CITATION_SATURATION
- score_signal uses SCORE_NORMALISER constant

**Synthesis branches (4 tests)** — all three exit paths:
- Out-of-corpus: top RRF < threshold → `.parse` never called
- Empty chunks: short-circuit
- LLM refusal: canonical brief + `llm_refusal=True` in signals
- Normal path: verified citations, publication_date enriched, cost > 0

### `test_main.py` (10 tests)

**`_daily_cost_so_far` (3 tests)** — accumulator correctness:
- Missing log file returns 0.0
- Only today's records count (yesterday excluded)
- Malformed JSON lines skipped without crashing

**Query length guardrail (3 tests)** — refusal + logging:
- Long query refused with canonical message
- Log record written with `failure_reason: guardrail: query_too_long`
- Short query proceeds through normal pipeline (positive control)

**Cost circuit breaker (3 tests)** — refusal + logging:
- Over-budget triggers refusal
- Refusal writes log record with `failure_reason: guardrail: daily_cost_limit`
- Under-budget allows normal pipeline (positive control)

**Constants (1 test)** — anti-drift:
- MAX_QUERY_CHARS == 500, DAILY_COST_LIMIT_USD == 5.00

---

## 6. Refactors forced by the tests (bonus outcome)

Test-first pressure exposed two real design problems that got fixed:

### Problem — `retrieval/__init__.py` eagerly imported `DenseRetriever`

Any test touching `retrieval` transitively loaded `sentence_transformers` → `transformers` → `torch`, which caused a Windows access-violation crash during pytest collection.

**Fix** — PEP 562 lazy `__getattr__` in `retrieval/__init__.py`:

```python
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever

def __getattr__(name: str):
    if name == "DenseRetriever":
        from retrieval.dense_retriever import DenseRetriever
        return DenseRetriever
    if name == "Reranker":
        from retrieval.reranker import Reranker
        return Reranker
    raise AttributeError(f"module 'retrieval' has no attribute {name!r}")
```

Cheap imports stay cheap. Heavy imports only pay their cost when someone actually uses them.

### Problem — `hybrid_retriever.py` imported `DenseRetriever` for a type hint

Even after the `__init__.py` fix, `HybridRetriever`'s type annotation triggered the heavy load.

**Fix** — `TYPE_CHECKING` guard + `from __future__ import annotations`:

```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from retrieval.dense_retriever import DenseRetriever

class HybridRetriever:
    def __init__(self, bm25, dense: "DenseRetriever", rrf_k: int = 60):
        ...
```

The `if TYPE_CHECKING:` block is `False` at runtime (never runs) and `True` for mypy/IDE (still get type-checking). `from __future__ import annotations` makes all annotations lazy strings so `"DenseRetriever"` is never resolved at runtime — only when someone explicitly asks (mypy, `inspect.get_type_hints()`).

### Problem — `main.py` imported `DenseRetriever` at module load

Even after the two fixes above, `main.py` needed `DenseRetriever` to actually construct one in `build_pipeline()`. Direct import brought torch back.

**Fix** — deferred import inside the function:

```python
def build_pipeline():
    from retrieval import DenseRetriever   # imports only when actually needed
    ...
```

Tests never call `build_pipeline()` (they mock the retriever directly), so torch never loads during test collection.

---

## 7. Execution — how I ran the tests

**Setup** — `conftest.py` at `tests/` root does two things:
1. Adds repo root to `sys.path` so `main.py` at the top level is importable
2. Defines shared fixtures (`sample_chunks`, `tmp_log_path`) available to all test files

**Command** — just `uv run pytest tests/` from repo root. `pyproject.toml` already has `[tool.pytest.ini_options] testpaths = ["tests"]` so pytest finds them automatically.

**Iteration loop** — for each test file:
1. Write the test with the mock boundary chosen deliberately
2. Run `uv run pytest tests/test_X.py -v` in isolation
3. Fix failures OR fix the code the failures exposed
4. Run the full suite `uv run pytest tests/ -v` — must stay green + fast

**Debugging the crash** — the Windows access-violation was diagnosed by:
1. Running each test file individually (`test_ingestion` pass, `test_retrieval` pass, `test_synthesis` pass, `test_main` crash) → problem localised
2. Reading the crash traceback backwards to find the offending module (pyarrow → pandas → transformers → torch)
3. Tracing the import chain from `test_main.py` → `main.py` → `retrieval.DenseRetriever` → root cause
4. Fixing at three levels (package init, module type hint, main.py deferred import)

**Final result**: 58 tests pass in 1.66 seconds on Windows, no torch load, no network calls.

---

## 8. What was deliberately NOT tested

Being explicit about non-coverage prevents future confusion.

| Not tested | Why | Where instead |
|---|---|---|
| `pdf_loader.load_pdf` | Requires real PDFs (gitignored) | `scripts/validate_pipeline_e2e.py` |
| Real `DenseRetriever.query` | Requires BGE model + Chroma | `scripts/validate_pipeline_e2e.py` |
| Real `Synthesiser` with live OpenAI | Costs money, needs network | Manual verification during development |
| Reranker | Not in v1 pipeline | Re-evaluation planned Week 5 |
| Retrieval quality (Recall@5, MRR) | Needs ground truth QA | Week 5 eval runner |
| Answer accuracy | Needs ground truth QA + LLM-as-judge | Week 5 eval runner |
| Narrative-hallucination detection | Known v1 limitation | Week 5 eval prompt should detect |
| Chroma persistence | Third-party responsibility | e2e script confirms it works |
| PyMuPDF extraction quality | Third-party responsibility | Week 2 audit already validated |

---

## 9. Handoff for Week 5

When ground truth QA (35–50 pairs) exists, extend testing at the integration level:

**New test file: `tests/test_eval_runner.py`**
- Verify the eval runner processes all ground truth queries
- Verify LLM-as-judge score is in [1, 5]
- Verify results file is written with expected schema

**New tests to add to `test_synthesis.py`** once confidence weights are calibrated:
- Verify fitted weights load from config or a checkpoint file
- Verify the combined formula uses the fitted weights, not the placeholder 0.25s
- Verify HIGH/MEDIUM/LOW threshold cutoffs match calibrated values

**Extend `test_retrieval.py`** if reranker is re-enabled after Week 5 ablation:
- Reranker score computation
- Reranker changes top-k order in expected way with mocked cross-encoder

**Integration test to formalise**: promote `scripts/validate_pipeline_e2e.py` into `tests/integration/test_e2e.py`, mark it as `@pytest.mark.integration`, exclude from default `pytest` run, run separately in CI.

---

## 10. Summary

| Metric | Value |
|---|---|
| Test files | 4 (plus conftest) |
| Total tests | 58 |
| Total runtime | 1.66 seconds |
| External deps loaded | None (no torch, no HuggingFace, no OpenAI network) |
| Code fixes forced by tests | 3 (lazy imports, TYPE_CHECKING, deferred main.py import) |
| Locked-decision guardrails | 3 tests (chunker, RRF k, main.py constants) |
| Mock boundaries | 2 (dense retriever, OpenAI client) |

Suite is intentionally small — every test earns its place by catching a specific class of regression. Doubling the test count with edge-case padding would slow the loop without meaningfully improving safety.

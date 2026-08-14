"""
CPIE — Streamlit chat UI (Week 5 Step 4d).

Single-page Streamlit app: chat-style input, structured brief render, per-answer
thumbs feedback widget writing to cpie.user_feedback.

Run:
    docker compose up -d postgres     # Postgres must be up for feedback + logs
    streamlit run app.py              # host-side; Dockerised in Week 6 Step 1

Notes:
  - No confidence badge (pipeline-derived confidence removed in Week 5 Step 2d
    after signals proved too weak). Every answer carries a standing "verify
    against sources" caveat instead.
  - No auth — anonymous single-user chat. Multi-tenant / session_id
    tracking is not in scope.
  - Postgres feedback write is never-raise (via db.py's own try/except + our
    outer safety wrap) — Postgres down won't break the chat.
"""

# torch must load its BLAS DLLs BEFORE numpy/rank_bm25/openai on Windows —
# same fix as main.py. Streamlit reruns the module top-to-bottom on interaction,
# but torch will only actually initialise once per process.
import torch  # noqa: F401  # isort: skip

import logging
import time
import uuid

import streamlit as st
from dotenv import load_dotenv

# Reuse the exact pipeline main.py uses — one source of truth for what runs.
from main import build_pipeline, run_query
from monitoring import QueryLogger
from monitoring.db import insert_feedback

load_dotenv()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("cpie.app")


# ────────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CPIE — Climate Policy Intelligence Engine",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ────────────────────────────────────────────────────────────────────────
# Pipeline — cached across reruns
# ────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def _load_pipeline():
    """Load Chroma + BM25 + Synthesiser exactly once per Streamlit process.

    Streamlit reruns the whole script on every widget interaction; without
    caching this would reload torch + Chroma + BGE on every click (~20s).
    @st.cache_resource caches non-serialisable objects across all reruns
    and all sessions in the same process.
    """
    hybrid, synth = build_pipeline()
    qlogger = QueryLogger()
    return hybrid, synth, qlogger


# ────────────────────────────────────────────────────────────────────────
# Session state — chat history + per-answer feedback tracking
# ────────────────────────────────────────────────────────────────────────
# Structure: list[dict] where each entry is:
#   { "role": "user",      "content": str }
#   { "role": "assistant", "content": brief_dict, "voted": bool }

if "messages" not in st.session_state:
    st.session_state.messages = []

MAX_CHAT_HISTORY = 40  # bound memory; older messages fall off


def _trim_history() -> None:
    if len(st.session_state.messages) > MAX_CHAT_HISTORY:
        st.session_state.messages = st.session_state.messages[-MAX_CHAT_HISTORY:]


# ────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ────────────────────────────────────────────────────────────────────────

CAVEAT = (
    "⚠️ **CPIE is an assistant, not a source of truth.** "
    "Verify every citation against the source document before relying on it. "
    "CPIE does not provide investment, legal, or regulatory advice."
)


def _render_brief(brief: dict, msg_index: int) -> None:
    """Render an AnalystBrief dict + attach the feedback widget below it."""
    # Error path (pipeline raised)
    if "error" in brief:
        st.error(f"Pipeline failure: {brief['error']}")
        return

    st.markdown(brief.get("answer", "(empty answer)"))

    citations = brief.get("citations", [])
    if citations:
        with st.expander(f"📚 Citations ({len(citations)})", expanded=True):
            for c in citations:
                doc_id = c.get("doc_id", "?")
                page = c.get("page", "?")
                passage = c.get("passage", "")
                pub_date = c.get("publication_date")
                header = f"**{doc_id}**, p.{page}"
                if pub_date:
                    header += f" ({pub_date})"
                st.markdown(header)
                st.markdown(f"> {passage}")
                st.divider()

    contradictions = brief.get("contradictions", [])
    if contradictions:
        with st.expander(f"⚠️ Contradictions flagged ({len(contradictions)})", expanded=False):
            for c in contradictions:
                st.markdown(f"- **{c.get('doc_a', '?')}** vs **{c.get('doc_b', '?')}**: {c.get('summary', '')}")
                st.caption("Contradiction detection is experimental — treat as a hint, not a verdict.")

    st.caption(CAVEAT)

    # ── Feedback widget ──────────────────────────────────────────────
    query_id = brief.get("query_id")
    if not query_id:
        # No query_id means the record wasn't written to Postgres — can't
        # attach feedback. Skip the widget silently.
        return

    msg = st.session_state.messages[msg_index]
    voted = msg.get("voted", False)

    with st.container():
        cols = st.columns([1, 1, 6])
        with cols[0]:
            if st.button("👍 Helpful", key=f"up_{msg_index}", disabled=voted, use_container_width=True):
                _record_feedback(query_id, +1, msg_index)
                st.rerun()
        with cols[1]:
            if st.button("👎 Not helpful", key=f"down_{msg_index}", disabled=voted, use_container_width=True):
                _record_feedback(query_id, -1, msg_index)
                st.rerun()
        with cols[2]:
            if voted:
                st.caption(f"✓ Feedback recorded (query_id: `{query_id[:8]}…`)")


def _record_feedback(query_id: str, verdict: int, msg_index: int) -> None:
    """Write feedback to Postgres via db.insert_feedback, then mark voted in state."""
    try:
        fb_id = insert_feedback(query_id, verdict)
        logger.info("Feedback recorded: query_id=%s verdict=%+d feedback_id=%s", query_id, verdict, fb_id)
    except Exception as e:
        logger.warning("Feedback write skipped: %s", e)
    # Mark voted regardless — user shouldn't be spamming double clicks
    st.session_state.messages[msg_index]["voted"] = True


# ────────────────────────────────────────────────────────────────────────
# Layout
# ────────────────────────────────────────────────────────────────────────

st.title("📄 CPIE — Climate Policy Intelligence Engine")
st.caption(
    "Domain-aware RAG over 12 UK & global climate policy documents "
    "(Ofgem, DESNZ, IEA, BoE, CCC, ESO). Ask a question, get a cited brief."
)

with st.sidebar:
    st.subheader("About this demo")
    st.markdown(
        "- **Pipeline:** Hybrid BM25 + Dense (BAAI/bge-base-en-v1.5) → RRF → "
        "GPT-5.4-mini synthesis with citation verification.\n"
        "- **Prompt version:** v2_numeric (Week 5 Step 3a A/B winner).\n"
        "- **Metadata filter:** on (Week 5 Step 2f).\n"
        "- **No confidence score** — pipeline-derived signals proved too weak "
        "in Week 5 Step 2d calibration; removed for honesty."
    )
    st.subheader("Corpus")
    st.markdown(
        "Ofgem SSES · DESNZ ZEV Mandate · IEA WEO 2025 · "
        "BoE CBES × 3 + Disclosure + Macro Implications · "
        "CCC Progress 2024/2025 + Seventh Carbon Budget · ESO Beyond 2030"
    )
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()


# Pipeline load — cached per process; spinner is shown in the main content area
# so the user gets clear feedback during the ~15 s first-run model load.
with st.spinner(
    "⏳ Loading CPIE pipeline — BM25 index + BAAI/bge-base-en-v1.5 embedding model "
    "(one-time setup, ~15 s on first run)…"
):
    hybrid, synth, qlogger = _load_pipeline()


# ── Render past messages ─────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            _render_brief(msg["content"], msg_index=i)


# ── New query input ──────────────────────────────────────────────────
prompt = st.chat_input("Ask a question about the corpus…")
if prompt:
    # 1. Echo the user's query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Run the pipeline (spinner while waiting)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving + synthesising…"):
            t0 = time.time()
            brief = run_query(prompt, hybrid, synth, qlogger)
            elapsed_s = time.time() - t0

        # Guarantee a query_id — run_query returns one on both happy and
        # error paths, but be defensive.
        brief.setdefault("query_id", str(uuid.uuid4()))

        # Push into history BEFORE rendering so the feedback widget can
        # look itself up by msg_index.
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": brief,
                "voted": False,
            }
        )
        _trim_history()

        _render_brief(brief, msg_index=len(st.session_state.messages) - 1)
        st.caption(f"⏱ {elapsed_s:.1f}s")

"""
CPIE — Streamlit chat UI.

Single-page Streamlit app: chat-style input, structured brief render, per-answer
thumbs feedback widget writing to cpie.user_feedback.

Run:
    docker compose up -d postgres     # Postgres must be up for feedback + logs
    streamlit run app.py

Notes:
  - No confidence badge (pipeline-derived confidence removed after calibration
    showed AUC 0.668 — too weak to show to users).
  - No auth — anonymous single-user chat.
  - Postgres feedback write is never-raise — Postgres down won't break the chat.
"""

# torch must load its BLAS DLLs BEFORE numpy/rank_bm25/openai on Windows —
# same fix as main.py.
import torch  # noqa: F401  # isort: skip

import base64
import logging
import math
import pathlib
import time
import uuid

from PIL import Image, ImageDraw
import streamlit as st
from dotenv import load_dotenv

from main import build_pipeline, run_query
from monitoring import QueryLogger
from monitoring.db import fetch_recent_queries, insert_feedback

load_dotenv()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("cpie.app")


# ────────────────────────────────────────────────────────────────────────
# Page icon (PIL) — drives the browser tab AND the chat-input avatar
# ────────────────────────────────────────────────────────────────────────

def _make_page_icon(size: int = 64) -> Image.Image:
    """Draw the CPIE logo as a PIL image: rounded-square + network graph."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bg = (14, 75, 68, 255)       # #0E4B44
    white = (255, 255, 255, 230)
    white_dim = (255, 255, 255, 160)

    # Rounded square background
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=round(size * 0.18), fill=bg)

    # Node positions (proportional to size)
    top  = (size * 0.50, size * 0.28)   # query node
    left = (size * 0.30, size * 0.65)   # BM25 node
    right= (size * 0.70, size * 0.65)   # Dense node

    def pt(p): return (round(p[0]), round(p[1]))

    # Edges
    lw = max(1, round(size * 0.03))
    draw.line([pt(top), pt(left)],  fill=white_dim, width=lw)
    draw.line([pt(top), pt(right)], fill=white_dim, width=lw)

    # Dashed fusion edge (approximate with short segments)
    segs = 5
    for i in range(segs):
        if i % 2 == 0:
            x0 = left[0]  + (right[0] - left[0])  * i / segs
            x1 = left[0]  + (right[0] - left[0])  * (i + 0.7) / segs
            y0 = left[1]  + (right[1] - left[1])  * i / segs
            y1 = left[1]  + (right[1] - left[1])  * (i + 0.7) / segs
            draw.line([(round(x0), round(y0)), (round(x1), round(y1))],
                      fill=white_dim, width=max(1, lw - 1))

    # Nodes (circles)
    r_top = round(size * 0.09)
    r_bot = round(size * 0.07)
    for centre, r, fill in [(top, r_top, white), (left, r_bot, white_dim), (right, r_bot, white_dim)]:
        x, y = centre
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)

    return img


# ────────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CPIE — Climate Policy Intelligence Engine",
    page_icon=_make_page_icon(),
    layout="wide",
    initial_sidebar_state="expanded",
)


# ────────────────────────────────────────────────────────────────────────
# Pipeline — cached across reruns
# ────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def _load_pipeline():
    """Load Chroma + BM25 + Synthesiser exactly once per Streamlit process."""
    hybrid, synth = build_pipeline()
    qlogger = QueryLogger()
    return hybrid, synth, qlogger


# ────────────────────────────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pg_history" not in st.session_state:
    st.session_state.pg_history = None  # None = not yet fetched

MAX_CHAT_HISTORY = 40


def _trim_history() -> None:
    if len(st.session_state.messages) > MAX_CHAT_HISTORY:
        st.session_state.messages = st.session_state.messages[-MAX_CHAT_HISTORY:]


# ────────────────────────────────────────────────────────────────────────
# Postgres history
# ────────────────────────────────────────────────────────────────────────


def _refresh_pg_history(limit: int = 30) -> None:
    """Pull recent queries from Postgres and cache in session_state."""
    st.session_state.pg_history = fetch_recent_queries(limit=limit)


# ────────────────────────────────────────────────────────────────────────
# Logo helper
# ────────────────────────────────────────────────────────────────────────

_LOGO_PATH = pathlib.Path("docs/images/cpie_logo.svg")


def _logo_img_tag(width: int = 120) -> str:
    """Return an <img> tag with the logo embedded as a base64 data URI."""
    try:
        b64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode()
        return (
            f'<img src="data:image/svg+xml;base64,{b64}" '
            f'width="{width}" style="display:block;margin:0 auto 4px"/>'
        )
    except Exception:
        return ""


# ────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ────────────────────────────────────────────────────────────────────────

CAVEAT = (
    "⚠️ **CPIE is an assistant, not a source of truth.** "
    "Verify every citation against the source document before relying on it. "
    "CPIE does not provide investment, legal, or regulatory advice."
)


def _render_brief(brief: dict, msg_index: int) -> None:
    """Render an AnalystBrief dict and attach the feedback widget below it."""
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
                st.markdown(
                    f"- **{c.get('doc_a', '?')}** vs **{c.get('doc_b', '?')}**: "
                    f"{c.get('summary', '')}"
                )
                st.caption("Contradiction detection is experimental — treat as a hint, not a verdict.")

    st.caption(CAVEAT)

    # Feedback widget
    query_id = brief.get("query_id")
    if not query_id:
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
    try:
        fb_id = insert_feedback(query_id, verdict)
        logger.info(
            "Feedback recorded: query_id=%s verdict=%+d feedback_id=%s",
            query_id, verdict, fb_id,
        )
    except Exception as e:
        logger.warning("Feedback write skipped: %s", e)
    st.session_state.messages[msg_index]["voted"] = True


# ────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo + wordmark
    logo_tag = _logo_img_tag(width=100)
    if logo_tag:
        st.markdown(logo_tag, unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;font-weight:700;font-size:1.15rem;"
        "letter-spacing:.04em;color:#0E4B44;margin-bottom:2px'>CPIE</div>"
        "<div style='text-align:center;font-size:0.72rem;color:gray;"
        "margin-bottom:6px'>Climate Policy Intelligence Engine</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    with st.expander("ℹ️ About"):
        st.markdown(
            "**Pipeline:** Hybrid BM25 + Dense (BAAI/bge-base-en-v1.5) → RRF → "
            "GPT-5.4-mini synthesis with citation verification.\n\n"
            "**Corpus:** Ofgem SSES · DESNZ ZEV Mandate · IEA WEO 2025 · "
            "BoE CBES × 3 + Disclosure + Macro Implications · "
            "CCC Progress 2024/2025 + Seventh Carbon Budget · ESO Beyond 2030"
        )

    # ── Query History ────────────────────────────────────────────────
    st.subheader("🕒 Query History")

    # Current session
    session_queries = [
        msg["content"]
        for msg in st.session_state.messages
        if msg["role"] == "user"
    ]

    if session_queries:
        st.caption(f"**This session** — {len(session_queries)} {'query' if len(session_queries) == 1 else 'queries'}")
        for i, q in enumerate(reversed(session_queries)):
            label = (q[:54] + "…") if len(q) > 54 else q
            if st.button(label, key=f"sess_{i}", use_container_width=True, help=q):
                st.session_state["pending_query"] = q
                st.rerun()
    else:
        st.caption("No queries yet this session.")

    st.divider()

    # Postgres history (cross-session)
    hdr_col, btn_col = st.columns([4, 1])
    with hdr_col:
        st.caption("**All recent queries**")
    with btn_col:
        if st.button("↺", key="refresh_pg", help="Refresh from database"):
            _refresh_pg_history()
            st.rerun()

    # Lazy-load on first render
    if st.session_state.pg_history is None:
        _refresh_pg_history()

    pg_rows: list[dict] = st.session_state.pg_history or []
    session_set = set(session_queries)
    pg_unique = [r for r in pg_rows if r.get("query", "") not in session_set]

    if pg_unique:
        for i, row in enumerate(pg_unique[:20]):
            q = row.get("query", "")
            ts = row.get("ts")
            is_refusal = row.get("is_refusal", False)

            label = (q[:54] + "…") if len(q) > 54 else q
            prefix = "🚫 " if is_refusal else ""

            ts_str = ""
            if ts:
                try:
                    ts_str = ts.strftime("%d %b")
                except Exception:
                    ts_str = str(ts)[:10]

            help_text = f"{q}\n\n{ts_str}" if ts_str else q
            if st.button(f"{prefix}{label}", key=f"pg_{i}", use_container_width=True, help=help_text):
                st.session_state["pending_query"] = q
                st.rerun()
    elif not pg_rows:
        st.caption("Database not reachable — showing session history only.")
    else:
        st.caption("All recent queries are already shown above.")

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ────────────────────────────────────────────────────────────────────────
# Main content
# ────────────────────────────────────────────────────────────────────────

# Compute logo base64 once — used for header image, CSS avatar, and ::before.
try:
    _b64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode()
    _logo_url = f"data:image/svg+xml;base64,{_b64}"
    _header_logo = (
        f'<img src="{_logo_url}" '
        f'width="84" style="display:block;margin:0 auto 14px"/>'
    )
except Exception:
    _b64 = ""
    _logo_url = ""
    _header_logo = ""

# Theme-aware title/subtitle colours + chat-input avatar via CSS ::before.
# The ::before pseudo-element on the chat input's inner flex container inserts
# the logo icon to the left of the textarea without touching the DOM directly.
_chat_avatar_css = (
    f"""
    [data-testid="stChatInput"] {{
        position: relative !important;
    }}
    [data-testid="stChatInput"]::before {{
        content: '' !important;
        position: absolute !important;
        left: 10px !important;
        bottom: 10px !important;
        width: 30px !important;
        height: 30px !important;
        border-radius: 7px !important;
        background-image: url('{_logo_url}') !important;
        background-size: cover !important;
        background-repeat: no-repeat !important;
        z-index: 10 !important;
    }}
    [data-testid="stChatInputTextArea"] {{
        padding-left: 46px !important;
    }}
    """
    if _logo_url else ""
)

st.markdown(
    f"""
    <style>
    :root                    {{ --cpie-title: #0D6E5B; --cpie-sub: #4A5568; }}
    @media (prefers-color-scheme: dark) {{ :root {{ --cpie-title: #34D399; --cpie-sub: #94A3B8; }} }}
    html[data-theme="dark"]  {{ --cpie-title: #34D399; --cpie-sub: #94A3B8; }}
    html[data-theme="light"] {{ --cpie-title: #0D6E5B; --cpie-sub: #4A5568; }}
    {_chat_avatar_css}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"<div style='text-align:center;padding:28px 0 18px'>"
    f"{_header_logo}"
    f"<div style='font-size:1.85rem;font-weight:700;letter-spacing:-0.01em;"
    f"color:var(--cpie-title);line-height:1.2;margin-bottom:6px'>"
    f"Climate Policy Intelligence Engine</div>"
    f"<div style='color:var(--cpie-sub);font-size:0.84rem'>"
    f"Domain-aware RAG over 12 UK &amp; global climate policy documents "
    f"(Ofgem, DESNZ, IEA, BoE, CCC, ESO)</div>"
    f"</div>",
    unsafe_allow_html=True,
)

# Pipeline load — cached per process
_startup_box = st.empty()
if "pipeline_ready" not in st.session_state:
    _startup_box.info(
        "⏳ **Initializing CPIE pipeline…**  \n"
        "Loading BM25 index + BAAI/bge-base-en-v1.5 embedding model.  \n"
        "**First run takes ~15 s.** Subsequent starts are instant (model stays cached).",
        icon="⏳",
    )
hybrid, synth, qlogger = _load_pipeline()
st.session_state["pipeline_ready"] = True
_startup_box.empty()


# ── Render past messages ─────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            _render_brief(msg["content"], msg_index=i)


# ── Pending query from sidebar click ────────────────────────────────
pending: str | None = None
if "pending_query" in st.session_state:
    pending = st.session_state["pending_query"]
    del st.session_state["pending_query"]


# ── New query input ──────────────────────────────────────────────────
prompt = st.chat_input("Ask a question about the corpus…") or pending

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + synthesising…"):
            t0 = time.time()
            brief = run_query(prompt, hybrid, synth, qlogger)
            elapsed_s = time.time() - t0

        brief.setdefault("query_id", str(uuid.uuid4()))

        st.session_state.messages.append(
            {"role": "assistant", "content": brief, "voted": False}
        )
        _trim_history()
        _render_brief(brief, msg_index=len(st.session_state.messages) - 1)
        st.caption(f"⏱ {elapsed_s:.1f}s")

    # Refresh sidebar history so the new query appears immediately
    _refresh_pg_history()
    st.rerun()

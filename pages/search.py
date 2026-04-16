"""
Multi-Doc Search — query across multiple uploaded Word documents.

Supports three search methods:
  Keyword  — exact substring match (case-insensitive)
  Regex    — regular expression match
  Relevance — BM25 ranking across all documents
  Semantic  — cosine similarity via sentence-transformers embeddings
"""

import io
import regex as re

import numpy as np
import streamlit as st
from annotated_text import annotated_text

from src.app_state import (
    KEY_SEARCH_HITS,
    KEY_SEARCH_HITS_KEY,
    KEY_SEARCH_PREF_METHOD,
    KEY_SEARCH_PREF_MIN_SCORE,
    KEY_SEARCH_PREF_MODEL,
    KEY_SEARCH_PREF_QUERY,
    KEY_SEARCH_STORED_FILES,
    KEY_SEARCH_VIEW,
    MAX_UPLOAD_MB,
    BYTES_PER_MB,
    MODEL_MINILM,
    MODEL_MPNET,
    get_file_bytes,
    get_file_name,
)
from src.comments.extract import extract_paragraphs
from src.shared import DocxParseError
from src.utils.models import get_sentence_transformer
from src.stats.config import CFG
from src.utils.text import (
    TOPIC_PALETTE,
    annotate_query_tokens,
    annotate_regex,
    annotate_term,
    bm25_scores,
)


# ---------------------------------------------------------------------------
# Cached extraction
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=10)
def _extract(file_bytes: bytes) -> list[tuple[int, str]]:
    doc = extract_paragraphs(io.BytesIO(file_bytes))
    return [
        (i, p.strip())
        for i, p in enumerate(doc.paragraphs)
        if len(p.strip()) >= CFG.multi_doc_search.min_para_chars
    ]


@st.cache_data(show_spinner=False, max_entries=10)
def _embed(texts: tuple[str, ...], model_name: str) -> np.ndarray:
    encoder = get_sentence_transformer(model_name)
    return encoder.encode(
        list(texts), show_progress_bar=False, normalize_embeddings=True
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.subheader("Multi-Doc Search")

global_bytes = get_file_bytes()
global_name = get_file_name()

# Sidebar — file upload
if global_name:
    st.sidebar.caption(f"📄 {global_name}")

uploaded = st.sidebar.file_uploader(
    "Add more files" if global_name else "Upload documents",
    type=["docx", "doc"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

# Persist extra uploaded files across navigations
if uploaded:
    oversized = [f for f in uploaded if f.size > MAX_UPLOAD_MB * BYTES_PER_MB]
    valid = [f for f in uploaded if f.size <= MAX_UPLOAD_MB * BYTES_PER_MB]
    for f in oversized:
        st.sidebar.warning(
            f"'{f.name}' ({f.size / BYTES_PER_MB:.1f} MB) exceeds the "
            f"{MAX_UPLOAD_MB} MB limit and was skipped."
        )
    st.session_state[KEY_SEARCH_STORED_FILES] = [(f.name, f.getvalue()) for f in valid]

stored_extra: list[tuple[str, bytes]] = st.session_state.get(
    KEY_SEARCH_STORED_FILES, []
)

# Build corpus sources
extra_files = [(f.name, f.getvalue()) for f in uploaded] if uploaded else stored_extra

files_to_use: list[tuple[str, bytes]] = []
if global_bytes and global_name:
    files_to_use.append((global_name, global_bytes))
files_to_use.extend(extra_files)

if not files_to_use:
    st.caption("Upload one or more Word documents to search across them.")
    st.stop()

if not uploaded and stored_extra:
    names = ", ".join(n for n, _ in stored_extra)
    st.sidebar.caption(f"Also: {names}")
    if st.sidebar.button("Clear extra files", key="search_clear_files"):
        del st.session_state[KEY_SEARCH_STORED_FILES]
        st.rerun()

# Build corpus: list of (doc_name, para_idx, para_text)
corpus: list[tuple[str, int, str]] = []
for name, file_bytes in files_to_use:
    try:
        paras = _extract(file_bytes)
    except DocxParseError:
        st.warning(f"'{name}' is not a valid Word document and was skipped.")
        continue
    for para_idx, p in paras:
        corpus.append((name, para_idx, p))

doc_names = [name for name, _, _ in corpus]
para_indices = [para_idx for _, para_idx, _ in corpus]
texts = [text for _, _, text in corpus]

total_docs = len(set(doc_names))
st.caption(
    f"{total_docs} document{'s' if total_docs != 1 else ''} · {len(texts)} passages"
)

# Search input
_saved_query = st.session_state.get(KEY_SEARCH_PREF_QUERY, "")
query = st.text_input(
    "Query",
    value=_saved_query,
    placeholder="Search across all documents…",
    label_visibility="collapsed",
    key="search_query",
)
st.session_state[KEY_SEARCH_PREF_QUERY] = query

_method_options = ["Keyword", "Regex", "Relevance", "Semantic"]
st.session_state.setdefault(KEY_SEARCH_PREF_METHOD, "Keyword")
st.session_state.setdefault("search_method", st.session_state[KEY_SEARCH_PREF_METHOD])
method = (
    st.pills(
        "Search type",
        options=_method_options,
        key="search_method",
        label_visibility="collapsed",
        selection_mode="single",
    )
    or "Keyword"
)
st.session_state[KEY_SEARCH_PREF_METHOD] = method

min_score: float = 0.0
if method in ("Relevance", "Semantic"):
    _saved_min_score = st.session_state.get(KEY_SEARCH_PREF_MIN_SCORE, 0.0)
    min_score = st.sidebar.slider(
        "Minimum score",
        min_value=0.0,
        max_value=1.0,
        value=_saved_min_score,
        step=0.01,
        key="search_min_score",
    )
    st.session_state[KEY_SEARCH_PREF_MIN_SCORE] = min_score

_model_options = [MODEL_MINILM, MODEL_MPNET]
if method == "Semantic":
    _saved_model = st.session_state.get(KEY_SEARCH_PREF_MODEL)
    model_name = st.sidebar.selectbox(
        "Embedding model",
        _model_options,
        index=_model_options.index(_saved_model)
        if _saved_model in _model_options
        else 0,
        key="search_model",
    )
    st.session_state[KEY_SEARCH_PREF_MODEL] = model_name

if not query or not query.strip():
    st.stop()

q = query.strip()

# Build a key that uniquely identifies this search. Hits are cached in session
# state so pagination never re-runs the search.
_score_key = round(min_score, 2) if method in ("Relevance", "Semantic") else 0.0
_model_key = model_name if method == "Semantic" else ""
_corpus_hash = hash(tuple(texts))
search_key = (q, method, _score_key, _model_key, _corpus_hash)

if st.session_state.get(KEY_SEARCH_HITS_KEY) != search_key:
    # Parameters changed — run the search and cache results.
    match method:
        case "Keyword":
            pattern = re.compile(re.escape(q), re.IGNORECASE)
            hits = [(i, 1.0) for i, t in enumerate(texts) if pattern.search(t)]
        case "Regex":
            try:
                pattern = re.compile(q, re.IGNORECASE)
            except re.error as e:
                st.warning(f"Invalid regex: {e}")
                st.stop()
            try:
                hits = [
                    (i, 1.0)
                    for i, t in enumerate(texts)
                    if pattern.search(t, timeout=1.0)
                ]
            except TimeoutError:
                st.warning("Regex pattern timed out. Please simplify your expression.")
                st.stop()
        case "Relevance":
            scores = bm25_scores(texts, q)
            ranked = np.argsort(-scores)
            all_positive = [(int(i), float(scores[i])) for i in ranked if scores[i] > 0]
            hits = [(i, s) for i, s in all_positive if s >= min_score]
            if not hits and all_positive:
                st.info(
                    f"No results above the minimum score ({min_score:.2f}). Try lowering it in the sidebar."
                )
                st.stop()
        case _:  # Semantic
            try:
                with st.spinner("Embedding…"):
                    corpus_embs = _embed(tuple(texts), model_name)
                    encoder = get_sentence_transformer(model_name)
                    q_emb = encoder.encode(
                        [q], show_progress_bar=False, normalize_embeddings=True
                    )[0]
            except RuntimeError as e:
                st.error(str(e))
                st.stop()
            sims = corpus_embs @ q_emb
            ranked = np.argsort(-sims)
            all_ranked = [(int(i), float(sims[i])) for i in ranked]
            hits = [(i, s) for i, s in all_ranked if s >= min_score]
            if not hits and any(s > 0 for _, s in all_ranked):
                st.info(
                    f"No results above the minimum score ({min_score:.2f}). Try lowering it in the sidebar."
                )
                st.stop()

    st.session_state[KEY_SEARCH_HITS] = hits
    st.session_state[KEY_SEARCH_HITS_KEY] = search_key
else:
    hits = st.session_state[KEY_SEARCH_HITS]

if not hits:
    st.info("No matches found.")
    st.stop()

total_hits = len(hits)

# Compile pattern for Regex highlighting (needed on every rerun, including cached pagination)
if method == "Regex":
    try:
        pattern = re.compile(q, re.IGNORECASE)
    except re.error:
        pattern = re.compile(re.escape(q), re.IGNORECASE)
else:
    pattern = None

# Assign a color to each unique document name
unique_docs = list(dict.fromkeys(doc_names))
doc_color = {
    name: TOPIC_PALETTE[i % len(TOPIC_PALETTE)] for i, name in enumerate(unique_docs)
}

col_count, col_view = st.columns([5, 2])
col_count.caption(f"{total_hits} result{'s' if total_hits != 1 else ''}")
show_table = col_view.toggle(
    "Show as table",
    value=st.session_state.get(KEY_SEARCH_VIEW, False),
    key="search_view_toggle",
)
st.session_state[KEY_SEARCH_VIEW] = show_table

if show_table:
    import pandas as pd

    rows = []
    for hit_idx, score in hits:
        rows.append(
            {
                "Document": doc_names[hit_idx],
                "#": para_indices[hit_idx],
                "Passage": texts[hit_idx],
                **(
                    {} if method in ("Keyword", "Regex") else {"Score": round(score, 3)}
                ),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Passage": st.column_config.TextColumn(width="large"),
            "#": st.column_config.NumberColumn(width="small"),
            **(
                {}
                if "Score" not in df.columns
                else {
                    "Score": st.column_config.NumberColumn(width="small", format="%.3f")
                }
            ),
        },
    )
else:
    for idx, score in hits:
        doc_name = doc_names[idx]
        para_idx = para_indices[idx]
        passage = texts[idx]
        color = doc_color[doc_name]

        match method:
            case "Keyword":
                display_parts = annotate_term(passage, q, color)
                score_str = ""
            case "Regex":
                display_parts = annotate_regex(passage, pattern, color)
                score_str = ""
            case "Relevance" | "Semantic":
                display_parts = annotate_query_tokens(passage, q, color)
                score_str = f"{score:.2f}"

        with st.container(border=True):
            col_doc, col_score = st.columns([5, 1])
            with col_doc:
                annotated_text((doc_name, "", color))
                st.caption(f"#{para_idx}")
            with col_score:
                if score_str:
                    st.caption(score_str)
            annotated_text(*display_parts)

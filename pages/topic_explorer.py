import hashlib
import io
import regex as re

import numpy as np
import pandas as pd
import streamlit as st

from src.app_state import (
    KEY_TOPIC_ACTIVE_METHOD,
    KEY_TOPIC_ACTIVE_QUERY,
    KEY_TOPIC_COUNTS,
    KEY_TOPIC_DOCS,
    KEY_TOPIC_EMBEDDINGS,
    KEY_TOPIC_HAD_SCORE,
    KEY_TOPIC_LABEL_LAYERS,
    KEY_TOPIC_MAP_SIG,
    KEY_TOPIC_MAP_TYPE_PREF,
    KEY_TOPIC_PREF_ANALYSIS_UNIT,
    KEY_TOPIC_PREF_EMBEDDING_MODEL,
    KEY_TOPIC_PREF_HIGHLEVEL,
    KEY_TOPIC_PREF_LOWLEVEL,
    KEY_TOPIC_PREF_MIDLEVEL,
    KEY_TOPIC_PREF_MIN_CHARS,
    KEY_TOPIC_REDUCED,
    KEY_TOPIC_STATE_KEY,
    MODEL_MINILM,
    MODEL_MPNET,
    WKEY_TOPIC_SEARCH_METHOD,
    WKEY_TOPIC_SEARCH_QUERY,
    WKEY_TOPIC_SORT_ASC,
    WKEY_TOPIC_SORT_COL,
)
from src.comments.extract import extract_paragraphs
from src.utils.models import get_sentence_transformer
from src.utils.page import expanded_view_controls, require_document
from src.stats.config import CFG
from src.utils.text import (
    bm25_scores,
    highlight_query_tokens,
    highlight_regex,
    highlight_term,
    highlight_topic_keywords,
)
from src.topics.model import (
    clean_docs,
    paragraphs_to_sentences,
    topic_labels,
    default_granularity,
    topic_color_map,
    embed_docs,
    fit_topics,
    reduce_to_2d,
)
from src.topics.render import render_interactive_map, show_map


file_bytes = require_document()

doc_paragraphs = extract_paragraphs(io.BytesIO(file_bytes))

_model_options = [MODEL_MPNET, MODEL_MINILM]

st.sidebar.markdown("### Topic Extraction")
analysis_unit = st.sidebar.segmented_control(
    "Analysis unit",
    options=["Paragraph", "Sentence"],
    default=st.session_state.get(KEY_TOPIC_PREF_ANALYSIS_UNIT, "Paragraph"),
    key="topic_analysis_unit",
    help=(
        "Paragraph is best for legal docs in most cases. It keeps context and yields "
        "cleaner, more stable topics. Sentence is useful for very fine-grained issue "
        "spotting but can be noisier."
    ),
)
st.session_state[KEY_TOPIC_PREF_ANALYSIS_UNIT] = analysis_unit

min_chars = st.sidebar.slider(
    "Minimum text length",
    20,
    400,
    st.session_state.get(KEY_TOPIC_PREF_MIN_CHARS, 80),
    5,
    key="topic_min_chars",
    help="Short text can be noisy. Increase this to focus on richer text.",
)
st.session_state[KEY_TOPIC_PREF_MIN_CHARS] = min_chars

_saved_model = st.session_state.get(KEY_TOPIC_PREF_EMBEDDING_MODEL)
embedding_model_name = st.sidebar.selectbox(
    "Embedding model",
    options=_model_options,
    index=_model_options.index(_saved_model) if _saved_model in _model_options else 0,
    key="topic_embedding_model",
    help="This model converts each paragraph into vectors before topic clustering.",
)
st.session_state[KEY_TOPIC_PREF_EMBEDDING_MODEL] = embedding_model_name

if analysis_unit == "Sentence":
    docs = paragraphs_to_sentences(doc_paragraphs.paragraphs, min_chars=min_chars)
else:
    docs = clean_docs(doc_paragraphs.paragraphs, min_chars=min_chars)

if len(docs) < CFG.topic.min_passages:
    st.warning(
        f"Only {len(docs)} passage{'s' if len(docs) != 1 else ''} remain after filtering "
        f"(minimum {CFG.topic.min_passages} required). "
        f"Try lowering **Minimum text length** (currently {min_chars} chars) "
        f"or switching the analysis unit to **Sentence**."
    )
    st.stop()

st.sidebar.metric("Passages", len(docs))

_PASSAGE_WARN = 1500
if len(docs) > _PASSAGE_WARN:
    st.warning(
        f"**{len(docs)} passages** — topic modelling on large documents may be slow. "
        f"Try raising **Minimum text length** or switching to **Paragraph** mode to reduce this count."
    )


def _on_search_submit() -> None:
    st.session_state[KEY_TOPIC_ACTIVE_QUERY] = st.session_state.get(
        WKEY_TOPIC_SEARCH_QUERY, ""
    )
    st.session_state[KEY_TOPIC_ACTIVE_METHOD] = (
        st.session_state.get(WKEY_TOPIC_SEARCH_METHOD) or "Keyword"
    )


def _search_method_pills() -> None:
    def _on_method_change() -> None:
        st.session_state[KEY_TOPIC_ACTIVE_METHOD] = (
            st.session_state.get(WKEY_TOPIC_SEARCH_METHOD) or "Keyword"
        )

    st.session_state.setdefault(WKEY_TOPIC_SEARCH_METHOD, "Keyword")
    st.pills(
        "Search type",
        options=["Keyword", "Regex", "Relevance", "Semantic"],
        key=WKEY_TOPIC_SEARCH_METHOD,
        label_visibility="collapsed",
        selection_mode="single",
        on_change=_on_method_change,
    )


active_query = st.session_state.get(KEY_TOPIC_ACTIVE_QUERY, "")
active_method = st.session_state.get(KEY_TOPIC_ACTIVE_METHOD, "Keyword")

if active_method in ("Relevance", "Semantic"):
    st.sidebar.markdown("### Search")
    rank_limit = st.sidebar.slider(
        "Result limit",
        min_value=10,
        max_value=500,
        value=200,
        step=10,
        key="topic_rank_limit",
    )

if active_method == "Semantic":
    semantic_min_score = st.sidebar.slider(
        "Minimum score",
        min_value=-1.0,
        max_value=1.0,
        value=0.20,
        step=0.01,
        key="topic_semantic_min",
        help="Higher values return stricter matches.",
    )

normalized_query = active_query.strip().lower()
highlevel_default, midlevel_default, lowlevel_default = default_granularity(len(docs))
_highlevel_max = max(3, min(50, len(docs) // 4))
_midlevel_max = max(3, min(30, len(docs) // 6))
_lowlevel_max = max(3, min(15, len(docs) // 10))


def _to_slider(min_topic_size: int, max_val: int) -> int:
    """Convert min_topic_size → normalized 1–100 slider position."""
    if max_val <= 2:
        return 1
    return max(1, min(100, round((max_val - min_topic_size) * 99 / (max_val - 2)) + 1))


def _from_slider(pos: int, max_val: int) -> int:
    """Convert normalized 1–100 slider position → min_topic_size."""
    if max_val <= 2:
        return 2
    return max(2, round(max_val - (pos - 1) * (max_val - 2) / 99))


st.sidebar.markdown("### Topic Levels")

_LEVEL_CFGS = [
    (
        "High-level",
        KEY_TOPIC_PREF_HIGHLEVEL,
        "topic_highlevel",
        highlevel_default,
        _highlevel_max,
    ),
    (
        "Mid-level",
        KEY_TOPIC_PREF_MIDLEVEL,
        "topic_midlevel",
        midlevel_default,
        _midlevel_max,
    ),
    (
        "Low-level",
        KEY_TOPIC_PREF_LOWLEVEL,
        "topic_lowlevel",
        lowlevel_default,
        _lowlevel_max,
    ),
]

granularity_sizes = []
for _label, _pref_key, _wkey, _default, _max_val in _LEVEL_CFGS:
    _pos = st.sidebar.slider(
        _label,
        1,
        100,
        st.session_state.get(_pref_key, _to_slider(_default, _max_val)),
        key=_wkey,
        help="Move right for more topics; move left for fewer, broader groupings.",
    )
    st.session_state[_pref_key] = _pos
    granularity_sizes.append(_from_slider(_pos, _max_val))

with st.sidebar.expander("Seed words (advanced)", expanded=False):
    st.caption(
        "Guide topics by entering seed words — one group per line, words separated by commas.\n\n"
        "Example:\n```\nprivacy, personal data, consent\nliability, damages\ntermination, expiry\n```"
    )
    seed_words_raw = st.text_area(
        "Seed words",
        label_visibility="collapsed",
        key="topic_seed_words",
        placeholder="privacy, personal data, consent\nliability, damages\ntermination, expiry",
        height=120,
    )

seed_topic_list: tuple[tuple[str, ...], ...] | None = None
if seed_words_raw and seed_words_raw.strip():
    parsed = tuple(
        tuple(w.strip() for w in line.split(",") if len(w.strip()) > 1)
        for line in seed_words_raw.strip().splitlines()
        if line.strip()
    )
    parsed = tuple(group for group in parsed if group)
    if parsed:
        seed_topic_list = parsed
    else:
        st.sidebar.warning(
            "Seed words ignored — no valid words found (each word must be at least 2 characters)."
        )

_topic_state_key = hashlib.md5(
    repr(
        {
            "doc": hashlib.md5(file_bytes).hexdigest(),
            "unit": analysis_unit,
            "min_chars": min_chars,
            "model": embedding_model_name,
            "granularity": granularity_sizes,
            "seeds": (seed_words_raw or "").strip(),
        }
    ).encode()
).hexdigest()

if st.session_state.get(KEY_TOPIC_STATE_KEY) != _topic_state_key:
    try:
        with st.spinner("Embedding and modeling topics..."):
            docs_tuple = tuple(docs)
            embeddings = embed_docs(docs_tuple, embedding_model_name)
            reduced_embeddings = reduce_to_2d(
                embeddings=embeddings,
                n_neighbors=min(30, max(5, len(docs) // 25)),
                min_dist=CFG.topic.umap_min_dist,
            )

            label_layers: list[np.ndarray] = []
            topic_counts: list[int] = []
            for min_topic_size in granularity_sizes:
                topic_model, topics = fit_topics(
                    docs_tuple, embeddings, min_topic_size, seed_topic_list
                )
                labels = topic_labels(topic_model, topics)
                label_layers.append(labels)
                n_topics = len({t for t in topics if t != -1})
                topic_counts.append(n_topics)

    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    st.session_state.update(
        {
            KEY_TOPIC_STATE_KEY: _topic_state_key,
            KEY_TOPIC_DOCS: docs,
            KEY_TOPIC_EMBEDDINGS: embeddings,
            KEY_TOPIC_REDUCED: reduced_embeddings,
            KEY_TOPIC_LABEL_LAYERS: label_layers,
            KEY_TOPIC_COUNTS: topic_counts,
        }
    )
else:
    docs = st.session_state[KEY_TOPIC_DOCS]
    embeddings = st.session_state[KEY_TOPIC_EMBEDDINGS]
    reduced_embeddings = st.session_state[KEY_TOPIC_REDUCED]
    label_layers = st.session_state[KEY_TOPIC_LABEL_LAYERS]
    topic_counts = st.session_state[KEY_TOPIC_COUNTS]

st.sidebar.markdown("### Topics Found")
sidebar_stats_cols = st.sidebar.columns(len(granularity_sizes))
for col, (name, *_), count in zip(sidebar_stats_cols, _LEVEL_CFGS, topic_counts):
    col.metric(name, count)

has_noise = any((labels == "Noise").any() for labels in label_layers)
if has_noise:
    st.sidebar.caption(
        "Noise: text that doesn't cluster into any topic. "
        "Raise minimum text length or adjust topic levels to reduce noise."
    )

score_map: dict[int, float] = {}
if not normalized_query:
    matched_indices = list(range(len(docs)))
else:
    match active_method:
        case "Keyword":
            matched_indices = [
                idx for idx, text in enumerate(docs) if normalized_query in text.lower()
            ]
        case "Regex":
            try:
                pattern = re.compile(active_query.strip(), flags=re.IGNORECASE)
                matched_indices = [
                    idx
                    for idx, text in enumerate(docs)
                    if pattern.search(text, timeout=1.0)
                ]
            except TimeoutError:
                st.warning("Regex pattern timed out. Please simplify your expression.")
                matched_indices = []
            except re.error as e:
                st.warning(f"Invalid regex: {e}")
                matched_indices = []
        case "Relevance":
            scores = bm25_scores(docs, normalized_query)
            ranked = np.argsort(-scores)
            matched_indices = [int(i) for i in ranked if scores[i] > 0][:rank_limit]
            score_map = {int(i): float(scores[i]) for i in matched_indices}
        case _:  # Semantic
            encoder = get_sentence_transformer(embedding_model_name)
            query_embedding = encoder.encode(
                [normalized_query], show_progress_bar=False
            )[0]
            doc_norms = np.linalg.norm(embeddings, axis=1)
            query_norm = float(np.linalg.norm(query_embedding))
            cosine_scores = (embeddings @ query_embedding) / np.clip(
                doc_norms * query_norm, 1e-9, None
            )
            ranked = np.argsort(-cosine_scores)
            matched_indices = [
                int(i) for i in ranked if cosine_scores[i] >= semantic_min_score
            ][:rank_limit]
            score_map = {int(i): float(cosine_scores[i]) for i in matched_indices}


st.subheader("Topic Explorer")

if not matched_indices:
    st.info("No text matches your search. Try a broader term.")
elif len(matched_indices) < 10:
    st.info(
        "Too few matches to render a map (minimum 10). Results are shown in the table below."
    )
else:
    plot_embeddings = reduced_embeddings[matched_indices]
    plot_label_layers = tuple(labels[matched_indices] for labels in label_layers)
    zoom = max(0.33, min(1.0, 15 / len(matched_indices)))

    _map_sig = (tuple(matched_indices), tuple(granularity_sizes), embedding_model_name)
    if st.session_state.get(KEY_TOPIC_MAP_SIG) != _map_sig:
        _n_unique = len({lbl for lbl in plot_label_layers[-1] if lbl != "Noise"})
        st.session_state[KEY_TOPIC_MAP_TYPE_PREF] = (
            "Static" if _n_unique <= CFG.topic.static_map_threshold else "Interactive"
        )
        st.session_state[KEY_TOPIC_MAP_SIG] = _map_sig
        with st.spinner("Building map…"):
            html_content = render_interactive_map(
                plot_embeddings, plot_label_layers, docs, tuple(matched_indices), zoom
            )
    else:
        html_content = render_interactive_map(
            plot_embeddings, plot_label_layers, docs, tuple(matched_indices), zoom
        )

    show_map(plot_embeddings, plot_label_layers, zoom, html_content)

st.markdown("#### Search")
st.text_input(
    "Filter text",
    label_visibility="collapsed",
    key=WKEY_TOPIC_SEARCH_QUERY,
    placeholder="Search across all topics…",
    on_change=_on_search_submit,
)
_search_method_pills()

max_rows = st.sidebar.slider(
    "Max rows", min_value=10, max_value=500, value=100, step=10, key="topic_max_rows"
)

for _k in ("topic_show_expanded", "topic_collapse"):
    st.session_state.setdefault(_k, False)


@st.fragment
def _results_section(
    docs: tuple[str, ...],
    label_layers: list[np.ndarray],
    matched_indices: list[int],
    score_map: dict[int, float],
    search_query: str,
    search_method: str,
    max_rows: int,
    granularity_sizes: list[int],
) -> None:
    topic_columns = ["High-level topics", "Mid-level topics", "Low-level topics"]
    results_df = pd.DataFrame(
        {
            "passage_idx": np.arange(len(docs)),
            "text": docs,
        }
    )
    for idx, size in enumerate(granularity_sizes):
        topic_col_name = (
            topic_columns[idx] if idx < len(topic_columns) else f"Topics {idx + 1}"
        )
        results_df[topic_col_name] = label_layers[idx]

    results_df = (
        results_df.iloc[matched_indices].reset_index(drop=True)
        if matched_indices
        else results_df.iloc[0:0]
    )
    if score_map:
        results_df["score"] = results_df["passage_idx"].map(score_map)
        results_df = results_df.sort_values("score", ascending=False).reset_index(
            drop=True
        )

    selected_topics: list[str] = []

    st.markdown("#### Filter")
    available_topic_cols = [col for col in topic_columns if col in results_df.columns]
    if available_topic_cols and not results_df.empty:
        fcol1, fcol2 = st.columns([1, 3])
        filter_col = fcol1.selectbox(
            "Granularity",
            options=available_topic_cols,
            index=len(available_topic_cols) - 1,
            key="topic_filter_col",
            label_visibility="collapsed",
        )
        topic_options = sorted(
            t for t in results_df[filter_col].unique() if t != "Noise"
        )
        selected_topics = fcol2.multiselect(
            "Topics",
            options=topic_options,
            default=[],
            key="topic_filter_values",
            placeholder=f"Select {filter_col.lower()}…",
            label_visibility="collapsed",
        )
        if selected_topics:
            results_df = results_df[
                results_df[filter_col].isin(selected_topics)
            ].reset_index(drop=True)

    st.markdown("#### Results")
    sortable_cols = [
        c for c in ["score", *topic_columns, "passage_idx"] if c in results_df.columns
    ]
    has_score = bool(score_map)
    if has_score != st.session_state.get(KEY_TOPIC_HAD_SCORE):
        st.session_state[KEY_TOPIC_HAD_SCORE] = has_score
        st.session_state[WKEY_TOPIC_SORT_COL] = "score" if has_score else "passage_idx"
        st.session_state[WKEY_TOPIC_SORT_ASC] = not has_score
    scol1, scol2 = st.columns([3, 1])
    sort_by = scol1.selectbox(
        "Sort by",
        options=sortable_cols,
        index=0,
        key=WKEY_TOPIC_SORT_COL,
        label_visibility="collapsed",
    )
    sort_asc = scol2.toggle("Ascending", value=False, key=WKEY_TOPIC_SORT_ASC)
    results_df = results_df.sort_values(sort_by, ascending=sort_asc).reset_index(
        drop=True
    )

    st.caption(f"{len(results_df)} results")
    show_expanded, _collapse = expanded_view_controls(
        "topic_show_expanded", "topic_collapse"
    )
    expand_all = show_expanded and not _collapse
    if not show_expanded:
        st.dataframe(
            results_df.head(max_rows),
            width="stretch",
            hide_index=True,
            column_config={
                "passage_idx": st.column_config.NumberColumn("#", width="small"),
                "text": st.column_config.TextColumn("Passage", width="large"),
                "score": st.column_config.NumberColumn(
                    "Score", format="%.4f", width="small"
                ),
            },
        )
        return

    shown_df = results_df.head(max_rows).copy()
    if shown_df.empty:
        st.markdown("_No matching rows._")
        return

    topic_cols = [col for col in topic_columns if col in shown_df.columns]
    finest_labels = label_layers[-1]
    color_map = topic_color_map(finest_labels)
    for _, row in shown_df.iterrows():
        passage_idx = int(row["passage_idx"])
        text = str(row["text"])
        match search_method:
            case "Relevance" | "Semantic":
                highlighted = highlight_query_tokens(text, search_query)
            case "Regex":
                try:
                    highlighted = highlight_regex(
                        text, re.compile(search_query.strip(), flags=re.IGNORECASE)
                    )
                except (re.error, TimeoutError):
                    highlighted = highlight_term(text, "")
            case _:
                highlighted = highlight_term(text, search_query)
        topic_label = (
            finest_labels[passage_idx] if passage_idx < len(finest_labels) else ""
        )
        color = color_map.get(str(topic_label), "#ffe066")
        highlighted = highlight_topic_keywords(highlighted, str(topic_label), color)
        topics = " → ".join(f"{row[col]}" for col in topic_cols)
        with st.expander(f"#{passage_idx} — {topics}", expanded=expand_all):
            st.markdown(highlighted, unsafe_allow_html=True)


_results_section(
    docs=docs,
    label_layers=label_layers,
    matched_indices=matched_indices,
    score_map=score_map,
    search_query=active_query,
    search_method=active_method,
    max_rows=max_rows,
    granularity_sizes=granularity_sizes,
)

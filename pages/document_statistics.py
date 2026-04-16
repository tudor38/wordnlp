from pathlib import Path
from typing import NamedTuple
import streamlit as st
from datetime import datetime
import pandas as pd

from src.shared import DocxParseError
from src.stats.compute import (
    build_stats_dfs,
    comment_metrics,
    comment_metrics_from_df,
    filter_by_date,
    load_document,
)
from src.stats.ui import sidebar_controls
from src.stats.render import (
    render_author_bar,
    render_comment_metrics,
    render_timeline,
    render_date_caption,
    COMMENT_FIELDS,
    REDLINE_FIELDS,
    MOVE_FIELDS,
)
from src.app_state import (
    KEY_DOC_BYTES,
    KEY_DOC_FINALIZED,
    KEY_DOC_FINALIZED_DATE,
    KEY_DOC_NAME,
    KEY_COMMENT_TL_EXPAND_ALL,
    KEY_COMMENT_TL_EXPANDED,
    KEY_COMMENT_TL_FIELDS,
    KEY_COMMENT_VIEW,
    KEY_FILTER_AUTHORS,
    KEY_FILTER_DATE_MAX,
    KEY_FILTER_DATE_MIN,
    KEY_REDLINE_TL_EXPAND_ALL,
    KEY_REDLINE_TL_EXPANDED,
    KEY_REDLINE_TL_FIELDS,
    KEY_REDLINE_VIEW,
    KEY_MOVE_TL_EXPAND_ALL,
    KEY_MOVE_TL_EXPANDED,
    KEY_MOVE_TL_FIELDS,
    KEY_MOVE_VIEW,
    KEY_STATS_MAIN_TAB,
    MAX_UPLOAD_MB,
    BYTES_PER_MB,
    get_file_bytes,
    get_file_name,
    set_file_bytes,
    set_file_name,
    seed_widget,
    make_store,
)
from src.stats.config import CFG

_ALLOWED_FILETYPES = CFG.display.allowed_filetypes
_CLOSED_DATE_OFFSET = CFG.display.closed_date_offset_days
_KEY_UPLOAD_SIZE_ERROR = "_upload_size_error"


# ---------------------------------------------------------------------------
# Tab structures
# ---------------------------------------------------------------------------
class MainTabs(NamedTuple):
    comments: str
    redlines: str
    moves: str


class CommentViews(NamedTuple):
    counts: str
    timeline: str


class RedlineViews(NamedTuple):
    counts: str
    timeline: str


class MoveViews(NamedTuple):
    counts: str
    timeline: str


MAIN_TABS = MainTabs(*CFG.document_statistics_tabs.main)
COMMENT_VIEWS = CommentViews(*CFG.document_statistics_tabs.comment_views)
REDLINE_VIEWS = RedlineViews(*CFG.document_statistics_tabs.redline_views)
MOVE_VIEWS = MoveViews(*CFG.document_statistics_tabs.move_views)


# ---------------------------------------------------------------------------
# Session state — initialize permanent keys once
# ---------------------------------------------------------------------------
DEFAULTS: dict = {
    KEY_DOC_FINALIZED: False,
    KEY_DOC_FINALIZED_DATE: None,
    KEY_DOC_BYTES: None,
    KEY_DOC_NAME: None,
    # comment timeline
    KEY_COMMENT_TL_EXPANDED: False,
    KEY_COMMENT_TL_EXPAND_ALL: False,
    KEY_COMMENT_TL_FIELDS: [f.label for f in COMMENT_FIELDS],
    # redline timeline
    KEY_REDLINE_TL_EXPANDED: False,
    KEY_REDLINE_TL_EXPAND_ALL: False,
    KEY_REDLINE_TL_FIELDS: [f.label for f in REDLINE_FIELDS],
    # move timeline
    KEY_MOVE_TL_EXPANDED: False,
    KEY_MOVE_TL_EXPAND_ALL: False,
    KEY_MOVE_TL_FIELDS: [f.label for f in MOVE_FIELDS],
    # filters
    KEY_FILTER_AUTHORS: [],
    KEY_FILTER_DATE_MIN: None,
    KEY_FILTER_DATE_MAX: None,
    # tabs
    KEY_STATS_MAIN_TAB: MAIN_TABS.comments,
    KEY_COMMENT_VIEW: COMMENT_VIEWS.counts,
    KEY_REDLINE_VIEW: REDLINE_VIEWS.counts,
    KEY_MOVE_VIEW: MOVE_VIEWS.counts,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


_CB: dict[str, tuple[str, object]] = {
    "is_closed": (KEY_DOC_FINALIZED, False),
    "closed_date": (KEY_DOC_FINALIZED_DATE, None),
    "show_fields": (KEY_COMMENT_TL_FIELDS, []),
    "r_show_fields": (KEY_REDLINE_TL_FIELDS, []),
    "m_show_fields": (KEY_MOVE_TL_FIELDS, []),
    "timeline_authors": (KEY_FILTER_AUTHORS, []),
    "main_tab": (KEY_STATS_MAIN_TAB, MAIN_TABS.comments),
    "comment_tab": (KEY_COMMENT_VIEW, COMMENT_VIEWS.counts),
    "redline_tab": (KEY_REDLINE_VIEW, REDLINE_VIEWS.counts),
    "move_tab": (KEY_MOVE_VIEW, MOVE_VIEWS.counts),
}
_stores = {name: make_store(key, default) for name, (key, default) in _CB.items()}

_store_is_closed = _stores["is_closed"]
_store_closed_date = _stores["closed_date"]
_store_show_fields = _stores["show_fields"]
_store_r_show_fields = _stores["r_show_fields"]
_store_m_show_fields = _stores["m_show_fields"]
_store_timeline_authors = _stores["timeline_authors"]
_store_main_tab = _stores["main_tab"]
_store_comment_tab = _stores["comment_tab"]
_store_redline_tab = _stores["redline_tab"]
_store_move_tab = _stores["move_tab"]


def _store_date_range() -> None:
    val = st.session_state.get("_filter_date_range")
    if val:
        st.session_state[KEY_FILTER_DATE_MIN] = val[0]
        st.session_state[KEY_FILTER_DATE_MAX] = val[1]


def _store_uploaded_file() -> None:
    f = st.session_state.get("_doc_upload")
    if f is not None:
        if f.size > MAX_UPLOAD_MB * BYTES_PER_MB:
            st.session_state[_KEY_UPLOAD_SIZE_ERROR] = (
                f"'{f.name}' is {f.size / BYTES_PER_MB:.1f} MB — "
                f"maximum allowed size is {MAX_UPLOAD_MB} MB."
            )
            return
        st.cache_data.clear()
        set_file_bytes(f.read())
        set_file_name(f.name)
        st.session_state.pop("_demo_banner_dismissed", None)
        for key, val in DEFAULTS.items():
            if key not in (KEY_DOC_BYTES, KEY_DOC_NAME):
                st.session_state[key] = val
        for widget_key in (
            "_filter_date_range",
            "_filter_authors",
            "_doc_finalized",
            "_doc_finalized_date",
            "_stats_main_tab",
            "_comment_view",
            "_redline_view",
            "_comment_tl_expanded",
            "_comment_tl_expand_all",
            "_comment_tl_fields",
            "_redline_tl_expanded",
            "_redline_tl_expand_all",
            "_redline_tl_fields",
            "_move_view",
            "_move_tl_expanded",
            "_move_tl_expand_all",
            "_move_tl_fields",
        ):
            st.session_state.pop(widget_key, None)
        f.seek(0)
    else:
        set_file_bytes(None)
        set_file_name(None)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------
def _render_comment_timeline(
    filtered_c_df: pd.DataFrame,
    all_authors: list[str],
) -> None:
    seed_widget(KEY_COMMENT_TL_FIELDS)
    render_timeline(
        filtered_c_df,
        "Who commented? When?",
        fields=COMMENT_FIELDS,
        display_cols=[
            "author",
            "date",
            "kind",
            "resolved",
            "comment",
            "selected",
            "sentence",
            "paragraph",
        ],
        default_fields=["Marked Resolved", "Sentence", "Comment"],
        all_authors=all_authors,
        expanded_key=KEY_COMMENT_TL_EXPANDED,
        collapse_key=KEY_COMMENT_TL_EXPAND_ALL,
        show_fields_key="_comment_tl_fields",
        on_show_fields=_store_show_fields,
    )


def _render_redline_timeline(
    filtered_r_df: pd.DataFrame,
    all_authors: list[str],
) -> None:
    seed_widget(KEY_REDLINE_TL_FIELDS)
    render_timeline(
        filtered_r_df,
        "Who redlined? When?",
        fields=REDLINE_FIELDS,
        display_cols=["author", "date", "kind", "text", "sentence", "paragraph"],
        default_fields=["Redline", "Sentence"],
        all_authors=all_authors,
        expanded_key=KEY_REDLINE_TL_EXPANDED,
        collapse_key=KEY_REDLINE_TL_EXPAND_ALL,
        show_fields_key="_redline_tl_fields",
        on_show_fields=_store_r_show_fields,
    )


def _render_comments(
    filtered_c_df: pd.DataFrame,
    reference_date: datetime,
    is_closed: bool,
    comments: list,
    all_authors: list[str],
) -> None:
    render_date_caption(filtered_c_df, reference_date, is_closed)
    render_comment_metrics(comment_metrics_from_df(filtered_c_df))

    seed_widget(KEY_COMMENT_VIEW)
    comment_tab = st.pills(
        "View",
        list(COMMENT_VIEWS),
        key="_comment_view",
        on_change=_store_comment_tab,
        selection_mode="single",
        label_visibility="collapsed",
    )

    comment_view_renderers = {
        COMMENT_VIEWS.counts: lambda: render_author_bar(
            filtered_c_df, "Who commented? How much?", all_authors=all_authors
        ),
        COMMENT_VIEWS.timeline: lambda: _render_comment_timeline(
            filtered_c_df, all_authors
        ),
    }
    if comment_tab in comment_view_renderers:
        comment_view_renderers[comment_tab]()


def _render_redlines(
    filtered_r_df: pd.DataFrame,
    reference_date: datetime,
    is_closed: bool,
    all_authors: list[str],
) -> None:
    render_date_caption(filtered_r_df, reference_date, is_closed)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", len(filtered_r_df), border=True)
    col2.metric(
        "Insertions",
        int((filtered_r_df["kind"] == "insertion").sum())
        if not filtered_r_df.empty
        else 0,
        border=True,
    )
    col3.metric(
        "Deletions",
        int((filtered_r_df["kind"] == "deletion").sum())
        if not filtered_r_df.empty
        else 0,
        border=True,
    )

    seed_widget(KEY_REDLINE_VIEW)
    redline_tab = st.pills(
        "View",
        list(REDLINE_VIEWS),
        key="_redline_view",
        on_change=_store_redline_tab,
        selection_mode="single",
        label_visibility="collapsed",
    )

    redline_view_renderers = {
        REDLINE_VIEWS.counts: lambda: render_author_bar(
            filtered_r_df, "Who redlined? How much?", all_authors=all_authors
        ),
        REDLINE_VIEWS.timeline: lambda: _render_redline_timeline(
            filtered_r_df, all_authors
        ),
    }
    if redline_tab in redline_view_renderers:
        redline_view_renderers[redline_tab]()


def _render_move_timeline(
    filtered_m_df: pd.DataFrame,
    all_authors: list[str],
) -> None:
    seed_widget(KEY_MOVE_TL_FIELDS)
    render_timeline(
        filtered_m_df,
        "Who moved text? When?",
        fields=MOVE_FIELDS,
        display_cols=[
            "author",
            "date",
            "text",
            "distance",
            "from_para_idx",
            "to_para_idx",
        ],
        default_fields=[f.label for f in MOVE_FIELDS],
        all_authors=all_authors,
        expanded_key=KEY_MOVE_TL_EXPANDED,
        collapse_key=KEY_MOVE_TL_EXPAND_ALL,
        show_fields_key="_move_tl_fields",
        on_show_fields=_store_m_show_fields,
    )


def _render_moves(
    filtered_m_df: pd.DataFrame,
    reference_date: datetime,
    is_closed: bool,
    all_authors: list[str],
) -> None:
    render_date_caption(filtered_m_df, reference_date, is_closed)
    st.metric("Total Moves", len(filtered_m_df), border=True)

    seed_widget(KEY_MOVE_VIEW)
    move_tab = st.pills(
        "View",
        list(MOVE_VIEWS),
        key="_move_view",
        on_change=_store_move_tab,
        selection_mode="single",
        label_visibility="collapsed",
    )

    move_view_renderers = {
        MOVE_VIEWS.counts: lambda: render_author_bar(
            filtered_m_df, "Who moved text? How much?", all_authors=all_authors
        ),
        MOVE_VIEWS.timeline: lambda: _render_move_timeline(filtered_m_df, all_authors),
    }
    if move_tab in move_view_renderers:
        move_view_renderers[move_tab]()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
_stored_name = st.session_state.get(KEY_DOC_NAME)
if _stored_name and not st.session_state.get("_doc_upload"):
    st.sidebar.caption(f"📄 {_stored_name}")
    if st.sidebar.button("Clear file", key="stats_clear_file", width="stretch"):
        set_file_bytes(None)
        set_file_name(None)
        st.rerun()
else:
    st.sidebar.file_uploader(
        "Choose a file",
        type=_ALLOWED_FILETYPES,
        key="_doc_upload",
        on_change=_store_uploaded_file,
    )
    _size_err = st.session_state.pop(_KEY_UPLOAD_SIZE_ERROR, None)
    if _size_err:
        st.sidebar.error(_size_err)

file_bytes = get_file_bytes()

_DEFAULT_DOC_NAME = "services_agreement.docx"

if not file_bytes:
    _default = Path(__file__).parent.parent / "test_docs" / _DEFAULT_DOC_NAME
    if _default.exists():
        _data = _default.read_bytes()
        set_file_bytes(_data)
        set_file_name(_default.name)
        file_bytes = _data
    else:
        st.info("Upload a Word document using the sidebar to get started.")
        st.stop()

if get_file_name() == _DEFAULT_DOC_NAME and not st.session_state.get(
    "_demo_banner_dismissed"
):
    _col_msg, _col_btn = st.columns([10, 1])
    with _col_msg:
        st.info(
            "**Demo document loaded** — this is a fictitious sample file for "
            "demonstration purposes only. Upload your own Word document using the "
            "sidebar to analyze it.",
            icon="ℹ️",
        )
    with _col_btn:
        st.write("")  # vertical alignment nudge
        if st.button("✕", key="_dismiss_demo_banner", help="Dismiss"):
            st.session_state["_demo_banner_dismissed"] = True
            st.rerun()

if file_bytes:
    try:
        comments, version, redlines, moves, doc_paragraphs = load_document(file_bytes)
    except DocxParseError as e:
        st.error(f"Could not read the uploaded file: {e}")
        st.stop()

    c_df, r_df, m_df, _all_authors = build_stats_dfs(
        comments, redlines, moves, datetime.now()
    )

    reference_date, is_closed, date_range, selected_authors = sidebar_controls(
        comments,
        redlines,
        [c_df, r_df, m_df],
        store_is_closed=_store_is_closed,
        store_closed_date=_store_closed_date,
        store_date_range=_store_date_range,
        store_timeline_authors=_store_timeline_authors,
    )

    if is_closed:
        c_df, r_df, m_df, _all_authors = build_stats_dfs(
            comments, redlines, moves, reference_date
        )

    all_authors = sorted(c_df["author"].unique().tolist()) if not c_df.empty else []

    filtered_c_df = filter_by_date(c_df, date_range[0], date_range[1])
    filtered_r_df = filter_by_date(r_df, date_range[0], date_range[1])
    filtered_m_df = filter_by_date(m_df, date_range[0], date_range[1])

    def _filter_authors(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or not selected_authors:
            return df
        return df[df["author"].isin(selected_authors)].reset_index(drop=True)

    filtered_c_df = _filter_authors(filtered_c_df)
    filtered_r_df = _filter_authors(filtered_r_df)
    filtered_m_df = _filter_authors(filtered_m_df)

    seed_widget(KEY_STATS_MAIN_TAB)
    main_tab = st.pills(
        "Section",
        list(MAIN_TABS),
        key="_stats_main_tab",
        on_change=_store_main_tab,
        selection_mode="single",
        label_visibility="collapsed",
    )

    match main_tab:
        case MAIN_TABS.comments:
            _render_comments(
                filtered_c_df, reference_date, is_closed, comments, all_authors
            )
        case MAIN_TABS.redlines:
            _render_redlines(filtered_r_df, reference_date, is_closed, all_authors)
        case MAIN_TABS.moves:
            _render_moves(filtered_m_df, reference_date, is_closed, all_authors)

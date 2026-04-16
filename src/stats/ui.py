from collections.abc import Callable
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

from src.stats.compute import latest_date
from src.stats.config import CFG
from src.app_state import (
    KEY_DOC_FINALIZED,
    KEY_DOC_FINALIZED_DATE,
    KEY_FILTER_AUTHORS,
    KEY_FILTER_DATE_MAX,
    KEY_FILTER_DATE_MIN,
)


def sidebar_controls(
    comments: list,
    redlines: list,
    all_dfs: list[pd.DataFrame],
    *,
    store_is_closed: Callable[[], None],
    store_closed_date: Callable[[], None],
    store_date_range: Callable[[], None],
    store_timeline_authors: Callable[[], None],
) -> tuple[datetime, bool, tuple, list[str]]:
    st.sidebar.markdown("### Document")

    # Ensure required session state exists.
    st.session_state.setdefault(KEY_DOC_FINALIZED, False)
    st.session_state.setdefault(KEY_DOC_FINALIZED_DATE, None)
    st.session_state.setdefault(KEY_FILTER_AUTHORS, [])
    st.session_state.setdefault(KEY_FILTER_DATE_MIN, None)
    st.session_state.setdefault(KEY_FILTER_DATE_MAX, None)

    # Initialize widget-backed values from permanent keys.
    st.session_state.setdefault(
        f"_{KEY_DOC_FINALIZED}", st.session_state[KEY_DOC_FINALIZED]
    )
    st.session_state.setdefault(
        f"_{KEY_DOC_FINALIZED_DATE}", st.session_state[KEY_DOC_FINALIZED_DATE]
    )

    is_closed = st.sidebar.toggle(
        "Matter is closed",
        key=f"_{KEY_DOC_FINALIZED}",
        on_change=store_is_closed,
        help=(
            "Toggle ON if the document is part of a closed matter. "
            "This setting affects date calculations."
        ),
    )

    if is_closed:
        latest = latest_date(comments, redlines)
        default = latest.date() if latest else datetime.today().date()
        if (
            KEY_DOC_FINALIZED_DATE not in st.session_state
            or st.session_state[KEY_DOC_FINALIZED_DATE] is None
        ):
            st.session_state[KEY_DOC_FINALIZED_DATE] = default
        st.session_state.setdefault(
            f"_{KEY_DOC_FINALIZED_DATE}", st.session_state[KEY_DOC_FINALIZED_DATE]
        )
        closed_date = st.sidebar.date_input(
            "Closed date",
            key=f"_{KEY_DOC_FINALIZED_DATE}",
            on_change=store_closed_date,
        )
        if closed_date is None:
            closed_date = st.session_state[KEY_DOC_FINALIZED_DATE]
        reference_date = datetime.combine(closed_date, datetime.min.time()) + timedelta(
            days=CFG.display.closed_date_offset_days
        )
    else:
        reference_date = datetime.now()

    non_empty = [df for df in all_dfs if not df.empty]
    if not non_empty:
        return (
            reference_date,
            is_closed,
            (datetime.now().date(), datetime.now().date()),
            [],
        )

    global_date_min = min(df["date"].min().date() for df in non_empty)
    data_date_max = max(df["date"].max().date() for df in non_empty)
    global_date_max = (
        data_date_max if is_closed else min(data_date_max, datetime.now().date())
    )

    saved_min = max(
        st.session_state[KEY_FILTER_DATE_MIN] or global_date_min, global_date_min
    )
    saved_max = min(
        st.session_state[KEY_FILTER_DATE_MAX] or global_date_max, global_date_max
    )

    date_range = st.sidebar.slider(
        "Date range",
        min_value=global_date_min,
        max_value=global_date_max,
        value=(saved_min, saved_max),
        key="_filter_date_range",
        on_change=store_date_range,
    )

    c_df = all_dfs[0]
    if not c_df.empty:
        all_authors = sorted(c_df["author"].unique().tolist())
        st.session_state["_filter_authors"] = (
            st.session_state[KEY_FILTER_AUTHORS] or all_authors
        )
        selected_authors = (
            st.sidebar.multiselect(
                "Comment authors",
                options=all_authors,
                key="_filter_authors",
                on_change=store_timeline_authors,
            )
            or all_authors
        )
    else:
        selected_authors = []

    return reference_date, is_closed, date_range, selected_authors

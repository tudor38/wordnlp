import html
from collections import defaultdict
from datetime import datetime
from typing import NamedTuple

import numpy as np

from src.stats.compute import CommentMetrics
from src.stats.config import CFG
from src.utils.page import expanded_view_controls
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from src.comments.render import (
    render_paragraph_with_highlight,
    render_paragraph_with_redline,
    render_paragraph_with_redline_pair,
)

# Tableau10 desaturated — consistent across Altair and Plotly
AUTHOR_PALETTE = [
    "#7fa7c9",
    "#f5b97a",
    "#e88b8c",
    "#9dcdc9",
    "#8dbc8a",
    "#f2d97e",
    "#c9a3c4",
    "#ffbfc8",
    "#c0a08a",
    "#d0ccc8",
]

_DATE_FMT = CFG.display.date_format
_CARD_HEIGHT = CFG.chart.card_height
_MARKER_SIZE = CFG.chart.marker_size
_MARKER_OPACITY = CFG.chart.marker_opacity
_BAR_HEIGHT_PER_ROW = CFG.chart.bar_height_per_row
_BAR_HEIGHT_BASE = CFG.chart.bar_height_base
_TL_HEIGHT_PER_AUTHOR = CFG.chart.timeline_height_per_author
_TL_HEIGHT_BASE = CFG.chart.timeline_height_base


def _author_colors(authors: list[str]) -> list[str]:
    return [AUTHOR_PALETTE[i % len(AUTHOR_PALETTE)] for i in range(len(authors))]


def _author_color_scale_from(authors: list[str]) -> alt.Scale:
    return alt.Scale(domain=authors, range=_author_colors(authors))


def _author_color_map(authors: list[str]) -> dict[str, str]:
    return dict(zip(authors, _author_colors(authors)))


# ---------------------------------------------------------------------------
# Timeline field config
# ---------------------------------------------------------------------------
class TimelineField(NamedTuple):
    """Maps a display label to a DataFrame column and an optional highlight column."""

    label: str
    col: str
    highlight: str | None = None  # column to use for paragraph highlight


# Pre-built field configs for comments and redlines
COMMENT_FIELDS: list[TimelineField] = [
    TimelineField("Marked Resolved", "Resolved", None),
    TimelineField("Comment", "Comment", None),
    TimelineField("Selected", "Selected", "Selected"),
    TimelineField("Sentence", "Sentence", "Selected"),
    TimelineField("Paragraph", "Paragraph", "Selected"),
]

REDLINE_FIELDS: list[TimelineField] = [
    TimelineField("Redline", "Text", None),
    TimelineField("Sentence", "Sentence", "Text"),
    TimelineField("Paragraph", "Paragraph", "Text"),
]

MOVE_FIELDS: list[TimelineField] = [
    TimelineField("Text", "Text", None),
    TimelineField("Distance", "Distance", None),
    TimelineField("From Para", "From_para_idx", None),
    TimelineField("To Para", "To_para_idx", None),
]


# ---------------------------------------------------------------------------
# Date caption
# ---------------------------------------------------------------------------
def render_date_caption(
    df: pd.DataFrame, reference_date: datetime, is_closed: bool
) -> None:
    if df.empty:
        return
    earliest = df["date"].min().strftime(_DATE_FMT)
    end = (
        df["date"].max().strftime(_DATE_FMT)
        if is_closed
        else reference_date.strftime(_DATE_FMT)
    )
    st.caption(f"From {earliest} → {end}")


# ---------------------------------------------------------------------------
# Comment metrics
# ---------------------------------------------------------------------------
def render_comment_metrics(metrics: CommentMetrics, n_cols: int = 2) -> None:
    items = [
        ("Total", metrics.total, None),
        ("Marked Resolved", metrics.resolved, None),
    ]
    cols = st.columns(n_cols)
    for col, (label, value, delta) in zip(cols, items):
        tile = col.container(border=True, height=_CARD_HEIGHT)
        tile.metric(label, value, delta=delta)


# ---------------------------------------------------------------------------
# Author bar
# ---------------------------------------------------------------------------
def render_author_bar(
    df: pd.DataFrame,
    title: str,
    all_authors: list[str] | None = None,
) -> None:
    if df.empty:
        st.caption(f"No data for `{title}`.")
        return

    authors = all_authors if all_authors else sorted(df["author"].unique().tolist())
    color_scale = _author_color_scale_from(authors)

    counts = (
        df.groupby("author")
        .size()
        .to_frame("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )

    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Count", axis=alt.Axis(tickMinStep=1, format="d")),
            y=alt.Y("author:N", sort="-x", title=None),
            color=alt.Color("author:N", scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip("author:N", title="Author"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
        .properties(
            title=title,
            height=_BAR_HEIGHT_PER_ROW * len(counts) + _BAR_HEIGHT_BASE,
        )
    )
    st.altair_chart(chart, width="stretch")


# ---------------------------------------------------------------------------
# Timeline expanded-view helpers
# ---------------------------------------------------------------------------
def _detect_pairs(display: pd.DataFrame) -> tuple[dict[int, int], set[int]]:
    """Find insertion/deletion pairs by the same author on the same date."""
    if "Kind" not in display.columns:
        return {}, set()
    kind_set = set(display["Kind"].unique())
    if "insertion" not in kind_set or "deletion" not in kind_set:
        return {}, set()
    ins_by_key: dict[tuple, list[int]] = defaultdict(list)
    del_by_key: dict[tuple, list[int]] = defaultdict(list)
    for idx, r in display.iterrows():
        key = (r["Author"], r["Date"])
        if r["Kind"] == "insertion":
            ins_by_key[key].append(idx)
        elif r["Kind"] == "deletion":
            del_by_key[key].append(idx)
    pair_partner: dict[int, int] = {}
    pair_skip: set[int] = set()
    for key in ins_by_key:
        if key in del_by_key:
            for ins_i, del_i in zip(ins_by_key[key], del_by_key[key]):
                leader, follower = min(ins_i, del_i), max(ins_i, del_i)
                pair_partner[leader] = follower
                pair_skip.add(follower)
    return pair_partner, pair_skip


def _render_fields(
    row: pd.Series,
    labels: list[str],
    field_map: dict[str, TimelineField],
    columns: frozenset[str],
) -> None:
    """Render field values for a single row in expanded view."""
    for label in labels:
        tf = field_map.get(label)
        if tf is None or tf.col not in columns:
            continue
        val = row[tf.col]
        if not val and val != 0:
            continue
        if tf.col == "Resolved":
            st.markdown(f"**Marked Resolved:** {'Yes' if val else 'No'}")
        elif tf.highlight and tf.highlight in columns:
            hl_val = row[tf.highlight]
            st.markdown(f"**{label}:**")
            items = val if isinstance(val, list) else [val]
            kind_val = row.get("Kind") if "Kind" in columns else None
            for item in items:
                if kind_val in ("insertion", "deletion"):
                    render_paragraph_with_redline(item, hl_val or "", kind_val)
                else:
                    render_paragraph_with_highlight(item, hl_val or "")
        else:
            st.markdown(f"**{label}:** {val}")


def _render_pair_fields(
    ins_row: pd.Series,
    del_row: pd.Series,
    labels: list[str],
    field_map: dict[str, TimelineField],
    columns: frozenset[str],
) -> None:
    """Render field values for an insertion+deletion pair in expanded view."""
    for label in labels:
        tf = field_map.get(label)
        if tf is None or tf.col not in columns:
            continue
        if tf.highlight and tf.highlight in columns:
            st.markdown(f"**{label}:**")
            items = del_row[tf.col]
            items = items if isinstance(items, list) else [items]
            for item in items:
                render_paragraph_with_redline_pair(
                    item, del_row[tf.highlight] or "", ins_row[tf.highlight] or ""
                )
        else:
            del_val = del_row[tf.col]
            ins_val = ins_row[tf.col]
            if not del_val and not ins_val:
                continue
            del_span = f'<span style="color:#ef4444;text-decoration:line-through">{html.escape(str(del_val))}</span>'
            ins_span = f'<span style="color:#3b82f6;text-decoration:underline">{html.escape(str(ins_val))}</span>'
            st.markdown(f"**{label}:** {del_span} → {ins_span}", unsafe_allow_html=True)


def _render_as_expanded(
    display: pd.DataFrame,
    show_fields: list[str],
    field_map: dict[str, TimelineField],
    expand_all: bool,
) -> None:
    columns = frozenset(display.columns)
    pair_partner, pair_skip = _detect_pairs(display)
    for idx, row in display.iterrows():
        if idx in pair_skip:
            continue
        if idx in pair_partner:
            partner_row = display.loc[pair_partner[idx]]
            ins_row, del_row = (
                (row, partner_row) if row["Kind"] == "insertion" else (partner_row, row)
            )
            with st.expander(
                f"{row['Author']} · {row['Date']} · edit", expanded=expand_all
            ):
                _render_pair_fields(ins_row, del_row, show_fields, field_map, columns)
        else:
            kind_label = f" · {row['Kind']}" if "Kind" in columns else ""
            with st.expander(
                f"{row['Author']} · {row['Date']}{kind_label}", expanded=expand_all
            ):
                _render_fields(row, show_fields, field_map, columns)


def _render_as_table(
    display: pd.DataFrame,
    show_fields: list[str],
    field_map: dict[str, TimelineField],
) -> None:
    keep_cols = ["Author", "Date"]
    if "Kind" in display.columns:
        keep_cols.append("Kind")
    for label in show_fields:
        tf = field_map.get(label)
        if tf and tf.col.capitalize() in display.columns:
            keep_cols.append(tf.col.capitalize())
    st.dataframe(
        display[[c for c in keep_cols if c in display.columns]],
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Generalized timeline
# ---------------------------------------------------------------------------
def render_timeline(
    df: pd.DataFrame,
    title: str,
    fields: list[TimelineField],
    display_cols: list[str],  # df columns to include in table
    default_fields: list[str],  # default multiselect selection
    all_authors: list[str] | None = None,
    expanded_key: str = "expanded_view",  # permanent key
    collapse_key: str = "collapse",  # permanent key
    show_fields_key: str = "_show_fields",
    on_show_fields=None,
) -> None:
    """
    Generalized timeline scatter + detail table.

    Parameters
    ----------
    fields         : list of TimelineField defining available fields for the
                     expanded view and multiselect options
    display_cols   : df columns to pull into the display table
    default_fields : which field labels are selected by default
    """
    if df.empty:
        st.caption("No data matches the current filters.")
        return

    rng = np.random.default_rng(42)
    df = df.copy().reset_index(drop=True)

    visible_authors = sorted(df["author"].unique().tolist())
    color_authors = all_authors if all_authors else visible_authors
    color_map = _author_color_map(color_authors)
    author_idx = {a: i for i, a in enumerate(visible_authors)}

    df["jitter"] = rng.uniform(-0.3, 0.3, size=len(df))
    df["_idx"] = df.index
    df["y_jittered"] = df.apply(lambda r: author_idx[r["author"]] + r["jitter"], axis=1)

    hover_data: dict = {
        "date": "|" + _DATE_FMT,
        "author": True,
        "y_jittered": False,
    }
    if "resolved" in df.columns:
        hover_data["resolved"] = True
    if "kind" in df.columns:
        hover_data["kind"] = True
    hover_data["_idx"] = True

    fig = px.scatter(
        df,
        x="date",
        y="y_jittered",
        color="author",
        color_discrete_map=color_map,
        hover_data=hover_data,
        title=title,
        height=_TL_HEIGHT_PER_AUTHOR * len(visible_authors) + _TL_HEIGHT_BASE,
    )

    fig.update_layout(
        yaxis=dict(
            tickvals=list(range(len(visible_authors))),
            ticktext=visible_authors,
            title=None,
        ),
        xaxis_title="Date",
        dragmode="select",
        legend_title="Author",
        modebar_add=["lasso2d", "select2d"],
    )
    fig.update_traces(marker=dict(size=_MARKER_SIZE, opacity=_MARKER_OPACITY))

    event = st.plotly_chart(fig, on_select="rerun", width="stretch")
    has_selection = bool(event and event["selection"] and event["selection"]["points"])

    if has_selection:
        indices = [int(p["customdata"][-1]) for p in event["selection"]["points"]]
        selected = df.iloc[indices]
    else:
        selected = df

    if has_selection:
        total_sel = len(selected)
        authors_sel = int(selected["author"].nunique())

        cols = st.columns(2)
        cols[0].metric("Selected", total_sel)
        cols[1].metric("Authors", authors_sel)

    # Build display table
    valid_cols = [c for c in display_cols if c in df.columns]
    display = selected[valid_cols].copy()

    # Join sentence lists to string
    if "sentence" in display.columns:
        display["sentence"] = pd.Series(display["sentence"]).apply(
            lambda s: " / ".join(s) if isinstance(s, list) else (s or "")
        )

    display["date"] = pd.to_datetime(pd.Series(display["date"])).dt.strftime(_DATE_FMT)
    display.columns = [c.capitalize() for c in display.columns]
    display = pd.DataFrame(display).sort_values("Date").reset_index(drop=True)

    # Table controls
    expanded_view, collapse = expanded_view_controls(expanded_key, collapse_key)
    expand_all = expanded_view and not collapse

    all_field_labels = [f.label for f in fields]
    show_fields = st.multiselect(
        "Fields to show",
        options=all_field_labels,
        key=show_fields_key,
        on_change=on_show_fields,
    )

    field_map = {f.label: f for f in fields}

    if not show_fields:
        return

    if expanded_view:
        _render_as_expanded(display, show_fields, field_map, expand_all)
    else:
        _render_as_table(display, show_fields, field_map)

import io
import logging
from dataclasses import dataclass
from datetime import datetime, date

import pandas as pd

logger = logging.getLogger(__name__)

from src.comments.extract import Comment, extract_comments, extract_paragraphs
from src.redlines.extract import Redline, Move, extract_redlines, extract_moves


# ---------------------------------------------------------------------------
# Document data container
# ---------------------------------------------------------------------------
@dataclass
class DocumentData:
    """
    All raw extracted data from a single .docx file.
    Intended as the single source of truth passed around the app.
    """

    comments: list[Comment]
    redlines: list[Redline]
    moves: list[Move]
    all_paragraphs: list[str]


# ---------------------------------------------------------------------------
# Comment metrics
# ---------------------------------------------------------------------------
@dataclass
class CommentMetrics:
    """Counts derived from a comment list, including replies."""

    total: int
    top_level: int
    replies: int
    resolved: int


def comment_metrics(comments: list[Comment]) -> CommentMetrics:
    top_level = len(comments)
    replies = sum(len(c.replies) for c in comments)
    resolved = sum(1 for c in comments if c.resolved) + sum(
        1 for c in comments for r in c.replies if r.resolved
    )
    total = top_level + replies
    return CommentMetrics(
        total=total,
        top_level=top_level,
        replies=replies,
        resolved=resolved,
    )


def comment_metrics_from_df(c_df: pd.DataFrame) -> CommentMetrics:
    """Compute comment metrics from a filtered comments DataFrame."""
    if c_df.empty:
        return CommentMetrics(total=0, top_level=0, replies=0, resolved=0)

    top_level = int((c_df["kind"] == "comment").sum())
    replies = int((c_df["kind"] == "reply").sum())
    resolved = int(c_df["resolved"].sum())
    total = top_level + replies

    return CommentMetrics(
        total=total,
        top_level=top_level,
        replies=replies,
        resolved=resolved,
    )


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
def _parse_dt(obj, kind: str) -> datetime | None:
    """Parse obj.date as ISO 8601, logging a warning and returning None on failure."""
    try:
        return datetime.fromisoformat(obj.date.rstrip("Z"))
    except ValueError:
        logger.warning(
            "Skipping %s with unparseable date %r (id=%s)", kind, obj.date, obj.id
        )
        return None


def latest_date(comments: list[Comment], redlines: list[Redline]) -> datetime | None:
    """Return the latest date across all comments and redlines."""
    dates = [
        dt
        for obj, kind in [
            *[(c, "comment") for c in comments],
            *[(r, "redline") for r in redlines],
        ]
        if (dt := _parse_dt(obj, kind)) is not None
    ]
    return max(dates) if dates else None


def filter_by_date(df: pd.DataFrame, date_min: date, date_max: date) -> pd.DataFrame:
    """Filter a DataFrame with a 'date' column to the given date range."""
    if df.empty:
        return df
    return df[
        (df["date"].dt.date >= date_min) & (df["date"].dt.date <= date_max)
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Age DataFrames
# ---------------------------------------------------------------------------
def _comment_context_fields(ctx) -> dict:
    if ctx is None:
        return {"selected": None, "sentence": [], "paragraph": None}
    return {
        "selected": ctx.selected_text,
        "sentence": [s.text for s in ctx.sentences],
        "paragraph": ctx.paragraph_text,
    }


def load_document(file_bytes: bytes) -> tuple:
    """Load parsed Word document objects from bytes."""
    comments, version = extract_comments(io.BytesIO(file_bytes))
    redlines, _ = extract_redlines(io.BytesIO(file_bytes))
    moves, _ = extract_moves(io.BytesIO(file_bytes))
    paragraphs = extract_paragraphs(io.BytesIO(file_bytes))
    return (comments, version, redlines, moves, paragraphs)


def build_stats_dfs(
    comments: list[Comment],
    redlines: list[Redline],
    moves: list[Move],
    reference_date: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Prepare age DataFrames and author list for stats pages."""
    reference_date = reference_date or datetime.now()
    c_df = comment_ages_df(comments, reference_date)
    r_df = redline_ages_df(redlines, reference_date)
    m_df = move_ages_df(moves, reference_date)
    all_authors = sorted(c_df["author"].unique().tolist()) if not c_df.empty else []
    return c_df, r_df, m_df, all_authors


def _build_age_grouper(
    items: list[tuple[object, str]],
    row_builder,
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    now = reference_date or datetime.now()
    rows = []

    for obj, kind in items:
        dt = _parse_dt(obj, kind)
        if dt is None:
            continue
        row = row_builder(obj, kind, dt)
        row["age_days"] = (now - dt).days
        row["date"] = dt
        rows.append(row)

    return pd.DataFrame(rows)


def comment_ages_df(
    comments: list[Comment],
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    all_items = [(c, "comment") for c in comments] + [
        (reply, "reply") for c in comments for reply in c.replies
    ]

    return _build_age_grouper(
        all_items,
        lambda c, kind, dt: {
            "author": c.author,
            "resolved": c.resolved,
            "kind": kind,
            "comment": c.text,
            **_comment_context_fields(c.context),
        },
        reference_date,
    )


def _redline_context_fields(ctx) -> dict:
    if ctx is None:
        return {"sentence": [], "paragraph": None}
    return {
        "sentence": [s.text for s in ctx.sentences],
        "paragraph": ctx.paragraph_text,
    }


def redline_ages_df(
    redlines: list[Redline],
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    return _build_age_grouper(
        [(r, "redline") for r in redlines],
        lambda r, kind, dt: {
            "author": r.author,
            "kind": r.kind,
            "text": r.text,
            **_redline_context_fields(r.context),
        },
        reference_date,
    )


def move_ages_df(
    moves: list[Move],
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    return _build_age_grouper(
        [(m, "move") for m in moves],
        lambda m, kind, dt: {
            "author": m.author,
            "text": m.text,
            "from_para_idx": m.from_para_idx,
            "to_para_idx": m.to_para_idx,
            "distance": abs(m.to_para_idx - m.from_para_idx),
        },
        reference_date,
    )

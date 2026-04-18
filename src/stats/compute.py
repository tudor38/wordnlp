import io
import logging
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import datetime, date

import pandas as pd

logger = logging.getLogger(__name__)

from src.comments.extract import (
    Comment,
    DocumentParagraphs,
    _assemble_comments,
    _extract_paragraphs_from_root,
)
from src.redlines.extract import (
    Redline,
    Move,
    _parse_moves_from_root,
    _parse_redlines_from_root,
)
from src.shared import (
    DocxParseError,
    WordVersion,
    _build_parent_map,
    detect_version,
)


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
    paragraphs: DocumentParagraphs
    version: WordVersion


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
    replies = 0
    resolved = 0
    for c in comments:
        if c.resolved:
            resolved += 1
        for r in c.replies:
            replies += 1
            if r.resolved:
                resolved += 1
    return CommentMetrics(
        total=top_level + replies,
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


def load_document(file_bytes: bytes) -> DocumentData:
    """
    Load all raw extracted data from a .docx file in a single pass.

    Opens the zip archive once and parses word/document.xml once. The shared
    XML root and parent map are reused across comment context, redline,
    move, and paragraph extraction, avoiding four redundant parses.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            names = zf.namelist()
            version = detect_version(names)

            def _read(name: str) -> bytes:
                return zf.read(name) if name in names else b""

            document_bytes = _read("word/document.xml")
            comments_bytes = _read("word/comments.xml")
            extended_bytes = _read("word/commentsExtended.xml")
            ids_bytes = _read("word/commentsIds.xml")
    except zipfile.BadZipFile as e:
        raise DocxParseError("Not a valid Word document (.docx).") from e

    try:
        if document_bytes:
            document_root = ET.fromstring(document_bytes)
            parent_map = _build_parent_map(document_root)
            paragraphs = _extract_paragraphs_from_root(document_root, parent_map)
            redlines = _parse_redlines_from_root(document_root, parent_map)
            moves = _parse_moves_from_root(document_root, parent_map)
        else:
            document_root = None
            parent_map = None
            paragraphs = DocumentParagraphs(paragraphs=[], moved_from={})
            redlines = []
            moves = []

        comments = _assemble_comments(
            comments_bytes,
            extended_bytes,
            ids_bytes,
            version,
            document_root,
            parent_map,
        )
    except ET.ParseError as e:
        raise DocxParseError(f"Document XML is malformed: {e}") from e

    return DocumentData(
        comments=comments,
        redlines=redlines,
        moves=moves,
        paragraphs=paragraphs,
        version=version,
    )


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


def _build_ages_df(
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

    return _build_ages_df(
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
    return _build_ages_df(
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
    return _build_ages_df(
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

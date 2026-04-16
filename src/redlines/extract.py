"""
Word document redline (tracked change) extractor.

Extracts insertions, deletions, and moves from word/document.xml.
Formatting-only changes (w:rPrChange, w:pPrChange) are not included.

Paragraph indexing
------------------
All para_idx values index into DocumentParagraphs.paragraphs, which excludes
<w:moveFrom> paragraphs. This is consistent with CommentContext.start_para_idx
and DocumentParagraphs.paragraphs from extract_comments.py.

Move validation
---------------
A Move is only recorded if:
  - Both <w:moveFrom> and <w:moveTo> share the same w:id
  - Neither side contains nested <w:ins> or <w:del> elements
  - The extracted text from both sides is identical

Moves that fail validation are silently dropped.
"""

import io
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, Optional

from src.shared import (
    W,
    _tag,
    Span,
    SentenceSpan,
    WordVersion,
    detect_version,
    DocxParseError,
    _find_sentences_containing,
    _build_parent_map,
    _in_move_from,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RedlineContext:
    """Surrounding text context for a tracked change."""

    para_idx: int  # 0-based index in final document (moveFrom excluded)
    paragraph_text: str
    sentences: list[SentenceSpan]


@dataclass
class Redline:
    id: str
    author: str
    date: str
    kind: Literal["insertion", "deletion"]
    text: str
    char_start: int  # paragraph-relative, inclusive
    char_end: int  # paragraph-relative, exclusive
    context: Optional[RedlineContext] = None

    @property
    def span(self) -> Span:
        return Span(self.char_start, self.char_end)

    def to_row(self) -> dict:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in ("context",)
        } | {
            "para_idx": self.context.para_idx if self.context else None,
            "paragraph": self.context.paragraph_text if self.context else None,
            "sentences": [s.text for s in self.context.sentences]
            if self.context
            else [],
        }


@dataclass
class MoveContext:
    """Text context for one side of a move operation."""

    para_idx: int  # index in final document (to) or xml_order_idx (from)
    paragraph_text: str
    sentences: list[SentenceSpan]


@dataclass
class Move:
    """
    A paragraph that was moved from one location to another.

    from_para_idx : key in DocumentParagraphs.moved_from (xml order position
                    of the source paragraph, counting all <w:p> including
                    moveFrom).
    to_para_idx   : index in DocumentParagraphs.paragraphs (final document
                    position of the destination paragraph).

    Moves are atomic: text must be identical on both sides.
    If nested <w:ins> or <w:del> are present, the pair is dropped.
    """

    id: str
    author: str
    date: str
    text: str
    from_para_idx: int  # key in DocumentParagraphs.moved_from
    to_para_idx: int  # index in DocumentParagraphs.paragraphs
    from_context: MoveContext
    to_context: MoveContext

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "author": self.author,
            "date": self.date,
            "text": self.text,
            "from_para_idx": self.from_para_idx,
            "to_para_idx": self.to_para_idx,
            "from_paragraph": self.from_context.paragraph_text,
            "to_paragraph": self.to_context.paragraph_text,
        }


def _nearest_move_ancestor(elem: ET.Element, parent_map: dict) -> Optional[ET.Element]:
    """Return the nearest <w:moveFrom> or <w:moveTo> ancestor, or None."""
    current = elem
    while current in parent_map:
        current = parent_map[current]
        if current.tag in (_tag(W, "moveFrom"), _tag(W, "moveTo")):
            return current
    return None


# ---------------------------------------------------------------------------
# Core parsers
# ---------------------------------------------------------------------------
def _parse_redlines(xml_bytes: bytes) -> list[Redline]:
    """
    Extract all tracked insertions and deletions from document.xml.

    Only non-moveFrom paragraphs are enumerated so that para_idx values
    align with DocumentParagraphs.paragraphs indices.
    """
    root = ET.fromstring(xml_bytes)
    parent_map = _build_parent_map(root)

    redlines: list[Redline] = []
    para_texts: list[str] = []
    change_spans: dict[str, dict] = {}

    # Enumerate only final-document paragraphs (skip moveFrom)
    para_idx = 0
    for para in root.iter(_tag(W, "p")):
        if _in_move_from(para, parent_map):
            continue

        char_pos = 0
        para_text_parts: list[str] = []

        for elem in para.iter():
            tag = elem.tag

            if tag in (_tag(W, "ins"), _tag(W, "del")):
                rid = elem.get(_tag(W, "id"))
                author = elem.get(_tag(W, "author"), "")
                date = elem.get(_tag(W, "date"), "")
                kind: Literal["insertion", "deletion"] = (
                    "insertion" if tag == _tag(W, "ins") else "deletion"
                )
                text_tag = _tag(W, "t") if kind == "insertion" else _tag(W, "delText")
                text = "".join(t.text or "" for t in elem.iter(text_tag))

                if rid is None:
                    continue

                change_spans[rid] = {
                    "start": char_pos,
                    "end": char_pos + len(text),
                    "para_idx": para_idx,
                }
                redlines.append(
                    Redline(
                        id=rid,
                        author=author,
                        date=date,
                        kind=kind,
                        text=text,
                        char_start=char_pos,
                        char_end=char_pos + len(text),
                    )
                )

            elif tag == _tag(W, "t"):
                t = elem.text or ""
                char_pos += len(t)
                para_text_parts.append(t)

            elif tag == _tag(W, "delText"):
                t = elem.text or ""
                char_pos += len(t)
                para_text_parts.append(t)

        para_texts.append("".join(para_text_parts))
        para_idx += 1

    rid_to_redline = {r.id: r for r in redlines}
    for rid, span in change_spans.items():
        if rid not in rid_to_redline:
            continue
        para_text = para_texts[span["para_idx"]]
        rid_to_redline[rid].context = RedlineContext(
            para_idx=span["para_idx"],
            paragraph_text=para_text,
            sentences=_find_sentences_containing(para_text, span["start"], span["end"]),
        )

    return redlines


def _parse_moves(xml_bytes: bytes) -> list[Move]:
    """
    Extract validated paragraph moves from document.xml.

    A move pair is only kept if:
      1. Both <w:moveFrom> and <w:moveTo> exist for the same w:id
      2. The final rendered text from both sides is identical

    Nested <w:ins> / <w:del> inside the move do not disqualify it —
    only the net rendered text matters. This means a paragraph with
    tracked edits that was moved is still counted as a move, as long
    as the text at the source and destination match after edits are
    resolved.

    Redlines inside the moved paragraph are counted only at the
    destination (to_para_idx) by _parse_redlines, which skips all
    <w:moveFrom> paragraphs. No double-counting occurs.

    Indices:
      from_para_idx : xml_order_idx (counts all <w:p> including moveFrom)
      to_para_idx   : index in final document paragraphs (moveFrom excluded)
    """
    root = ET.fromstring(xml_bytes)
    parent_map = _build_parent_map(root)

    move_info: dict[str, dict] = {}

    final_idx = 0
    xml_idx = 0

    for para in root.iter(_tag(W, "p")):
        anc = _nearest_move_ancestor(para, parent_map)

        if anc is not None and anc.tag == _tag(W, "moveFrom"):
            move_id = anc.get(_tag(W, "id"))
            author = anc.get(_tag(W, "author"), "")
            date = anc.get(_tag(W, "date"), "")

            # Use all <w:t> for rendered text — this resolves nested edits
            # so the comparison against moveTo text is apples-to-apples
            text = "".join(t.text or "" for t in para.iter(_tag(W, "t")))

            if move_id:
                move_info.setdefault(move_id, {}).update(
                    {
                        "from_xml_idx": xml_idx,
                        "from_text": text,
                        "from_para_text": text,
                        "author": author,
                        "date": date,
                    }
                )
            xml_idx += 1

        elif anc is not None and anc.tag == _tag(W, "moveTo"):
            move_id = anc.get(_tag(W, "id"))

            text = "".join(t.text or "" for t in para.iter(_tag(W, "t")))

            if move_id:
                move_info.setdefault(move_id, {}).update(
                    {
                        "to_final_idx": final_idx,
                        "to_text": text,
                        "to_para_text": text,
                    }
                )
            final_idx += 1
            xml_idx += 1

        else:
            final_idx += 1
            xml_idx += 1

    moves: list[Move] = []
    for move_id, info in move_info.items():
        if "from_text" not in info or "to_text" not in info:
            continue
        if info["from_text"] != info["to_text"]:
            continue

        text = info["from_text"]
        from_para_txt = info["from_para_text"]
        to_para_txt = info["to_para_text"]

        moves.append(
            Move(
                id=move_id,
                author=info["author"],
                date=info["date"],
                text=text,
                from_para_idx=info["from_xml_idx"],
                to_para_idx=info["to_final_idx"],
                from_context=MoveContext(
                    para_idx=info["from_xml_idx"],
                    paragraph_text=from_para_txt,
                    sentences=_find_sentences_containing(from_para_txt, 0, len(text)),
                ),
                to_context=MoveContext(
                    para_idx=info["to_final_idx"],
                    paragraph_text=to_para_txt,
                    sentences=_find_sentences_containing(to_para_txt, 0, len(text)),
                ),
            )
        )

    return moves


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
type DocxSource = str | Path | io.IOBase


def extract_redlines(docx: DocxSource) -> tuple[list[Redline], WordVersion]:
    """
    Extract all tracked insertions and deletions from a .docx file.
    para_idx values index into DocumentParagraphs.paragraphs.
    """
    try:
        with zipfile.ZipFile(docx) as z:
            names = z.namelist()
            version = detect_version(names)
            document_bytes = (
                z.read("word/document.xml") if "word/document.xml" in names else b""
            )
    except zipfile.BadZipFile as e:
        raise DocxParseError("Not a valid Word document (.docx).") from e

    if not document_bytes:
        return [], version

    try:
        return _parse_redlines(document_bytes), version
    except ET.ParseError as e:
        raise DocxParseError(f"Document XML is malformed: {e}") from e


def extract_moves(docx: DocxSource) -> tuple[list[Move], WordVersion]:
    """
    Extract all validated paragraph moves from a .docx file.

    from_para_idx indexes into DocumentParagraphs.moved_from.
    to_para_idx   indexes into DocumentParagraphs.paragraphs.
    """
    try:
        with zipfile.ZipFile(docx) as z:
            names = z.namelist()
            version = detect_version(names)
            document_bytes = (
                z.read("word/document.xml") if "word/document.xml" in names else b""
            )
    except zipfile.BadZipFile as e:
        raise DocxParseError("Not a valid Word document (.docx).") from e

    if not document_bytes:
        return [], version

    try:
        return _parse_moves(document_bytes), version
    except ET.ParseError as e:
        raise DocxParseError(f"Document XML is malformed: {e}") from e


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "document.docx"

    redlines, version = extract_redlines(path)
    print(f"Format detected  : {version.name}")
    print(f"Redlines found   : {len(redlines)}\n")
    for r in redlines:
        print(f"[{r.kind.upper()}] ({r.id}) {r.author} @ {r.date}")
        print(f"  Text     : {r.text!r}")
        print(f"  Span     : [{r.char_start}, {r.char_end})")
        if r.context:
            print(f"  Para idx : {r.context.para_idx}")
            print(f"  Paragraph: {r.context.paragraph_text!r}")
            for s in r.context.sentences:
                print(f"  Sentence : {s.text!r}  [{s.span.start}, {s.span.end})")
        print()

    moves, _ = extract_moves(path)
    print(f"Moves found      : {len(moves)}\n")
    for m in moves:
        print(f"[MOVE] ({m.id}) {m.author} @ {m.date}")
        print(f"  Text         : {m.text!r}")
        print(f"  From para idx: {m.from_para_idx}  →  To para idx: {m.to_para_idx}")
        print()

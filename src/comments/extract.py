"""
Word document comment extractor.

Supports three format generations, detected automatically:

  LEGACY   (Word 2007-2010)  word/comments.xml only
                             → author, date, text
  EXTENDED (Word 2013-2016)  + word/commentsExtended.xml
                             → + resolved status, reply threading
  MODERN   (Word 2016+/365)  + word/commentsIds.xml
                             → same, but paraId→commentId mapping is
                               read from commentsIds.xml instead of
                               being inferred from paragraph order

Document context (all versions):
  word/document.xml is always parsed to extract, per comment:
    - start_para_idx : 0-based index in the FINAL document (moveFrom excluded)
    - end_para_idx   : same, for multi-paragraph ranges
    - selected_text  : the exact text the comment is anchored to
    - selected_span  : Span of selected_text within paragraph_text
    - paragraph_text : full text of the containing paragraph(s)
    - sentences      : SentenceSpan objects overlapping the selected range

Paragraph indexing
------------------
All para_idx values index into DocumentParagraphs.paragraphs, which excludes
<w:moveFrom> paragraphs. This keeps indices consistent across comments,
redlines, moves, and extract_paragraphs.

<w:moveFrom> paragraphs are tracked in DocumentParagraphs.moved_from,
keyed by their position in XML order (counting all <w:p> elements).
"""

import io
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


from src.shared import (
    W,
    W14,
    W15,
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
# Document paragraph container
# ---------------------------------------------------------------------------
@dataclass
class DocumentParagraphs:
    """
    All paragraphs extracted from word/document.xml.

    paragraphs : list of paragraph texts in document order, <w:moveFrom>
                 paragraphs excluded.  Index i here is the same as para_idx
                 in CommentContext and RedlineContext.

    moved_from : xml_order_idx → text for paragraphs that were moved away.
                 xml_order_idx counts ALL <w:p> elements in the XML including
                 moveFrom, providing a stable reference to the original
                 position of each moved paragraph.
    """

    paragraphs: list[str]
    moved_from: dict[int, str]  # xml_order_idx → text


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CommentContext:
    """Text context of a comment's anchor point in the document body."""

    start_para_idx: int  # 0-based index in final document
    end_para_idx: int  # 0-based index in final document
    selected_text: str  # exact text between commentRangeStart/End
    selected_span: Span  # span of selected_text within paragraph_text
    paragraph_text: str  # full text of the containing paragraph(s)
    sentences: list[SentenceSpan]  # sentences overlapping the selected range


@dataclass
class Comment:
    id: str
    author: str
    date: str
    text: str
    resolved: bool = False
    parent_id: Optional[str] = None
    replies: list["Comment"] = field(default_factory=list)
    context: Optional[CommentContext] = None


# ---------------------------------------------------------------------------
# Parsers — one per file, version-agnostic internally
# ---------------------------------------------------------------------------
def _parse_comments(xml_bytes: bytes) -> tuple[dict[str, Comment], dict[str, str]]:
    """
    Parse word/comments.xml.

    Returns
    -------
    comments        : {comment_id: Comment}
    para_to_comment : {paraId: comment_id}
    """
    root = ET.fromstring(xml_bytes)

    comments: dict[str, Comment] = {}
    para_to_comment: dict[str, str] = {}

    for c in root.findall(_tag(W, "comment")):
        cid = c.get(_tag(W, "id"))
        author = c.get(_tag(W, "author"), "")
        date = c.get(_tag(W, "date"), "")
        text = "".join(t.text or "" for t in c.iter(_tag(W, "t")))

        if cid is None:
            continue

        comments[cid] = Comment(id=cid, author=author, date=date, text=text)

        first_para = c.find(_tag(W, "p"))
        if first_para is not None:
            para_id = first_para.get(_tag(W14, "paraId"))
            if para_id:
                para_to_comment[para_id] = cid

    return comments, para_to_comment


def _parse_comments_ids(xml_bytes: bytes) -> dict[str, str]:
    root = ET.fromstring(xml_bytes)
    para_to_owner: dict[str, str] = {}
    for ci in root.findall(_tag(W14, "commentId")):
        para_id = ci.get(_tag(W14, "paraId"))
        owner_id = ci.get(_tag(W14, "paraIdOwner"))
        if para_id and owner_id:
            para_to_owner[para_id] = owner_id
    return para_to_owner


def _build_para_to_comment_from_document(xml_bytes: bytes) -> dict[str, str]:
    root = ET.fromstring(xml_bytes)
    para_to_comment: dict[str, str] = {}
    for para in root.iter(_tag(W, "p")):
        para_id = para.get(_tag(W14, "paraId"))
        if para_id is None:
            continue
        ref = para.find(".//" + _tag(W, "commentReference"))
        if ref is not None:
            cid = ref.get(_tag(W, "id"))
            if cid:
                para_to_comment[para_id] = cid
    return para_to_comment


def _apply_extended(
    comments: dict[str, Comment],
    para_to_comment: dict[str, str],
    xml_bytes: bytes,
) -> None:
    root = ET.fromstring(xml_bytes)

    for ce in root.findall(_tag(W15, "commentEx")):
        para_id = ce.get(_tag(W15, "paraId"))
        parent_id = ce.get(_tag(W15, "paraIdParent"))
        done = ce.get(_tag(W15, "done"), "0") == "1"

        if para_id is None:
            continue

        cid = para_to_comment.get(para_id)

        # LibreOffice fallback: paraId is the comment w:id encoded as
        # a little-endian 32-bit hex string e.g. "01000000" → id "1"
        if cid is None:
            try:
                cid = str(int.from_bytes(bytes.fromhex(para_id), "little"))
            except (ValueError, TypeError):
                continue
            if cid not in comments:
                continue

        comments[cid].resolved = done

        if parent_id:
            parent_cid = para_to_comment.get(parent_id)
            if parent_cid is None:
                try:
                    parent_cid = str(int.from_bytes(bytes.fromhex(parent_id), "little"))
                except (ValueError, TypeError):
                    parent_cid = None
            if parent_cid and parent_cid in comments:
                comments[cid].parent_id = parent_cid


# ---------------------------------------------------------------------------
# Document context — selected text, paragraph, and sentences
# ---------------------------------------------------------------------------
def _parse_document_context(xml_bytes: bytes) -> dict[str, CommentContext]:
    """
    Parse word/document.xml and extract a CommentContext for each comment id.

    Only non-moveFrom paragraphs are enumerated so that para_idx values
    align with DocumentParagraphs.paragraphs indices.
    """
    root = ET.fromstring(xml_bytes)
    parent_map = _build_parent_map(root)

    # Only paragraphs that appear in the final document
    para_elements: list[ET.Element] = [
        p for p in root.iter(_tag(W, "p")) if not _in_move_from(p, parent_map)
    ]

    open_ranges: dict[str, dict] = {}
    completed: dict[str, dict] = {}
    para_texts: list[str] = []

    for para_idx, para in enumerate(para_elements):
        char_pos = 0
        para_text_parts: list[str] = []

        for acc in open_ranges.values():
            acc["sel_chunks"].append("\n")

        for elem in para.iter():
            tag = elem.tag

            if tag == _tag(W, "commentRangeStart"):
                cid = elem.get(_tag(W, "id"))
                if cid:
                    open_ranges[cid] = {
                        "start_para": para_idx,
                        "start_char": char_pos,
                        "sel_chunks": [],
                    }

            elif tag == _tag(W, "commentRangeEnd"):
                cid = elem.get(_tag(W, "id"))
                if cid and cid in open_ranges:
                    acc = open_ranges.pop(cid)
                    completed[cid] = {
                        "selected": "".join(acc["sel_chunks"]),
                        "start_para": acc["start_para"],
                        "start_char": acc["start_char"],
                        "end_para": para_idx,
                        "end_char": char_pos,
                    }

            elif tag == _tag(W, "t"):
                text = elem.text or ""
                char_pos += len(text)
                para_text_parts.append(text)
                for acc in open_ranges.values():
                    acc["sel_chunks"].append(text)

        para_texts.append("".join(para_text_parts))

    contexts: dict[str, CommentContext] = {}

    for cid, info in completed.items():
        sp = info["start_para"]
        ep = info["end_para"]

        if sp == ep:
            para_text = para_texts[sp]
            sel_start = info["start_char"]
            sel_end = info["end_char"]
        else:
            para_text = "\n".join(para_texts[sp : ep + 1])
            sel_start = info["start_char"]
            offset_to_ep = sum(len(para_texts[i]) + 1 for i in range(sp, ep))
            sel_end = offset_to_ep + info["end_char"]

        contexts[cid] = CommentContext(
            start_para_idx=sp,
            end_para_idx=ep,
            selected_text=info["selected"],
            selected_span=Span(sel_start, sel_end),
            paragraph_text=para_text,
            sentences=_find_sentences_containing(para_text, sel_start, sel_end),
        )

    return contexts


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------
def _build_tree(comments: dict[str, Comment]) -> list[Comment]:
    """Nest replies under their parents; return only top-level comments."""
    top_level: list[Comment] = []
    for c in comments.values():
        if c.parent_id and c.parent_id in comments:
            comments[c.parent_id].replies.append(c)
        else:
            top_level.append(c)
    return top_level


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
type DocxSource = str | Path | io.IOBase


def extract_comments(docx: DocxSource) -> tuple[list[Comment], WordVersion]:
    """
    Extract all comments from a .docx file.

    Returns a list of top-level Comment objects (replies nested inside
    Comment.replies) and the detected WordVersion.
    """
    try:
        with zipfile.ZipFile(docx) as zf:
            names = zf.namelist()
            version = detect_version(names)
            comments_bytes = (
                zf.read("word/comments.xml") if "word/comments.xml" in names else b""
            )
            extended_bytes = (
                zf.read("word/commentsExtended.xml")
                if "word/commentsExtended.xml" in names
                else b""
            )
            ids_bytes = (
                zf.read("word/commentsIds.xml")
                if "word/commentsIds.xml" in names
                else b""
            )
            document_bytes = (
                zf.read("word/document.xml") if "word/document.xml" in names else b""
            )
    except zipfile.BadZipFile as e:
        raise DocxParseError("Not a valid Word document (.docx).") from e

    if not comments_bytes:
        return [], version

    try:
        comments, para_to_comment = _parse_comments(comments_bytes)

        if version == WordVersion.MODERN:
            para_to_owner = _parse_comments_ids(ids_bytes)
            for para_id, owner_para_id in para_to_owner.items():
                if para_id not in para_to_comment and owner_para_id in para_to_comment:
                    para_to_comment[para_id] = para_to_comment[owner_para_id]
            _apply_extended(comments, para_to_comment, extended_bytes)

        elif version == WordVersion.EXTENDED:
            if document_bytes:
                para_to_comment.update(
                    _build_para_to_comment_from_document(document_bytes)
                )
            _apply_extended(comments, para_to_comment, extended_bytes)

        if document_bytes:
            contexts = _parse_document_context(document_bytes)
            for cid, ctx in contexts.items():
                if cid in comments:
                    comments[cid].context = ctx
    except ET.ParseError as e:
        raise DocxParseError(f"Document XML is malformed: {e}") from e

    return _build_tree(comments), version


def extract_paragraphs(docx: DocxSource) -> DocumentParagraphs:
    """
    Extract all paragraphs from the document, returning a DocumentParagraphs
    object that separates the final document paragraphs from moved-away ones.

    paragraphs  : final document order, <w:moveFrom> paragraphs excluded.
                  Index i aligns with para_idx in CommentContext and
                  RedlineContext.

    moved_from  : xml_order_idx → text for paragraphs that were moved away.
                  xml_order_idx counts ALL <w:p> in XML order (including
                  moveFrom) and provides a stable reference to original
                  paragraph positions.
    """
    try:
        with zipfile.ZipFile(docx) as z:
            if "word/document.xml" not in z.namelist():
                return DocumentParagraphs(paragraphs=[], moved_from={})
            root = ET.fromstring(z.read("word/document.xml"))
    except zipfile.BadZipFile as e:
        raise DocxParseError("Not a valid Word document (.docx).") from e
    except ET.ParseError as e:
        raise DocxParseError(f"Document XML is malformed: {e}") from e

    parent_map = _build_parent_map(root)

    paragraphs: list[str] = []
    moved_from: dict[int, str] = {}

    for xml_idx, para in enumerate(root.iter(_tag(W, "p"))):
        text = "".join(t.text or "" for t in para.iter(_tag(W, "t")))
        if _in_move_from(para, parent_map):
            moved_from[xml_idx] = text
        else:
            paragraphs.append(text)

    return DocumentParagraphs(paragraphs=paragraphs, moved_from=moved_from)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "document.docx"

    comments, version = extract_comments(path)

    print(f"Format detected : {version.name}")
    print(f"Comments found  : {len(comments)}\n")

    for comment in comments:
        status = "RESOLVED" if comment.resolved else "OPEN"
        print(f"[{status}] ({comment.id}) {comment.author} @ {comment.date}")
        print(f"  Comment  : {comment.text}")
        if comment.context:
            print(
                f"  Para idx : [{comment.context.start_para_idx}, {comment.context.end_para_idx}]"
            )
            print(
                f"  Selected : {comment.context.selected_text!r}  [{comment.context.selected_span.start}, {comment.context.selected_span.end})"
            )
            print(f"  Paragraph: {comment.context.paragraph_text!r}")
            for s in comment.context.sentences:
                print(f"  Sentence : {s.text!r}  [{s.span.start}, {s.span.end})")
        for reply in comment.replies:
            r_status = "RESOLVED" if reply.resolved else "OPEN"
            print(f"  ↳ [{r_status}] ({reply.id}) {reply.author} @ {reply.date}")
            print(f"      Comment  : {reply.text}")
            if reply.context:
                print(
                    f"      Para idx : [{reply.context.start_para_idx}, {reply.context.end_para_idx}]"
                )
                print(f"      Selected : {reply.context.selected_text!r}")
        print()

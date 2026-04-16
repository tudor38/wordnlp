"""
Shared utilities for Word document parsing (comments and redlines modules).
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum, auto

import spacy


class DocxParseError(ValueError):
    """Raised when a .docx file cannot be opened or its XML is malformed."""


# ---------------------------------------------------------------------------
# XML Namespaces
# ---------------------------------------------------------------------------
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W14 = "http://schemas.microsoft.com/office/word/2010/wordml"
W15 = "http://schemas.microsoft.com/office/word/2012/wordml"


def _tag(ns: str, local: str) -> str:
    return f"{{{ns}}}{local}"


# ---------------------------------------------------------------------------
# Span types
# ---------------------------------------------------------------------------
@dataclass
class Span:
    """A [start, end) character range, paragraph-relative."""

    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and self.end > other.start


@dataclass
class SentenceSpan:
    """A sentence and its paragraph-relative character span."""

    text: str
    span: Span


# ---------------------------------------------------------------------------
# Word format version
# ---------------------------------------------------------------------------
class WordVersion(Enum):
    LEGACY = auto()  # comments.xml only
    EXTENDED = auto()  # + commentsExtended.xml
    MODERN = auto()  # + commentsIds.xml


def detect_version(zip_names: list[str]) -> WordVersion:
    """
    Infer the Word format generation from which XML files are present.

    commentsIds.xml was introduced in Word 2016 alongside the modern
    threaded-comments UI; commentsExtended.xml appeared in Word 2013.
    """
    has_extended = "word/commentsExtended.xml" in zip_names
    has_ids = "word/commentsIds.xml" in zip_names
    if has_extended and has_ids:
        return WordVersion.MODERN
    if has_extended:
        return WordVersion.EXTENDED
    return WordVersion.LEGACY


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------
_nlp = spacy.blank("en")
_nlp.add_pipe("sentencizer")


def _find_sentences_containing(
    text: str, sel_start: int, sel_end: int
) -> list[SentenceSpan]:
    if not text or sel_start >= sel_end:
        return []
    doc = _nlp(text)
    return [
        SentenceSpan(
            text=sent.text.strip(),
            span=Span(sent.start_char, sent.end_char),
        )
        for sent in doc.sents
        if sent.start_char < sel_end and sent.end_char > sel_start
    ]


# ---------------------------------------------------------------------------
# XML parent-map helpers
# ---------------------------------------------------------------------------
def _build_parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
    return {child: parent for parent in root.iter() for child in parent}


def _in_move_from(elem: ET.Element, parent_map: dict[ET.Element, ET.Element]) -> bool:
    """Return True if elem is a descendant of a <w:moveFrom> element."""
    move_from = _tag(W, "moveFrom")
    move_to = _tag(W, "moveTo")
    current = elem
    while current in parent_map:
        current = parent_map[current]
        if current.tag == move_from:
            return True
        if current.tag == move_to:
            return False
    return False

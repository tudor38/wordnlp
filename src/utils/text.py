"""
Shared text processing utilities.
"""

import math
import regex as re
from collections.abc import Sequence
from html import escape as _escape

import numpy as np
import Stemmer as _PyStemmer
from spacy.lang.en.stop_words import STOP_WORDS

from src.stats.config import CFG

_stemmer = _PyStemmer.Stemmer("english")

TOPIC_PALETTE = [
    "#ffe066",
    "#b5ead7",
    "#b5d5ff",
    "#e8b5ff",
    "#ffb5b5",
    "#ffd9b5",
    "#b5ffe4",
    "#c9ffb5",
    "#ffb5e8",
    "#b5f0ff",
]


def highlight_term(text: str, query: str, color: str = "") -> str:
    """Return HTML with query matches wrapped in <mark>. Non-match text is HTML-escaped."""
    if not query.strip():
        return _escape(text)
    style = f' style="background:{color}"' if color else ""
    safe = _escape(text)
    return re.compile(re.escape(_escape(query)), flags=re.IGNORECASE).sub(
        lambda m: f"<mark{style}>{m.group(0)}</mark>", safe
    )


def highlight_query_tokens(text: str, query: str, color: str = "") -> str:
    """Return HTML with stemmed query tokens wrapped in <mark>. Non-match text is HTML-escaped."""
    stemmed_terms = {
        _stemmer.stemWord(term) for term in tokenize(query) if term not in STOP_WORDS
    }
    if not stemmed_terms:
        return _escape(text)
    style = f' style="background:{color}"' if color else ""
    safe = _escape(text)

    def _replace(m: re.Match) -> str:
        word = m.group(0)
        if _stemmer.stemWord(word.lower()) in stemmed_terms:
            return f"<mark{style}>{word}</mark>"
        return word

    return re.sub(r"\b\w+\b", _replace, safe)


def highlight_topic_keywords(html_text: str, topic_label: str, color: str) -> str:
    """Wrap topic keyword matches in <mark> within already-HTML text.

    Operates on HTML output from highlight_term / highlight_query_tokens.
    Skips existing HTML tags so attributes are never corrupted.
    """
    if not topic_label or topic_label == "Noise":
        return html_text
    stemmed_keywords = {
        _stemmer.stemWord(w.lower())
        for w in topic_label.split()
        if w.lower() not in STOP_WORDS and len(w) > 2
    }
    if not stemmed_keywords:
        return html_text

    def _replace(m: re.Match) -> str:
        word = m.group(0)
        if _stemmer.stemWord(word.lower()) in stemmed_keywords:
            return f'<mark style="background:{color}">{word}</mark>'
        return word

    # Split into alternating [text_node, tag, text_node, tag, ...] segments.
    # Only apply word highlighting to text nodes (segments that don't start with "<").
    parts = re.split(r"(<[^>]+>)", html_text)
    return "".join(
        re.sub(r"\b\w+\b", _replace, part) if not part.startswith("<") else part
        for part in parts
    )


def highlight_regex(text: str, pattern: re.Pattern, color: str = "") -> str:
    """Return HTML with regex matches wrapped in <mark>. All text is HTML-escaped."""
    style = f' style="background:{color}"' if color else ""
    parts: list[str] = []
    last = 0
    try:
        iterator = pattern.finditer(text, timeout=1.0)
    except TypeError:
        iterator = pattern.finditer(text)

    for m in iterator:
        parts.append(_escape(text[last : m.start()]))
        parts.append(f"<mark{style}>{_escape(m.group(0))}</mark>")
        last = m.end()
    parts.append(_escape(text[last:]))
    return "".join(parts)


# ---------------------------------------------------------------------------
# annotated_text-compatible helpers (no HTML, no unsafe_allow_html)
# ---------------------------------------------------------------------------
def annotate_term(text: str, query: str, color: str = "") -> list:
    if not query.strip():
        return [text]
    parts = []
    last = 0
    for m in re.compile(re.escape(query), flags=re.IGNORECASE).finditer(text):
        if m.start() > last:
            parts.append(text[last : m.start()])
        parts.append((m.group(0), "", color))
        last = m.end()
    if last < len(text):
        parts.append(text[last:])
    return parts or [text]


def annotate_regex(text: str, pattern: re.Pattern, color: str = "") -> list:
    parts = []
    last = 0
    try:
        try:
            iterator = pattern.finditer(text, timeout=1.0)
        except TypeError:
            iterator = pattern.finditer(text)
        for m in iterator:
            if m.start() > last:
                parts.append(text[last : m.start()])
            parts.append((m.group(0), "", color))
            last = m.end()
    except TimeoutError:
        return [text]
    if last < len(text):
        parts.append(text[last:])
    return parts or [text]


def annotate_query_tokens(text: str, query: str, color: str = "") -> list:
    stemmed_terms = {
        _stemmer.stemWord(term) for term in tokenize(query) if term not in STOP_WORDS
    }
    if not stemmed_terms:
        return [text]
    parts = []
    last = 0
    for m in re.finditer(r"\b\w+\b", text):
        word = m.group(0)
        if _stemmer.stemWord(word.lower()) in stemmed_terms:
            if m.start() > last:
                parts.append(text[last : m.start()])
            parts.append((word, "", color))
            last = m.end()
    if last < len(text):
        parts.append(text[last:])
    return parts or [text]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def bm25_scores(
    docs: Sequence[str],
    query: str,
    k1: float = CFG.multi_doc_search.bm25_k1,
    b: float = CFG.multi_doc_search.bm25_b,
) -> np.ndarray:
    query_terms = tokenize(query)
    if not query_terms:
        return np.zeros(len(docs), dtype=float)

    tokenized_docs = [tokenize(doc) for doc in docs]
    doc_lens = np.array([len(tokens) for tokens in tokenized_docs], dtype=float)
    avgdl = float(doc_lens.mean()) if len(doc_lens) else 0.0
    n_docs = len(docs)

    doc_freq: dict[str, int] = {}
    for tokens in tokenized_docs:
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    scores = np.zeros(n_docs, dtype=float)
    for idx, tokens in enumerate(tokenized_docs):
        tf: dict[str, int] = {}
        for term in tokens:
            tf[term] = tf.get(term, 0) + 1
        dl = max(doc_lens[idx], 1.0)
        for term in query_terms:
            if term not in tf:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            freq = tf[term]
            denom = freq + k1 * (1 - b + b * dl / max(avgdl, 1.0))
            scores[idx] += idf * (freq * (k1 + 1)) / max(denom, 1e-9)
    if scores.size:
        max_score = scores.max()
        if max_score > 0:
            scores /= max_score
    return scores

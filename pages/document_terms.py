import hashlib
import html
import io
import regex as re

import pandas as pd
import spacy
import streamlit as st

from src.app_state import (
    KEY_DT_CACHE_KEY,
    KEY_DT_DATES,
    KEY_DT_DEFS,
    KEY_DT_MONEY,
    KEY_DT_NUMBERS,
    KEY_DT_PARTIES,
    KEY_DT_SPACY_MODEL,
    make_store,
)
from src.comments.extract import extract_paragraphs
from src.utils.models import get_spacy_nlp
from src.utils.page import expanded_view_controls, require_document
from annotated_text import annotated_text
from src.utils.text import annotate_regex


@st.cache_data(show_spinner=False, max_entries=5)
def _extract_definitions(paragraphs: tuple[str, ...]) -> pd.DataFrame:
    patterns = [
        # "Term" means / shall mean / has the meaning
        re.compile(
            r'["\u201c\u2018](?P<term>[^"\u201d\u2019]{1,80})["\u201d\u2019]\s+(?:shall\s+)?(?:mean|have\s+the\s+meaning)',
            re.IGNORECASE,
        ),
        # defined as / referred to as / referred to herein as
        re.compile(
            r"\b(?P<term>[A-Z][A-Za-z\s]{1,60})\b\s+(?:is\s+)?(?:defined\s+as|referred\s+to\s+(?:herein\s+)?as)",
            re.IGNORECASE,
        ),
        # (hereinafter "Term"), (the "Term"), (together, the "Parties"), (each, a "Party"), etc.
        re.compile(
            r'\([^"\u201c\u2018]{0,50}["\u201c\u2018](?P<term>[^"\u201d\u2019]{1,80})["\u201d\u2019]\)',
            re.IGNORECASE,
        ),
        # (together, the Parties), (hereinafter, the Disclosing Party) — no quotes, Title Case term
        re.compile(
            r'\([^")]{0,60}(?:the|an?)\s+(?P<term>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s*[,;.]?\s*\)',
        ),
    ]
    rows = []
    seen = set()
    for idx, para in enumerate(paragraphs):
        for pattern in patterns:
            for m in pattern.finditer(para):
                term = m.group("term").strip()
                if term.lower() in seen:
                    continue
                seen.add(term.lower())
                rows.append({"Para": idx, "Term": term, "Context": para})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=5)
def _extract_entities(
    paragraphs: tuple[str, ...],
    labels: tuple[str, ...],
    model_name: str = "en_core_web_sm",
) -> pd.DataFrame:
    nlp = get_spacy_nlp(model_name)
    rows = []
    for idx, para in enumerate(paragraphs):
        doc = nlp(para)
        for ent in doc.ents:
            if ent.label_ in labels:
                rows.append(
                    {
                        "Para": idx,
                        "Value": ent.text,
                        "Type": ent.label_,
                        "Context": para,
                    }
                )
    return pd.DataFrame(rows)


_MONEY_RE = re.compile(
    r"(?:"
    r"(?:USD|EUR|GBP|CHF|CAD|AUD|JPY)\s+\d{1,3}(?:,\d{3})*(?:\.\d+)?"  # USD 8,500
    r"|[\$€£]\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?"  # $8,500 / € 100,000
    r"|\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:USD|EUR|GBP|CHF)"  # 8,500 USD
    r"|\d+(?:\.\d+)?\s*(?:%|percent|per\s+cent)"  # 18%, 1.5 %
    r")",
    re.IGNORECASE,
)


@st.cache_data(show_spinner=False, max_entries=5)
def _extract_money(paragraphs: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for idx, para in enumerate(paragraphs):
        seen: set[str] = set()
        for m in _MONEY_RE.finditer(para):
            val = m.group(0).strip()
            if val not in seen:
                seen.add(val)
                rows.append({"Para": idx, "Value": val, "Context": para})
    return pd.DataFrame(rows)


_FALSE_DATE_PATTERNS = re.compile(
    r"^("
    r"[Ss]ection\s+"  # "Section 4.3.1"
    r"|\d{1,4}\.\d+"  # "4.3.1", "23.1"
    r"|\d{1,3}-\d{3,}"  # "23-1202", "15-123456"
    r"|\d{1,3}-\d+-[a-zA-Z]"  # "23-1202(c)"
    r"|§\s*\d+"  # "§ 23"
    r"|\d+\([a-zA-Z]\)"  # "4(b)"
    r"|\d+\)\s*(years?|months?|days?|weeks?|hours?)"  # "30) days", "2) years" (spaCy picks up the tail of parenthetical numbers)
    r"|\d+\s+(years?|months?|days?|weeks?|hours?)$"  # "30 days", "12 months" — plain durations, not calendar dates
    r")"
)

_NUMERIC_PATTERN = re.compile(
    r"(\$|€|£|USD|EUR|GBP|%|percent"
    r"|\b\d[\d,]*\.?\d*\s*(dollars?|cents?|million|billion|thousand|hundred)"
    r"|\b\d+\s*(days?|months?|years?|weeks?|hours?|minutes?|times?|attempts?|copies?|users?|seats?|units?)"
    r"|\bat\s+least\s+\d|\bup\s+to\s+\d|\bno\s+more\s+than\s+\d|\bno\s+less\s+than\s+\d"
    r")",
    re.IGNORECASE,
)
_SECTION_REF = re.compile(
    r"^\d{1,4}\.\d|\d{1,3}-\d{3,}|^§|^\d+\([a-zA-Z]\)|^[Ss]ection\s",
)


def _clean_amounts(amounts_df: pd.DataFrame) -> pd.DataFrame:
    mask = ~amounts_df["Value"].apply(lambda v: bool(_SECTION_REF.search(v)))
    return amounts_df[mask].reset_index(drop=True)


def _clean_dates(dates_df: pd.DataFrame) -> pd.DataFrame:
    mask = ~dates_df["Value"].str.strip().apply(
        lambda v: bool(_FALSE_DATE_PATTERNS.match(v))
    )
    return dates_df[mask].reset_index(drop=True)


def _render_expanded(
    df: pd.DataFrame, heading_col: str, expand_all: bool = True
) -> None:
    for _, row in df.iterrows():
        term = str(row[heading_col])
        context_parts = annotate_regex(
            row["Context"],
            re.compile(
                r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])",
                re.IGNORECASE,
            ),
        )
        with st.expander(f"#{row['Para']} — {html.escape(term)}", expanded=expand_all):
            annotated_text(*context_parts)


@st.cache_data(show_spinner=False, max_entries=5)
def _get_paragraphs(file_bytes: bytes) -> tuple[str, ...]:
    doc = extract_paragraphs(io.BytesIO(file_bytes))
    return tuple(p.strip() for p in doc.paragraphs if p and p.strip())


# ---------------------------------------------------------------------------
# Expanded-view state — persists across navigation via permanent keys
# ---------------------------------------------------------------------------
_DT_SECTIONS = ("defs", "dates", "parties", "money", "numbers")

# Init permanent keys only. Widget keys are never written to manually —
# value= seeds them on first render (navigation / first load) and Streamlit
# manages them on subsequent same-page reruns via session state.
for _s in _DT_SECTIONS:
    st.session_state.setdefault(f"dt_{_s}_expanded", False)
    st.session_state.setdefault(f"dt_{_s}_collapse", False)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
file_bytes = require_document()

paragraphs = _get_paragraphs(file_bytes)

if not paragraphs:
    st.warning("No usable text found in the document.")
    st.stop()

with st.sidebar:
    _all_models = [
        "en_core_web_sm",
        "en_core_web_md",
        "en_core_web_lg",
    ]
    _installed = [m for m in _all_models if spacy.util.is_package(m)]
    _default = next(
        (
            m
            for m in ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]
            if m in _installed
        ),
        _installed[0] if _installed else None,
    )
    _saved = st.session_state.get(KEY_DT_SPACY_MODEL, _default)
    if _saved in _installed:
        _index = _installed.index(_saved)
    elif _default in _installed:
        _index = _installed.index(_default)
    else:
        _index = 0
    spacy_model = st.selectbox(
        "spaCy model",
        options=_installed,
        index=_index,
        help="Larger models improve entity recognition quality. the **md/lg** are a good balance.\n\nThe **trf** model is most accurate but not included because Streamlit free hosting cannot handle the workload.",
    )
    st.session_state[KEY_DT_SPACY_MODEL] = spacy_model

st.subheader("Document Terms")
st.markdown(
    "Extracts key terms from the document: defined terms, dates, parties and entities, monetary values, and numbers."
)

# Recompute only when the document or model changes
_cache_key = hashlib.md5(file_bytes).hexdigest() + "|" + spacy_model

if st.session_state.get(KEY_DT_CACHE_KEY) != _cache_key:
    defs_df = pd.DataFrame()
    dates_df = pd.DataFrame()
    parties_df = pd.DataFrame()
    money_df = pd.DataFrame()
    numbers_df = pd.DataFrame()
    try:
        with st.spinner("Extracting document terms…"):
            defs_df = _extract_definitions(paragraphs)
            dates_df = _clean_dates(
                _extract_entities(paragraphs, ("DATE",), spacy_model).drop(
                    columns="Type", errors="ignore"
                )
            )
            parties_df = _extract_entities(
                paragraphs,
                ("LAW", "PERSON", "ORG", "GPE", "LOC", "PRODUCT"),
                spacy_model,
            )
            money_df = _extract_money(paragraphs)
            numbers_df = _clean_amounts(
                _extract_entities(
                    paragraphs, ("QUANTITY", "CARDINAL"), spacy_model
                ).drop(columns="Type", errors="ignore")
            )
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    st.session_state.update(
        {
            KEY_DT_CACHE_KEY: _cache_key,
            KEY_DT_DEFS: defs_df,
            KEY_DT_DATES: dates_df,
            KEY_DT_PARTIES: parties_df,
            KEY_DT_MONEY: money_df,
            KEY_DT_NUMBERS: numbers_df,
        }
    )
else:
    defs_df = st.session_state[KEY_DT_DEFS]
    dates_df = st.session_state[KEY_DT_DATES]
    parties_df = st.session_state[KEY_DT_PARTIES]
    money_df = st.session_state[KEY_DT_MONEY]
    numbers_df = st.session_state[KEY_DT_NUMBERS]

tab_defs, tab_dates, tab_parties, tab_money, tab_numbers = st.tabs(
    ["Definitions", "Dates", "Parties & Entities", "Money", "Numbers"]
)

# The four standard tabs share the same render pattern.
# Parties has extra controls (type filter, dedup) and is handled separately below.
_STANDARD_TABS = [
    (
        tab_defs,
        defs_df,
        "defs",
        "No definitions found.",
        "defined terms",
        "Term",
        {
            "Para": st.column_config.NumberColumn("#", width="small"),
            "Term": st.column_config.TextColumn("Term", width="medium"),
            "Context": st.column_config.TextColumn("Context", width="large"),
        },
    ),
    (
        tab_dates,
        dates_df,
        "dates",
        "No dates found.",
        "dates",
        "Value",
        {
            "Para": st.column_config.NumberColumn("#", width="small"),
            "Value": st.column_config.TextColumn("Date", width="medium"),
            "Context": st.column_config.TextColumn("Context", width="large"),
        },
    ),
    (
        tab_money,
        money_df,
        "money",
        "No monetary values found.",
        "monetary values",
        "Value",
        {
            "Para": st.column_config.NumberColumn("#", width="small"),
            "Value": st.column_config.TextColumn("Amount", width="medium"),
            "Context": st.column_config.TextColumn("Context", width="large"),
        },
    ),
    (
        tab_numbers,
        numbers_df,
        "numbers",
        "No numbers found.",
        "numbers",
        "Value",
        {
            "Para": st.column_config.NumberColumn("#", width="small"),
            "Value": st.column_config.TextColumn("Value", width="medium"),
            "Context": st.column_config.TextColumn("Context", width="large"),
        },
    ),
]

for (
    _tab,
    _df,
    _section,
    _empty_msg,
    _count_label,
    _heading_col,
    _col_config,
) in _STANDARD_TABS:
    with _tab:
        if _df.empty:
            st.info(_empty_msg)
        else:
            st.caption(f"{len(_df)} {_count_label}")
            _expanded, _collapse = expanded_view_controls(
                f"dt_{_section}_expanded", f"dt_{_section}_collapse"
            )
            if not _expanded:
                st.dataframe(
                    _df, width="stretch", hide_index=True, column_config=_col_config
                )
            else:
                _render_expanded(_df, _heading_col, expand_all=not _collapse)

with tab_parties:
    if parties_df.empty:
        st.info("No parties or entities found.")
    else:
        _TYPE_ORDER = ["LAW", "PERSON", "ORG", "GPE", "LOC", "PRODUCT"]
        available = [t for t in _TYPE_ORDER if t in parties_df["Type"].unique()]
        type_filter = st.pills(
            "Type",
            options=available,
            default=None,
            key="dt_party_filter",
            label_visibility="collapsed",
            selection_mode="single",
        )
        display_df = (
            parties_df[parties_df["Type"] == type_filter] if type_filter else parties_df
        )
        _col_dedup, _col_v, _col_c = st.columns([1, 1, 1])
        dedup = _col_dedup.toggle("Unique only", value=False, key="dt_parties_dedup")
        _expanded = _col_v.toggle(
            "Show expanded view",
            value=st.session_state.get("dt_parties_expanded", False),
            key="_dt_parties_expanded",
            disabled=dedup,
            on_change=make_store("dt_parties_expanded"),
        )
        if _expanded and not dedup:
            _collapse = _col_c.toggle(
                "Collapse all",
                value=st.session_state.get("dt_parties_collapse", False),
                key="_dt_parties_collapse",
                on_change=make_store("dt_parties_collapse"),
            )
        else:
            _collapse = False

        if dedup:
            table_df = display_df.drop_duplicates(subset="Value")[
                ["Value", "Type"]
            ].reset_index(drop=True)
            st.caption(f"{len(table_df)} unique entities")
            st.dataframe(
                table_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Value": st.column_config.TextColumn("Entity", width="medium"),
                    "Type": st.column_config.TextColumn("Type", width="small"),
                },
            )
        elif not _expanded:
            st.caption(f"{len(display_df)} entities")
            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Para": st.column_config.NumberColumn("#", width="small"),
                    "Value": st.column_config.TextColumn("Entity", width="medium"),
                    "Type": st.column_config.TextColumn("Type", width="small"),
                    "Context": st.column_config.TextColumn("Context", width="large"),
                },
            )
        else:
            st.caption(f"{len(display_df)} entities")
            _render_expanded(display_df, "Value", expand_all=not _collapse)

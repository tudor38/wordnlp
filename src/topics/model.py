"""
Topic modelling utilities: text preparation, BERTopic fitting, label generation.

All functions here are pure computation (no Streamlit UI calls). The cache
decorators use Streamlit's caching layer but do not render anything.
"""

import re
from collections.abc import Sequence

import numpy as np
import streamlit as st
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from src.utils.models import get_sentence_transformer
from src.utils.text import TOPIC_PALETTE


def clean_docs(paragraphs: Sequence[str], min_chars: int) -> list[str]:
    docs: list[str] = []
    for paragraph in paragraphs:
        text = (paragraph or "").strip()
        if len(text) >= min_chars:
            docs.append(text)
    return docs


def paragraphs_to_sentences(paragraphs: Sequence[str], min_chars: int) -> list[str]:
    sentences: list[str] = []
    for paragraph in paragraphs:
        text = (paragraph or "").strip()
        if not text:
            continue
        parts = re.split(r"(?<=[.!?])\s+", text)
        for part in parts:
            sent = part.strip()
            if len(sent) >= min_chars:
                sentences.append(sent)
    return sentences


def topic_labels(topic_model: BERTopic, topics: list[int]) -> np.ndarray:
    labels: list[str] = []
    for topic in topics:
        if topic == -1:
            labels.append("Noise")
            continue
        words = topic_model.get_topic(topic) or []
        top_words = [word for word, _ in words[:3]]
        if top_words:
            labels.append(" ".join(top_words))
        else:
            labels.append(f"Topic {topic}")
    return np.array(labels, dtype=object)


def default_granularity(n_docs: int) -> tuple[int, int, int]:
    highlevel = max(4, n_docs // 4)
    midlevel = max(3, n_docs // 10)
    lowlevel = 2
    # Enforce strictly decreasing so all three levels stay distinct after set deduplication
    midlevel = min(midlevel, highlevel - 1)
    if midlevel <= lowlevel:
        midlevel = lowlevel + 1
    if highlevel <= midlevel:
        highlevel = midlevel + 1
    return highlevel, midlevel, lowlevel


def topic_color_map(label_layer: np.ndarray) -> dict[str, str]:
    unique = [l for l in dict.fromkeys(label_layer) if l != "Noise"]
    return {
        label: TOPIC_PALETTE[i % len(TOPIC_PALETTE)] for i, label in enumerate(unique)
    }


@st.cache_resource(show_spinner=False, max_entries=3)
def embed_docs(docs: tuple[str, ...], embedding_model_name: str) -> np.ndarray:
    encoder = get_sentence_transformer(embedding_model_name)
    return encoder.encode(list(docs), show_progress_bar=False)


@st.cache_resource(show_spinner=False, max_entries=9)
def fit_topics(
    docs: tuple[str, ...],
    embeddings: np.ndarray,
    min_topic_size: int,
    seed_topic_list: tuple[tuple[str, ...], ...] | None = None,
) -> tuple[BERTopic, list[int]]:
    model = BERTopic(
        min_topic_size=min_topic_size,
        vectorizer_model=CountVectorizer(stop_words="english"),
        seed_topic_list=[list(group) for group in seed_topic_list]
        if seed_topic_list
        else None,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(list(docs), embeddings)
    return model, topics


@st.cache_resource(show_spinner=False, max_entries=3)
def reduce_to_2d(
    embeddings: np.ndarray, n_neighbors: int, min_dist: float
) -> np.ndarray:
    reducer = UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)

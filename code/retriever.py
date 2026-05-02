"""
Corpus loader and retriever.

Loads all Markdown files from data/ into memory, embeds them with
sentence-transformers, and exposes a top-k semantic search function.
"""

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_DATA_ROOT = Path(__file__).parent.parent / "data"


class Document(NamedTuple):
    path: str
    company: str   # hackerrank | claude | visa
    text: str


class Retriever:
    def __init__(self, data_root: Path = _DATA_ROOT, model_name: str = _MODEL_NAME):
        self._docs: list[Document] = _load_corpus(data_root)
        self._model = SentenceTransformer(model_name)
        texts = [d.text for d in self._docs]
        self._embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    def search(self, query: str, company: str | None = None, top_k: int = 3) -> list[tuple[Document, float]]:
        """Return top-k (document, score) pairs most relevant to query.

        If company is provided and has matching docs, search is restricted to
        that company's corpus. Falls back to the full corpus when company is
        None or unrecognised.
        """
        q_emb = self._model.encode([query], convert_to_numpy=True)[0]

        if company:
            company_lower = company.lower()
            indices = [i for i, doc in enumerate(self._docs) if doc.company == company_lower]
        else:
            indices = list(range(len(self._docs)))

        # Fall back to full corpus if no docs matched the company
        if not indices:
            indices = list(range(len(self._docs)))

        subset_embeddings = self._embeddings[indices]
        scores = _cosine_similarity(q_emb, subset_embeddings)

        top_local = np.argsort(scores)[::-1][:top_k]
        return [(self._docs[indices[i]], float(scores[i])) for i in top_local]


def _load_corpus(root: Path) -> list[Document]:
    docs: list[Document] = []
    for md_file in root.rglob("*.md"):
        rel = md_file.relative_to(root)
        parts = rel.parts
        company = parts[0].lower() if parts else "unknown"
        text = md_file.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append(Document(path=str(md_file), company=company, text=text))
    return docs


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q_norm = query / (np.linalg.norm(query) + 1e-10)
    m_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    return matrix.dot(q_norm) / m_norms.squeeze()

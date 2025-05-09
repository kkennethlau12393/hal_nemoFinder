"""Embedding service using sentence-transformers with lazy loading and pgvector storage."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingService:
    """Provides text embeddings via sentence-transformers.

    The underlying model is loaded lazily on first use to avoid heavy
    imports at module-load time.  When the ``ENABLE_EMBEDDINGS`` feature
    flag is *False*, all methods degrade gracefully:

    * :meth:`similarity` returns ``0.5``
    * Other methods are no-ops (return empty arrays / empty lists).
    """

    def __init__(self, *, enabled: bool | None = None) -> None:
        if enabled is not None:
            self._enabled = enabled
        else:
            raw = os.environ.get("HAL_ENABLE_EMBEDDINGS", "true")
            self._enabled = raw.lower() not in ("0", "false", "no")
        self._model: Any = None  # lazy-loaded SentenceTransformer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string to a dense embedding vector."""
        if not self._enabled:
            return np.array([], dtype=np.float32)
        model = self._get_model()
        return model.encode(text, convert_to_numpy=True)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts. Returns shape ``(len(texts), dim)``."""
        if not self._enabled or not texts:
            return np.array([], dtype=np.float32).reshape(0, 0)
        model = self._get_model()
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Return cosine similarity between two texts (0.0 – 1.0 range)."""
        if not self._enabled:
            return 0.5
        vec_a = self.encode(text_a)
        vec_b = self.encode(text_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    async def store_embedding(
        self,
        text: str,
        metadata: dict,
        session: AsyncSession,
    ) -> None:
        """Compute an embedding and persist it to the pgvector table.

        Expects a table ``knowledge_embeddings`` with columns:
        ``id``, ``text``, ``embedding`` (vector), ``metadata`` (jsonb).
        """
        if not self._enabled:
            return
        vector = self.encode(text)
        await session.execute(
            _insert_embedding_sql(),
            {
                "text": text,
                "embedding": vector.tolist(),
                "metadata": metadata,
            },
        )
        await session.flush()

    async def search_similar(
        self,
        query: str,
        top_k: int,
        session: AsyncSession,
    ) -> list[tuple[str, float, dict]]:
        """Find the *top_k* most similar stored texts via pgvector.

        Returns a list of ``(text, similarity_score, metadata)`` tuples
        ordered by descending similarity.
        """
        if not self._enabled:
            return []
        vector = self.encode(query)
        result = await session.execute(
            _search_similar_sql(),
            {"embedding": vector.tolist(), "top_k": top_k},
        )
        rows = result.fetchall()
        return [(row.text, float(row.similarity), row.metadata) for row in rows]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_model(self) -> Any:
        """Lazy-load the sentence-transformers model on first call."""
        if self._model is None:
            logger.info("Loading sentence-transformers model %r ...", _MODEL_NAME)
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(_MODEL_NAME)
            except Exception:
                logger.exception("Failed to load sentence-transformers model")
                raise
        return self._model


# ------------------------------------------------------------------
# SQL helpers (raw SQL to avoid tight SQLAlchemy model coupling)
# ------------------------------------------------------------------

def _insert_embedding_sql():
    """Return a textual SQL statement for inserting an embedding row."""
    from sqlalchemy import text
    return text(
        "INSERT INTO knowledge_embeddings (text, embedding, metadata) "
        "VALUES (:text, :embedding, :metadata)"
    )


def _search_similar_sql():
    """Return a textual SQL statement for cosine-similarity search."""
    from sqlalchemy import text
    return text(
        "SELECT text, 1 - (embedding <=> :embedding::vector) AS similarity, metadata "
        "FROM knowledge_embeddings "
        "ORDER BY embedding <=> :embedding::vector "
        "LIMIT :top_k"
    )

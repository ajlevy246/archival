"""Embedding implementations."""

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from archival.types import Representation


class SentenceTransformerEmbedder:
    """Generate embeddings using a Sentence Transformers model."""

    def __init__(self, model_id: str):
        self._model_id = model_id
        self._model = SentenceTransformer(model_id)

        probe = self._model.encode(
            ["testing embedding dimensions"],
            normalize_embeddings=True,
        )
        self._dimension = probe.shape[1]

    @property
    def model_id(self) -> str:
        """Return the Hugging Face model identifier."""
        return self._model_id

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality."""
        return self._dimension

    def encode_documents(
        self,
        representations: list[Representation],
    ) -> NDArray[np.float32]:
        """Encode document representations into normalized vectors."""
        texts = [representation.text for representation in representations]

        return self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def encode_query(self, text: str) -> NDArray[np.float32]:
        """Encode a search query into a normalized vector."""
        return self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
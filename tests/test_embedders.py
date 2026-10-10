"""Tests for Archival embedding implementations."""

import numpy as np

from archival.embedders import SentenceTransformerEmbedder
from archival.types import Representation

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def test_sentence_transformer_embedder():
    embedder = SentenceTransformerEmbedder(MODEL_ID)

    representations = [
        Representation(text="A paper about databases.", fields={}),
        Representation(text="A paper about computer graphics.", fields={}),
    ]

    embeddings = embedder.encode_documents(representations)
    query = embedder.encode_query("database systems")

    assert embeddings.shape == (2, embedder.dimension)
    assert query.shape == (embedder.dimension,)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(query), 1.0, atol=1e-5)
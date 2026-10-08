"""Protocols defining the interfaces between core components."""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from archival.types import Hit, PaperMeta, PaperRecord, Representation


class Source(Protocol):
    """Provides a stream of papers to the ingestion pipeline."""

    def __iter__(self) -> Iterator[PaperRecord]:
        """Yield papers from the source."""
        ...


class Representer(Protocol):
    """Converts a paper into the text used for semantic retrieval."""

    def represent(self, paper: PaperRecord) -> Representation:
        """Create a searchable representation of a paper."""
        ...


class Embedder(Protocol):
    """Converts representations into dense embedding vectors."""

    @property
    def model_id(self) -> str:
        """Return the identifier of the embedding model."""
        ...

    @property
    def dimension(self) -> int:
        """Return the dimensionality of each embedding."""
        ...

    def encode_documents(
        self,
        representations: list[Representation],
    ) -> NDArray[np.floating]:
        """Encode document representations into embedding vectors."""
        ...

    def encode_query(self, text: str) -> NDArray[np.floating]:
        """Encode a search query into a single embedding vector."""
        ...


class Codec(Protocol):
    """Encodes embeddings into their persisted storage representation."""

    @property
    def name(self) -> str:
        """Return the codec's identifier."""
        ...

    def encode(self, vectors: NDArray[np.floating]) -> NDArray[np.generic]:
        """Encode floating-point vectors for storage."""
        ...

    def decode(self, vectors: NDArray[np.generic]) -> NDArray[np.floating]:
        """Decode stored vectors back into floating-point vectors."""
        ...


class VectorIndex(Protocol):
    """Searches a collection of encoded vectors."""

    def search(
        self,
        query: NDArray[np.floating],
        k: int,
    ) -> list[Hit]:
        """Return the top-k matching vector rows."""
        ...

    def save(self, path: Path) -> None:
        """Persist the index to disk."""
        ...

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        """Load an index from disk."""
        ...


class LexicalIndex(Protocol):
    """Searches paper text using lexical matching."""

    def search(self, query: str, k: int) -> list[Hit]:
        """Return the top-k matching paper rows."""
        ...

    def save(self, path: Path) -> None:
        """Persist the index to disk."""
        ...

    @classmethod
    def load(cls, path: Path) -> "LexicalIndex":
        """Load an index from disk."""
        ...


class MetadataProvider(Protocol):
    """Resolves vector rows to paper metadata."""

    def get(self, row: int) -> PaperMeta:
        """Return metadata for a vector row."""
        ...
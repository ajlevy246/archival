"""Vector index implementations for semantic retrieval."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from archival.types import Hit


class FlatVectorIndex:
    """Perform exact brute-force search over a matrix of stored vectors."""

    def __init__(self, vectors: NDArray[np.floating]) -> None:
        """Initialize the index from a two-dimensional vector matrix."""
        if vectors.ndim != 2:
            raise ValueError("Vectors must be a two-dimensional matrix.")
        if vectors.shape[0] == 0:
            raise ValueError("The vector index cannot be empty.")
        if vectors.shape[1] == 0:
            raise ValueError("Vectors must have at least one dimension.")
        if not np.isfinite(vectors).all():
            raise ValueError("Vectors must contain only finite values.")

        self._vectors = vectors

    @property
    def size(self) -> int:
        """Return the number of indexed vectors."""
        return self._vectors.shape[0]

    @property
    def dimension(self) -> int:
        """Return the dimensionality of each vector."""
        return self._vectors.shape[1]

    def search(
        self,
        query: NDArray[np.floating],
        k: int,
    ) -> list[Hit]:
        """Return the top-k rows ranked by descending dot-product score."""
        if query.ndim != 1:
            raise ValueError("The query must be a one-dimensional vector.")
        if query.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension {query.shape[0]} does not match "
                f"index dimension {self.dimension}."
            )
        if not np.isfinite(query).all():
            raise ValueError("The query must contain only finite values.")
        if k <= 0:
            raise ValueError("k must be positive.")

        scores = self._vectors.astype(np.float32) @ query.astype(np.float32)
        count = min(k, self.size)
        candidate_rows = np.argpartition(scores, -count)[-count:]
        ranked_rows = candidate_rows[
            np.argsort(scores[candidate_rows])[::-1]
        ]

        return [
            Hit(row=int(row), score=float(scores[row]))
            for row in ranked_rows
        ]

    def save(self, path: Path) -> None:
        """Save the vector matrix as a NumPy .npy file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self._vectors, allow_pickle=False)

    @classmethod
    def load(cls, path: Path) -> "FlatVectorIndex":
        """Load vectors from a NumPy .npy file using memory mapping."""
        vectors = np.load(path, mmap_mode="r", allow_pickle=False)
        return cls(vectors)
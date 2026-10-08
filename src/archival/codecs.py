"""Vector codecs for reducing the storage representation of embeddings."""

import numpy as np
from numpy.typing import NDArray


class Float16Codec:
    """Store floating-point embeddings using IEEE float16."""

    @property
    def name(self) -> str:
        """Return the codec identifier."""
        return "float16"

    def encode(
        self,
        vectors: NDArray[np.floating],
    ) -> NDArray[np.float16]:
        """Convert floating-point vectors to float16."""
        return vectors.astype(np.float16)

    def decode(
        self,
        vectors: NDArray[np.float16],
    ) -> NDArray[np.float32]:
        """Convert stored float16 vectors back to float32."""
        return vectors.astype(np.float32)
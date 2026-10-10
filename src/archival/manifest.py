"""Manifest metadata for persisted document indexes."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class IndexManifest:
    """Describe the configuration and dimensions of a persisted index."""

    format_version: int
    source: str
    representation: str
    embedding_model: str
    embedding_dimension: int
    codec: str
    vector_dtype: str
    row_count: int
    normalized_embeddings: bool

    def save(self, path: Path) -> None:
        """Serialize the manifest to a JSON file."""
        path.write_text(
            json.dumps(asdict(self), indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "IndexManifest":
        """Load a manifest from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)
"""Tests for index manifest serialization."""

from archival.manifest import IndexManifest


def test_manifest_round_trip(tmp_path):
    manifest = IndexManifest(
        format_version=1,
        source="fixture",
        representation="title_abstract_v1",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        codec="float16",
        vector_dtype="float16",
        row_count=10,
        normalized_embeddings=True,
    )

    path = tmp_path / "manifest.json"
    manifest.save(path)
    loaded = IndexManifest.load(path)

    assert loaded == manifest
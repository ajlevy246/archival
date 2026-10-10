"""Tests for vector index implementations."""

import numpy as np
import pytest

from archival.indexes import FlatVectorIndex


def test_search_returns_rows_in_descending_score_order():
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float16,
    )
    index = FlatVectorIndex(vectors)

    hits = index.search(
        np.array([1.0, 0.0], dtype=np.float32),
        k=2,
    )

    assert [hit.row for hit in hits] == [0, 1]
    assert [hit.score for hit in hits] == pytest.approx([1.0, 0.0])


def test_search_returns_all_rows_when_k_exceeds_index_size():
    vectors = np.eye(3, dtype=np.float32)
    index = FlatVectorIndex(vectors)

    hits = index.search(np.array([1.0, 0.0, 0.0]), k=10)

    assert len(hits) == 3


def test_search_rejects_dimension_mismatch():
    index = FlatVectorIndex(np.eye(3, dtype=np.float32))

    with pytest.raises(ValueError, match="dimension"):
        index.search(np.array([1.0, 0.0]), k=1)


def test_search_rejects_non_positive_k():
    index = FlatVectorIndex(np.eye(3, dtype=np.float32))

    with pytest.raises(ValueError, match="positive"):
        index.search(np.array([1.0, 0.0, 0.0]), k=0)


def test_index_rejects_non_finite_vectors():
    vectors = np.array([[1.0, np.nan]], dtype=np.float32)

    with pytest.raises(ValueError, match="finite"):
        FlatVectorIndex(vectors)

def test_index_save_load_round_trip(tmp_path):
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float16,
    )
    index = FlatVectorIndex(vectors)
    path = tmp_path / "vectors.npy"

    index.save(path)
    loaded = FlatVectorIndex.load(path)

    assert loaded.size == index.size
    assert loaded.dimension == index.dimension
    assert loaded._vectors.dtype == np.float16

    query = np.array([1.0, 0.0], dtype=np.float32)
    assert loaded.search(query, k=2) == index.search(query, k=2)
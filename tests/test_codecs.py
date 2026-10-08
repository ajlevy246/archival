"""Tests for vector codecs."""

import numpy as np

from archival.codecs import Float16Codec


def test_float16_codec():
    codec = Float16Codec()

    vectors = np.array(
        [
            [0.1, 0.2, 0.3],
            [-0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )

    encoded = codec.encode(vectors)
    decoded = codec.decode(encoded)

    assert encoded.dtype == np.float16
    assert decoded.dtype == np.float32
    assert decoded.shape == vectors.shape
    assert np.allclose(decoded, vectors, rtol=1e-3, atol=1e-3)
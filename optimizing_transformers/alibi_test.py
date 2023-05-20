import unittest

import jax
import numpy as np

from optimizing_transformers.alibi import alibi_mask
from optimizing_transformers.simple_transformer import \
    TransformerDecoder
from optimizing_transformers.simple_transformer.preprocess import \
    preprocess_sequences


class TestALiBi(unittest.TestCase):
    d_state: int = 512
    n_heads: int = 2
    n_context: int = 5
    vocab_size: int = 10
    seed: int = 0

    mask = alibi_mask(n_context=n_context, m=[1.0] * n_heads)

    def test_shape(self):
        np.testing.assert_array_equal(
            self.mask,
            np.array([[[ 0,  0,  0,  0,  0],
                       [-1,  0,  0,  0,  0],
                       [-2, -1,  0,  0,  0],
                       [-3, -2, -1,  0,  0],
                       [-4, -3, -2, -1,  0]],
                      [[ 0,  0,  0,  0,  0],
                       [-1,  0,  0,  0,  0],
                       [-2, -1,  0,  0,  0],
                       [-3, -2, -1,  0,  0],
                       [-4, -3, -2, -1,  0]]]))

    def test_forward(self):
        """Test that the forward pass works with an ALiBi mask."""
        seq = [[1, 2, 3, 4, 5], [1, 2], [1, 2, 3], [1, 2, 3, 4]]
        x, *_ = preprocess_sequences(
            sequences=seq,
            n_context=self.n_context,
        )
        transformer_decoder = TransformerDecoder(
            n_layers=1,
            d_state=self.d_state,
            n_heads=self.n_heads,
            vocab_size=self.vocab_size,
        )
        out, _ = transformer_decoder.init_with_output(
            jax.random.PRNGKey(self.seed),
            x=x,
            mask=np.expand_dims(self.mask, axis=0),
        )
        assert out.shape == (len(seq), self.n_context, self.vocab_size)

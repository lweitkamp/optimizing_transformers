import unittest

import jax

from optimizing_transformers.simple_transformer import \
    SingleLayerTransformerDecoder
from optimizing_transformers.simple_transformer.preprocess import \
    preprocess_sequences


class TestSimpleTransformer(unittest.TestCase):
    d_state: int = 32
    vocab_size: int = 10
    n_heads: int = 3
    n_context: int = 5
    seed: int = 0

    def test_forward(self):
        seq = [[1, 2, 3, 4, 5], [1, 2], [1, 2, 3], [1, 2, 3, 4]]
        x, mask, _ = preprocess_sequences(
            sequences=seq,
            n_context=self.n_context,
        )
        transformer_decoder = SingleLayerTransformerDecoder(
            d_state=self.d_state,
            vocab_size=self.vocab_size,
            n_heads=self.n_heads,
        )
        out, _ = transformer_decoder.init_with_output(
            jax.random.PRNGKey(self.seed),
            x=x,
            mask=mask,
        )
        assert out.shape == (x.shape[0], x.shape[1], self.vocab_size)

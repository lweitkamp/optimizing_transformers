import unittest

import jax
import jax.numpy as jnp
import numpy as np

from optimizing_transformers.qk_normalization import \
    MultiHeadedAttentionQKNormed
from optimizing_transformers.simple_transformer.preprocess import create_mask


class TestPacking(unittest.TestCase):
    d_state: int = 512
    n_heads: int = 8
    n_context: int = 3
    seed: int = 0

    x = jnp.ones((1, n_context, d_state))
    mask = create_mask([n_context])

    def test_forward(self):
        """There should be no difference in the forward pass with the
        input and output shapes. This is just a basic test."""
        mha_qknormed = MultiHeadedAttentionQKNormed(
            d_state=self.d_state,
            n_heads=self.n_heads,
        )
        out, _ = mha_qknormed.init_with_output(
            jax.random.PRNGKey(self.seed),
            x=self.x,
            mask=self.mask,
        )
        assert out.shape == self.x.shape

    def test_norm(self):
        """The Q and K matrices should be l2 normalized along the head
        dimension. Here we check that this dimension sums to 1.0."""
        mha_qknormed = MultiHeadedAttentionQKNormed(
            d_state=self.d_state,
            n_heads=self.n_heads,
        )
        weights = mha_qknormed.init(
            jax.random.PRNGKey(0),
            x=self.x,
            mask=self.mask,
        )
        _, mod_vars = mha_qknormed.apply(
            weights,
            x=self.x,
            mask=self.mask,
            mutable=['intermediates'],
        )
        Q = mod_vars['intermediates']['Q'][0]
        K = mod_vars['intermediates']['K'][0]
        np.testing.assert_allclose(jnp.linalg.norm(Q, axis=-1), 1.0)
        np.testing.assert_allclose(jnp.linalg.norm(K, axis=-1), 1.0)

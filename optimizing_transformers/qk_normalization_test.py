import unittest

import jax
import jax.numpy as jnp
import numpy as np

from optimizing_transformers.qk_normalization import \
    MultiHeadedAttentionQKNormed, qk_l2_norm, qk_layer_norm
from optimizing_transformers.simple_transformer.preprocess import create_mask


class TestPacking(unittest.TestCase):
    d_state: int = 512
    n_heads: int = 8
    n_context: int = 5
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

    def test_l2_norm(self):
        """The Q and K matrices should be l2 normalized along the head
        dimension. Here we check that the norm of this is 1.0."""
        mha_qknormed = MultiHeadedAttentionQKNormed(
            d_state=self.d_state,
            n_heads=self.n_heads,
            norm=qk_l2_norm,
            L=self.n_context,
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
        np.testing.assert_allclose(jnp.linalg.norm(K, axis=-1), 1.0, rtol=1e-5)

    def test_layer_norm(self):
        """The Q and K matrices should be layer normalized along the head
        dimension."""
        mha_qknormed = MultiHeadedAttentionQKNormed(
            d_state=self.d_state,
            n_heads=self.n_heads,
            norm=qk_layer_norm,
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
        np.testing.assert_allclose(Q.mean(), 0.0, atol=1e-08, rtol=1e-05)
        np.testing.assert_allclose(Q.std(), 1.0, atol=1e-08, rtol=1e-05)
        np.testing.assert_allclose(K.mean(), 0.0, atol=1e-08, rtol=1e-05)
        np.testing.assert_allclose(K.std(), 1.0, atol=1e-08, rtol=1e-05)

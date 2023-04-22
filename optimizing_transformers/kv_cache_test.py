import unittest

import jax
import jax.numpy as jnp
import numpy as np

from optimizing_transformers.kv_cache import MultiHeadedAttention


class TestKVCache(unittest.TestCase):
    d_state: int = 512
    n_heads: int = 8
    n_context: int = 3
    seed: int = 0

    x: jnp.ndarray = jnp.ones((1, n_context, d_state))
    mask: jnp.ndarray = jnp.triu(
        np.ones((n_context, n_context)), k=1).astype('bool')

    def test_forward(self):
        mha = MultiHeadedAttention(
            d_state=self.d_state,
            n_heads=self.n_heads,
        )
        (out, _), weights = mha.init_with_output(
            jax.random.PRNGKey(self.seed),
            x=self.x,
            mask=self.mask,
        )
        x0, x1 = jnp.array_split(self.x, 2, axis=1)
        out0, kv_cache = mha.apply(weights, x=x0)
        out1, kv_cache = mha.apply(weights, x=x1, kv_cache=kv_cache)
        out_kv = jnp.concatenate([out0, out1], axis=1)
        np.testing.assert_allclose(out, out_kv, atol=1e-5)

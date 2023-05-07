import unittest

import jax
import jax.numpy as jnp
import numpy as np

from optimizing_transformers.kv_cache import MultiHeadedAttentionKVCache
from optimizing_transformers.simple_transformer.preprocess import \
    create_mask


class TestKVCache(unittest.TestCase):
    d_state: int = 512
    n_heads: int = 8
    n_context: int = 3
    seed: int = 0

    def test_forward(self):
        """There should be no difference in the forward pass with and
        without KV cache. Note that masking is not required in decode phase."""
        x = jnp.ones((1, self.n_context, self.d_state))
        mask = create_mask([self.n_context])
        mha_kvcache = MultiHeadedAttentionKVCache(
            d_state=self.d_state,
            n_heads=self.n_heads,
        )
        (out, _), weights = mha_kvcache.init_with_output(
            jax.random.PRNGKey(self.seed),
            x=x,
            mask=mask,
        )
        x0, x1 = jnp.array_split(x, 2, axis=1)
        out0, kv_cache = mha_kvcache.apply(weights, x=x0)
        out1, kv_cache = mha_kvcache.apply(weights, x=x1, kv_cache=kv_cache)
        out_kv = jnp.concatenate([out0, out1], axis=1)
        np.testing.assert_allclose(out, out_kv, atol=1e-5)

from typing import Optional

import flax.linen as nn
import jax.numpy as jnp

from optimizing_transformers.attention import dot_product_attention


class MultiQueryAttention(nn.Module):
    d_state: int = 512
    n_heads: int = 8

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Multi-headed auto-regressive attention.

        x: Input tensor of shape [b, n_context, d_state]
        mask: Mask tensor of shape [b, n_context, n_context]
        kv_cache: Key and value cache tensor of shape [b, n_heads, d_head]
        """
        # Transform to Query, Key and Value matrices.
        features_q = (self.n_heads, self.d_state // self.n_heads)
        features_kv = (1, self.d_state // self.n_heads)
        q = nn.DenseGeneral(features=features_q, axis=-1, use_bias=False)(x)
        k = nn.DenseGeneral(features=features_kv, axis=-1, use_bias=False)(x)
        v = nn.DenseGeneral(features=features_kv, axis=-1, use_bias=False)(x)

        attention = dot_product_attention(
            q.transpose((0, 2, 1, 3)),
            k.transpose((0, 2, 3, 1)),
            v.transpose((0, 2, 1, 3)),
            mask,
        )

        out = nn.DenseGeneral(features=self.d_state, axis=(-3, -1))(attention)
        return out

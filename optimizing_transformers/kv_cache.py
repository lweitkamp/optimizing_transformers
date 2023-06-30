from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp

from optimizing_transformers.attention import dot_product_attention


class MultiHeadedAttentionKVCache(nn.Module):
    d_state: int = 512
    n_heads: int = 8

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        kv_cache: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """Multi-headed auto-regressive attention.

        x: Input tensor of shape [b, n_context, d_state]
        mask: Mask tensor of shape [b, n_context, n_context]
        kv_cache: Key and value cache tensor of shape [b, n_heads, d_head]
        """
        # Transform to Query, Key and Value matrices.
        features = (self.n_heads, self.d_state // self.n_heads)
        Q = nn.DenseGeneral(features=features, axis=-1)(x)
        K = nn.DenseGeneral(features=features, axis=-1)(x)
        V = nn.DenseGeneral(features=features, axis=-1)(x)

        if kv_cache is not None:
            K = jnp.concatenate([kv_cache[0], K], axis=1)
            V = jnp.concatenate([kv_cache[1], V], axis=1)
            kv_cache = K, V
            mask = None

        if kv_cache is None:
            kv_cache = K, V

        attention = dot_product_attention(
            Q.transpose((0, 2, 1, 3)),
            K.transpose((0, 2, 3, 1)),
            V.transpose((0, 2, 1, 3)),
            mask,
        )

        out = nn.DenseGeneral(features=self.d_state, axis=(-3, -1))(attention)
        return out, kv_cache

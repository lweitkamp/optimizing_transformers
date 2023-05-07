from typing import Optional

import flax.linen as nn
import jax.numpy as jnp

from optimizing_transformers.attention import dot_product_attention


class MultiHeadedAttentionQKNormed(nn.Module):
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
        features = (self.n_heads, self.d_state // self.n_heads)
        Q = nn.DenseGeneral(features=features, axis=-1)(x)
        K = nn.DenseGeneral(features=features, axis=-1)(x)
        V = nn.DenseGeneral(features=features, axis=-1)(x)

        # l2 normalize Q and K along the head dimension.
        Q = Q / jnp.linalg.norm(Q, axis=-1, keepdims=True)
        K = K / jnp.linalg.norm(K, axis=-1, keepdims=True)

        # Optionally save these values for unit tests.
        self.sow('intermediates', 'Q', Q)
        self.sow('intermediates', 'K', K)

        attention = dot_product_attention(
            Q.transpose((0, 2, 1, 3)),
            K.transpose((0, 2, 3, 1)),
            V.transpose((0, 2, 1, 3)),
            mask,
        )

        out = nn.DenseGeneral(features=self.d_state, axis=(-3, -1))(attention)
        return out

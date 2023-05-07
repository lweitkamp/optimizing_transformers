from typing import Optional, Callable, Union

import flax.linen as nn
import jax.numpy as jnp

from optimizing_transformers.attention import dot_product_attention


def qk_layer_norm(x: jnp.ndarray):
    """Layer normalization over the last dimension of x.
    Used in Scaling Vision Transformers to 22 Billion Parameters,
    Dehghani et al.
    """
    return nn.LayerNorm(reduction_axes=-1)(x)


def qk_l2_norm(x: jnp.ndarray):
    """L2 normalization over the last dimension of x.
    Used in Query-Key Normalization for Transformers, Henry et al."""
    return x / jnp.linalg.norm(x, axis=-1, keepdims=True)


class MultiHeadedAttentionQKNormed(nn.Module):
    d_state: int = 512
    n_heads: int = 8

    norm: Union[qk_l2_norm, qk_layer_norm] = qk_layer_norm
    L: Optional[int] = None  # 97.5th percentile sequence length

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

        # Normalize Q and K.
        Q = self.norm(Q)
        K = self.norm(K)

        # Figure out the attention scale.
        if self.norm.__name__ == "qk_l2_norm":
            assert self.L is not None, "L must be set for qk_l2_norm."
            init_value = jnp.log2(self.L**2 - self.L)
            scale = self.param('g', lambda *_: init_value, (1,))
        else:
            scale = jnp.sqrt(jnp.array(self.d_state // self.n_heads))

        # Optionally save these values for unit tests.
        self.sow('intermediates', 'Q', Q)
        self.sow('intermediates', 'K', K)

        attention = dot_product_attention(
            Q.transpose((0, 2, 1, 3)),
            K.transpose((0, 2, 3, 1)),
            V.transpose((0, 2, 1, 3)),
            mask,
            scale=scale,
        )

        out = nn.DenseGeneral(features=self.d_state, axis=(-3, -1))(attention)
        return out

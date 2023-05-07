import flax.linen as nn
import jax.numpy as jnp
import jax

from typing import Optional


def dot_product_attention(q, k, v, mask, scale: Optional[float] = None):
    scale = scale or jnp.sqrt(k.shape[-1])
    attention_weights = (q @ k) / scale

    if mask is not None:
        if attention_weights.ndim == 4 and mask.ndim == 3:
            mask = jnp.expand_dims(mask, axis=1)
       
        min_infty = jnp.finfo(attention_weights.dtype).min
        attention_weights = jnp.where(mask, min_infty, attention_weights)

    attention_weights = jax.nn.softmax(attention_weights, axis=-1)

    attention = attention_weights @ v
    return attention


class Attention(nn.Module):
    d_state: int = 512

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
        Q = nn.Dense(features=self.d_state)(x)
        K = nn.Dense(features=self.d_state)(x)
        V = nn.Dense(features=self.d_state)(x)

        attention = dot_product_attention(
            Q,
            K.transpose((0, 2, 1)),
            V,
            mask,
        )

        out = nn.Dense(features=self.d_state)(attention)

        return out


class MultiHeadedAttention(nn.Module):
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

        attention = dot_product_attention(
            Q.transpose((0, 2, 1, 3)),
            K.transpose((0, 2, 3, 1)),
            V.transpose((0, 2, 1, 3)),
            mask,
        )

        out = nn.DenseGeneral(features=self.d_state, axis=(-3, -1))(attention)
        return out

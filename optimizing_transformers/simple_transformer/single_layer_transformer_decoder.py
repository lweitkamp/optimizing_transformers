"""Implementation of a single layer transformer decoder.
Positional encoding is omitted for simplicity."""
import flax.linen as nn
import jax.numpy as jnp

from optimizing_transformers.attention import Attention
from optimizing_transformers.simple_transformer.mlp import MultiLayerPerceptron


class SingleLayerTransformerDecoder(nn.Module):
    d_state: int
    vocab_size: int

    attn_fn: nn.Module = Attention
    mlp_fn: nn.Module = MultiLayerPerceptron

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:

        # Embed tokens
        x = nn.Embed(self.vocab_size, self.d_state)(x)

        # (Multi-Headed) self-attention
        attn = self.attn_fn(self.d_state)(x, mask=mask)
        x = nn.LayerNorm()(x + attn)

        # MLP
        x = nn.LayerNorm()(x + self.mlp_fn(self.d_state)(x))
        x = nn.Dense(self.vocab_size)(x)
        return x

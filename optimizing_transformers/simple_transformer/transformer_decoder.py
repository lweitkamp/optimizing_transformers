"""Implementation of a single layer transformer decoder.
Positional encoding is omitted for simplicity."""
import flax.linen as nn
import jax.numpy as jnp

from optimizing_transformers.attention import MultiHeadedAttention
from optimizing_transformers.simple_transformer.mlp import MultiLayerPerceptron
from optimizing_transformers.simple_transformer.decoder_block import \
    TransformerDecoderBlock


class TransformerDecoder(nn.Module):
    n_layers: int
    d_state: int
    vocab_size: int
    n_heads: int

    attn_fn: nn.Module = MultiHeadedAttention
    mlp_fn: nn.Module = MultiLayerPerceptron

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:

        # Embed tokens
        x = nn.Embed(self.vocab_size, self.d_state)(x)

        for _ in range(self.n_layers):
            x = TransformerDecoderBlock(
                    d_state=self.d_state,
                    n_heads=self.n_heads,
                    attn_fn=self.attn_fn,
                    mlp_fn=self.mlp_fn,
            )(x, mask=mask)

        # Unembed tokens
        x = nn.Dense(self.vocab_size)(x)
        return x

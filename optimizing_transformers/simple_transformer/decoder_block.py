import flax.linen as nn
import jax.numpy as jnp

from optimizing_transformers.multi_headed_attention import MultiHeadedAttention
from optimizing_transformers.simple_transformer.mlp import MultiLayerPerceptron


class TransformerDecoderBlock(nn.Module):
    d_state: int
    n_heads: int

    attn_fn: nn.Module = MultiHeadedAttention
    mlp_fn: nn.Module = MultiLayerPerceptron

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        # (Multi-Headed) self-attention
        attn = self.attn_fn(self.d_state, self.n_heads)(x, mask=mask)
        x = nn.LayerNorm()(x + attn)

        # MLP
        x = nn.LayerNorm()(x + self.mlp_fn(self.d_state)(x))
        x = nn.Dense(self.d_state)(x)
        return x

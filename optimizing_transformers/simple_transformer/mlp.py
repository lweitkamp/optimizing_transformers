import flax.linen as nn
import jax.numpy as jnp

import jax


class MultiLayerPerceptron(nn.Module):
    d_state: int = 512

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.d_state * 4)(x)
        h = jax.nn.relu(h)
        h = nn.Dense(self.d_state)(h)
        return x

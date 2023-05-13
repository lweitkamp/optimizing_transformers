import unittest

import jax
import jax.numpy as jnp
import numpy as np

from optimizing_transformers.alibi import alibi_mask


class TestALiBi(unittest.TestCase):
    d_state: int = 512
    n_heads: int = 2
    n_context: int = 5
    seed: int = 0

    def test_shape(self):
        np.testing.assert_array_equal(
            alibi_mask(n_context=self.n_context, m=[1.0] * self.n_heads),
            np.array([[[ 0,  0,  0,  0,  0],
                       [-1,  0,  0,  0,  0],
                       [-2, -1,  0,  0,  0],
                       [-3, -2, -1,  0,  0],
                       [-4, -3, -2, -1,  0]],
                      [[ 0,  0,  0,  0,  0],
                       [-1,  0,  0,  0,  0],
                       [-2, -1,  0,  0,  0],
                       [-3, -2, -1,  0,  0],
                       [-4, -3, -2, -1,  0]]]))

    def test_forward(self):
        """Test that the forward pass works with an ALiBi mask."""
        # TODO
        return True

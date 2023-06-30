from optimizing_transformers.multi_query_attention import MultiQueryAttention
from optimizing_transformers.multi_headed_attention import MultiHeadedAttention
import jax
import unittest


class TestMultiQueryAttention(unittest.TestCase):
    d_state: int = 512
    n_heads: int = 8
    n_context: int = 5
    seed: int = 0

    def test_forward(self):
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (2, 5, 32))

        mqa = MultiQueryAttention(32, 8)
        mha = MultiHeadedAttention(32, 8)

        mqa_params = mqa.init(rng, x)
        mha_params = mha.init(rng, x)

        print("MultiQueryAttention")
        return True

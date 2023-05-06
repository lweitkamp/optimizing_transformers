import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax

from optimizing_transformers.packing import greedy_histogram_pack
from optimizing_transformers.simple_transformer import \
    SingleLayerTransformerDecoder
from optimizing_transformers.simple_transformer.preprocess import \
    preprocess_sequences


class TestPacking(unittest.TestCase):
    d_state: int = 2
    vocab_size: int = 10
    n_context: int = 5
    seed: int = 0

    seq = [
        jax.random.randint(jax.random.PRNGKey(0), (1, 2), 0, vocab_size),
        jax.random.randint(jax.random.PRNGKey(0), (1, 2), 0, vocab_size),
        jax.random.randint(jax.random.PRNGKey(0), (1, 3), 0, vocab_size),
    ]

    def test_gradient_equal(self):
        """The loss for packed and unpacked sequences should be equal.
        The gradients /should/ be equal, but are not due to numerical
        instability. They are pretty close though!
        """
        seq = [[1, 2, 3, 4, 5], [1, 2], [1, 2, 3], [1, 2, 3, 4]]
        x, mask, _ = preprocess_sequences(
            sequences=seq,
            n_context=self.n_context,
        )
        packed_seq, packed_ind = greedy_histogram_pack(seq, self.n_context)
        x_pack, mask_pack, _ = preprocess_sequences(
            sequences=packed_seq,
            sequences_lengths=packed_ind,
            n_context=self.n_context,
        )

        print("")

        transformer_decoder = SingleLayerTransformerDecoder(
            d_state=self.d_state,
            vocab_size=self.vocab_size,
            n_heads=1,
        )
        weights = transformer_decoder.init(
            jax.random.PRNGKey(self.seed),
            x=x,
            mask=mask,
        )

        def loss(weights, x, mask):
            labels = jnp.where(x == 0, -100, x)
            out = transformer_decoder.apply(weights, x=x, mask=mask)
            loss_ = optax.softmax_cross_entropy_with_integer_labels(
                logits=out, labels=labels
            )
            loss_ = jnp.nan_to_num(loss_)
            return jnp.sum(loss_)

        # Calculate loss for padded and packed+padded samples.
        l, grad = jax.value_and_grad(loss)(weights, x, mask)
        l_p, grad_pack = jax.value_and_grad(loss)(weights, x_pack, mask_pack)

        # # Loss is equal.
        np.testing.assert_allclose(l, l_p)

        # Gradients are (somewhat) equal.
        for g, g_p in zip(
            jax.tree_util.tree_flatten(grad)[0],
            jax.tree_util.tree_flatten(grad_pack)[0],
        ):
            np.testing.assert_allclose(g, g_p, atol=1e-5)

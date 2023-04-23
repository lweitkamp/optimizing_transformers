"""Functions and classes related to tokenization, including padding."""
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


def pad_sample(
        sequence: jnp.ndarray,
        mask: jnp.ndarray,
        n_context: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pad a sample (sequence and mask) to fit the context length.

    Args:
        sequence: The sequence to pad.
        mask: The mask to pad.
        n_context: The context length.

    Returns:
        A tuple of the padded sequence and mask.
    """
    padding = jnp.zeros((1, n_context - sequence.shape[1]))
    padded_sequence = jnp.concatenate([sequence, padding], axis=1)
    mask = pad_mask(mask, n_context)
    return padded_sequence, mask


def create_mask(seq_len: int) -> jnp.ndarray:
    """Create an autoregressive mask for a sequence of length seq_len.

    Args:
        seq_len: The length of the sequence.

    Returns:
        A mask of shape (seq_len, seq_len) with ones on the lower triangle.
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype('bool')
    return jax.device_put(mask)


def pad_mask(mask: jnp.ndarray, n_context: int) -> jnp.ndarray:
    """Pad an autoregressive mask to fit context length.

    Args:
        mask: The mask to pad.
        n_context: The context length.

    Returns:
        A mask of shape (n_context, n_context) with ones on the lower triangle.
    """
    new_mask = np.ones((n_context, n_context), dtype=bool)
    seq_len = mask.shape[1]
    new_mask[:seq_len, :seq_len] = mask
    return jax.device_put(new_mask)


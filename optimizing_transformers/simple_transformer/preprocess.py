from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def pad_sequence_and_mask(
    sequences: np.ndarray,
    masks: np.ndarray,
    n_context: int,
    pad_value: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad sequence and mask to the context length."""
    padded_sequences = np.full(
        (len(sequences), n_context),
        fill_value=pad_value,
    )
    padded_masks = np.full(
        (len(masks), n_context, n_context),
        fill_value=False,
    )

    for i, sequence in enumerate(sequences):
        padded_sequences[i, : len(sequence)] = sequence
        padded_masks[i, : masks[i].shape[0], : masks[i].shape[0]] = masks[i]

    return padded_sequences, padded_masks


def create_mask(sequence_lengths: List[int]) -> np.ndarray:
    """Create autoregressive mask. The input is assumed to be a single
    sequence (list of size one) or a pack of sequences. The output is a single
    autoregressive mask.

    Args:
        sequence_lengths: List of sequences lengths.

    Returns:
        Autoregressive masks.
    """

    def create_autoregressive_mask(sequence_length: int) -> np.ndarray:
        mask = np.triu(np.ones((sequence_length, sequence_length)), k=1)
        return mask

    sequence, offset = sum(sequence_lengths), 0
    mask = np.ones((sequence, sequence), dtype=np.bool_)
    for length in sequence_lengths:
        mask[
            offset:offset+length,
            offset:offset+length,
        ] = create_autoregressive_mask(length)
        offset += length
    return mask


def preprocess_sequences(
    sequences: List[np.ndarray],
    n_context: int,
    sequences_lengths: Optional[List[int]] = None,
    pad_value: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Preprocess sequences and create masks.

    Args:
        sequences: List of sequences.
        n_context: Context length.
        sequences_lengths: List of sequences lengths.
        pad_value: Value to pad sequences.

    Returns:
        Padded sequences, masks and sequences lengths.
    """

    if sequences_lengths is None:
        sequences_lengths = [[len(sequence)] for sequence in sequences]

    masks = [create_mask(seq_len) for seq_len in sequences_lengths]
    padded_sequences, padded_masks = pad_sequence_and_mask(
        sequences,
        masks,
        n_context,
        pad_value=pad_value,
    )

    return (
        jax.device_put(padded_sequences),
        jax.device_put(padded_masks),
        sequences_lengths,
    )

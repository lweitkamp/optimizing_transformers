from typing import List, Tuple

import jax.numpy as jnp
import numpy as np

from optimizing_transformers.simple_transformer.token import (
    create_mask, pad_sample)


def pack_samples(
        samples: List[Tuple[jnp.ndarray, jnp.ndarray]],
        n_context: int,
) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Pack samples greedily until the context length is reached.
    Only one pass is made over the samples, this is just a demo.
    Repeat this function untill the length of each sample does not change."""
    sorted_samples = sorted(samples, key=lambda x: x[0].shape[1])

    # Pack the sequences greedily.
    packed_sequences = []
    (current_s, current_m) = sorted_samples[0]
    for sample in sorted_samples[1:]:
        s, m = sample

        if current_s.shape[1] + s.shape[1] <= n_context:
            # Merge the sequences and masks.
            new_mask = np.ones((current_s.shape[1] + s.shape[1],
                                current_s.shape[1] + s.shape[1]), dtype=bool)
            new_mask[:current_s.shape[1], :current_s.shape[1]] = current_m
            new_mask[current_s.shape[1]:, current_s.shape[1]:] = m

            current_s = jnp.concatenate([current_s, s], axis=1)
            current_m = new_mask

        else:
            packed_sequences.append((current_s, current_m))
            current_s, current_m = s, m

    packed_sequences.append((current_s, current_m))
    return packed_sequences


def preprocess_sequences(
        sequences: List[jnp.ndarray],
        n_context: int,
        pack: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pad sequences and add masks. Optionally packs the sequences."""
    masks = [create_mask(s.shape[1]) for s in sequences]
    samples = [(s, m) for s, m in zip(sequences, masks)]

    if pack:
        samples = pack_samples(samples, n_context=n_context)

    samples = [pad_sample(s, m, n_context=n_context) for s, m in samples]
    x = jnp.concatenate([s[0] for s in samples], axis=0)
    mask = jnp.stack([s[1] for s in samples], axis=0)
    return x, mask

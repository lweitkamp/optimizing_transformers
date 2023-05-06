from typing import List, Tuple, Union

import numpy as np


def greedy_histogram_pack(
    sequences: List[Union[np.ndarray, List[int]]],
    n_context: int,
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Two pointer greedy approach to packing.
    Sort the sequences by length and then pack the longest and shortest.
    Complexity is O(n log n) due to sorting.

    Args:
        sequences: List of sequences to pack.
        n_context: The maximum length of the packed sequence.

    Returns:
        packed_sequences: A list of tuples (packed_sequence, list_of_lengths)
        that indicates the length of each sequence in a packed sequence.
    """
    sorted_sequences = sorted(sequences, key=len)
    sorted_sequences = [(np.array(seq), [len(seq)])
                        for seq in sorted_sequences]

    packed_sequences = []
    i, j = 0, len(sequences) - 1

    while i < j:
        seq_i, ind_i = sorted_sequences[i]
        seq_j, ind_j = sorted_sequences[j]

        # packable
        if len(seq_i) + len(seq_j) <= n_context:
            sorted_sequences[j] = (
                np.concatenate([seq_j, seq_i], axis=0),
                ind_j + ind_i,
            )
            i += 1
        else:
            packed_sequences.append((np.asarray(seq_j), ind_j))
            j -= 1

    # add the last sequence
    packed_sequences.append(sorted_sequences[j])
    packed_equences, packed_indices = zip(*packed_sequences)
    return packed_equences, packed_indices

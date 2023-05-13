import numpy as np
from typing import List


def alibi_mask(
        n_context: int,
        m: List[float],
) -> np.ndarray:
    """Create a mask for the heads of the alibi model.

    Args:
        n_context: Number of context tokens.
        m: List of floats, one for each head.

    Returns:
        A mask for the heads of the alibi model."""
    mask_template = np.tril(
        [np.arange(-i, n_context-i, 1) for i in range(n_context)]
    )
    mask_heads = np.array([mask_template * m_i for m_i in m])
    return mask_heads

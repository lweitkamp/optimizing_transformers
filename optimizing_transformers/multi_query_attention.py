import jax.numpy as jnp
import jax


def DotProductAttention(
    q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
) -> jnp.ndarray:
    """Dot-Product Attention on one query.
    
    Args:
        q: a vector with shape [k]
        K: a matrix with shape [m, k]
        V: a matrix with shape [m, v]
    
    Returns:
        y: a vector with shape [v]
    """
    logits = jnp.einsum("k, mk -> m", q, K)
    weights = jax.nn.softmax(logits)
    return jnp.einsum("m, mv -> v", weights, V)


def MultiHeadAttention(
    x: jnp.ndarray,
    M: jnp.ndarray,
    P_q: jnp.ndarray,
    P_k: jnp.ndarray,
    P_v: jnp.ndarray,
    P_o: jnp.ndarray,
) -> jnp.ndarray:
    """Multi-head Attention on one query.

    Args:
        x: a vector with shape [d]
        M: a matrix with shape [m, d]
        P_q: a tensor with shape [h, d, k]
        P_k: a tensor with shape [h, d, k]
        P_v: a tensor with shape [h, d, v]
        P_o: a tensor with shape [h, v, d]

    Returns:
        y: a vector with shape [d]
    """
    q = jnp.einsum("d, hdk -> hk", x, P_q)
    K = jnp.einsum("md, hdk -> hmk", M, P_k)
    V = jnp.einsum("md, hdv -> hmv", M, P_v)
    logits = jnp.einsum("hk, hmk -> hm", q, K)
    weights = jax.nn.softmax(logits)
    o = jnp.einsum("hm, hmv -> hv", weights, V)
    y = jnp.einsum("hv, hdv -> d", o, P_o)
    return y


def MultiheadAttentionBatched(
    X: jnp.ndarray,
    M: jnp.ndarray,
    mask: jnp.ndarray,
    P_q: jnp.ndarray,
    P_k: jnp.ndarray,
    P_v: jnp.ndarray,
    P_o: jnp.ndarray,
) -> jnp.ndarray:
    """Multi-head Attention.

    Args:
        X: a tensor with shape [b, n, d]
        M: a tensor with shape [b, m, d]
        masks: a tensor with shape [b, h, n, m]
        P_q: a tensor with shape [h, d, k]
        P_k: a tensor with shape [h, d, k]
        P_v: a tensor with shape [h, d, v]
        P_o: a tensor with shape [h, d, v]

    Returns:
        Y: a tensor with shape [b, n, d]
    """
    Q = jnp.einsum("bnd, hdk -> bhnk", X, P_q)
    K = jnp.einsum("bmd, hdk -> bhmk", M, P_k)
    V = jnp.einsum("bmd, hdv -> bhmv", M, P_v)
    logits = jnp.einsum("bhnk, bhmk -> bhnm", Q, K)
    weights = jax.nn.softmax(logits + mask)
    O = jnp.einsum("bhnm, bhmv -> bhnv", weights, V)
    Y = jnp.einsum("bhnv, hdv -> bnd", O, P_o)
    return Y

def MultiqueryAttentionBatched(
        X: jnp.ndarray,
        M: jnp.ndarray,
        mask: jnp.ndarray,
        P_q: jnp.ndarray,
        P_k: jnp.ndarray,
        P_v: jnp.ndarray,
        P_o: jnp.ndarray,
) -> jnp.ndarray:
    """Multi-query Attention.

    Args:
        X: a tensor with shape [b, n, d]
        M: a tensor with shape [b, m, d]
        masks: a tensor with shape [b, h, n, m]
        P_q: a tensor with shape [h, d, k]
        P_k: a tensor with shape [h, d, k]
        P_v: a tensor with shape [h, d, v]
        P_o: a tensor with shape [h, d, v]

    Returns:
        Y: a tensor with shape [b, n, d]
    """
import numpy as np
import time

BLOCK_SIZE = 1024
NEG_INF = -1e10
EPSILON = 1e-10


def normal_attention(Q, K, V, mask=None):
    scale = 1 / np.sqrt(Q.shape[-1])
    Q_scaled = Q * scale
    QKt = np.einsum("...id,...jd->...ij", Q_scaled, K)

    if mask is not None:
        key_mask = mask[:, None, None, :]  # Shape: (b, 1, 1, j)
        QKt = np.where(key_mask > 0, QKt, NEG_INF)

    # Stable softmax
    QKt_max = np.max(QKt, axis=-1, keepdims=True)
    exp_QKt = np.exp(QKt - QKt_max)
    attn = exp_QKt / np.sum(exp_QKt, axis=-1, keepdims=True)
    return np.matmul(attn, V)


def flash_attention_forward(Q, K, V, mask=None):
    O = np.zeros_like(Q)
    l = np.zeros(Q.shape[:-1] + (1,))
    m = np.full(Q.shape[:-1] + (1,), NEG_INF)

    scale = 1 / np.sqrt(Q.shape[-1])

    Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[2])
    KV_BLOCK_SIZE = BLOCK_SIZE
    seq_len = Q.shape[2]
    Tr = int(np.ceil(seq_len / Q_BLOCK_SIZE))
    Tc = int(np.ceil(seq_len / KV_BLOCK_SIZE))

    for j in range(Tc):
        start_j = j * KV_BLOCK_SIZE
        end_j = min(start_j + KV_BLOCK_SIZE, seq_len)
        Kj = K[:, :, start_j:end_j, :]
        Vj = V[:, :, start_j:end_j, :]
        maskj = None if mask is None else mask[:, start_j:end_j]

        for i in range(Tr):
            start_i = i * Q_BLOCK_SIZE
            end_i = min(start_i + Q_BLOCK_SIZE, seq_len)
            Qi = Q[:, :, start_i:end_i, :]
            Oi_slice = (
                slice(None, None),
                slice(None, None),
                slice(start_i, end_i),
                slice(None),
            )
            li_slice = (
                slice(None, None),
                slice(None, None),
                slice(start_i, end_i),
                slice(None),
            )
            mi_slice = (
                slice(None, None),
                slice(None, None),
                slice(start_i, end_i),
                slice(None),
            )

            Qi_scaled = Qi * scale
            S_ij = np.einsum("...id,...jd->...ij", Qi_scaled, Kj)

            if mask is not None:
                maskj_temp = (maskj > 0)[:, None, None, :]  # Shape: (b, 1, 1, block_j)
                S_ij = np.where(maskj_temp, S_ij, NEG_INF)

            m_block_ij = np.max(S_ij, axis=-1, keepdims=True)
            P_ij = np.exp(S_ij - m_block_ij)

            if mask is not None:
                P_ij = np.where(maskj_temp, P_ij, 0.0)

            l_block_ij = np.sum(P_ij, axis=-1, keepdims=True) + EPSILON
            P_ij_Vj = np.matmul(P_ij, Vj)

            mi_new = np.maximum(m_block_ij, m[mi_slice])
            li_new = (
                np.exp(m[mi_slice] - mi_new) * l[li_slice]
                + np.exp(m_block_ij - mi_new) * l_block_ij
            )

            O[Oi_slice] = (l[li_slice] / li_new) * np.exp(m[mi_slice] - mi_new) * O[
                Oi_slice
            ] + (np.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj
            l[li_slice] = li_new
            m[mi_slice] = mi_new

    return O, l, m


def flash_attention_backward(Q, K, V, mask, O, l, m, dO):
    dQ = np.zeros_like(Q)
    dK = np.zeros_like(K)
    dV = np.zeros_like(V)

    scale = 1 / np.sqrt(Q.shape[-1])

    Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[2])
    KV_BLOCK_SIZE = BLOCK_SIZE
    seq_len = Q.shape[2]
    Tr = int(np.ceil(seq_len / Q_BLOCK_SIZE))
    Tc = int(np.ceil(seq_len / KV_BLOCK_SIZE))

    for j in range(Tc):
        start_j = j * KV_BLOCK_SIZE
        end_j = min(start_j + KV_BLOCK_SIZE, seq_len)
        Kj = K[:, :, start_j:end_j, :]
        Vj = V[:, :, start_j:end_j, :]
        maskj = None if mask is None else mask[:, start_j:end_j]

        dKj_block = np.zeros_like(Kj)
        dVj_block = np.zeros_like(Vj)

        for i in range(Tr):
            start_i = i * Q_BLOCK_SIZE
            end_i = min(start_i + Q_BLOCK_SIZE, seq_len)
            Qi = Q[:, :, start_i:end_i, :]
            Oi = O[:, :, start_i:end_i, :]
            dOi = dO[:, :, start_i:end_i, :]
            li = l[:, :, start_i:end_i, :]
            mi = m[:, :, start_i:end_i, :]

            Qi_scaled = Qi * scale
            S_ij = np.einsum("...id,...jd->...ij", Qi_scaled, Kj)

            if mask is not None:
                maskj_temp = (maskj > 0)[:, None, None, :]
                S_ij = np.where(maskj_temp, S_ij, NEG_INF)

            P_ij = (1 / li) * np.exp(S_ij - mi)

            if mask is not None:
                P_ij = np.where(maskj_temp, P_ij, 0.0)

            dVj_block += np.einsum("...ij,...id->...jd", P_ij, dOi)
            dP_ij = np.einsum("...id,...jd->...ij", dOi, Vj)

            Di = np.sum(dOi * Oi, axis=-1, keepdims=True)
            dS_ij = P_ij * (dP_ij - Di)

            dQ[:, :, start_i:end_i, :] += scale * np.einsum(
                "...ij,...jd->...id", dS_ij, Kj
            )
            dKj_block += scale * np.einsum("...ij,...id->...jd", dS_ij, Qi)

        dK[:, :, start_j:end_j, :] = dKj_block
        dV[:, :, start_j:end_j, :] = dVj_block

    return dQ, dK, dV


def flash_attention(Q, K, V, mask=None):
    return flash_attention_forward(Q, K, V, mask)[0]


if __name__ == "__main__":
    Q = np.random.randn(1, 2, 4096, 1024)
    K = np.random.randn(1, 2, 4096, 1024)
    V = np.random.randn(1, 2, 4096, 1024)
    mask = np.random.randint(0, 2, (1, 4096))

    for _ in range(10):
        start1 = time.time()
        out1 = flash_attention(Q, K, V, mask)
        end1 = time.time()

        start2 = time.time()
        out2 = normal_attention(Q, K, V, mask)
        end2 = time.time()

        t1 = (end1 - start1) * 1000
        t2 = (end2 - start2) * 1000

        print(f"{t1:.2f}ms, {t2:.2f}ms")
        print(np.allclose(out1, out2, atol=1e-5))

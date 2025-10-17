import numpy as np
import time

BLOCK_SIZE = 1024
NEG_INF = -1e10
EPSILON = 1e-10


def normal_attention_causal(Q, K, V):
    scale = 1 / np.sqrt(Q.shape[-1])
    Q_scaled = Q * scale
    QKt = np.einsum("...id,...jd->...ij", Q_scaled, K)

    seq_q, seq_k = Q.shape[2], K.shape[2]
    causal_mask = np.triu(np.ones((seq_q, seq_k)), diagonal=seq_k - seq_q + 1)
    QKt = np.where(causal_mask > 0, NEG_INF, QKt)

    # Stable softmax
    QKt_max = np.max(QKt, axis=-1, keepdims=True)
    exp_QKt = np.exp(QKt - QKt_max)
    attn = exp_QKt / (np.sum(exp_QKt, axis=-1, keepdims=True) + EPSILON)
    return np.matmul(attn, V)


def flash_attention_causal_forward(Q, K, V):
    O = np.zeros_like(Q)
    l = np.zeros(Q.shape[:-1] + (1,))
    m = np.full(Q.shape[:-1] + (1,), NEG_INF)

    Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[-1])
    KV_BLOCK_SIZE = BLOCK_SIZE
    seq_q, seq_k = Q.shape[2], K.shape[2]
    Tr = int(np.ceil(seq_q / Q_BLOCK_SIZE))
    Tc = int(np.ceil(seq_k / KV_BLOCK_SIZE))
    offset = seq_k - seq_q

    for j in range(Tc):
        start_j = j * KV_BLOCK_SIZE
        end_j = min(start_j + KV_BLOCK_SIZE, seq_k)
        Kj = K[:, :, start_j:end_j, :]
        Vj = V[:, :, start_j:end_j, :]

        k_block = np.arange(start_j, end_j)[None, :]

        for i in range(Tr):
            start_i = i * Q_BLOCK_SIZE
            end_i = min(start_i + Q_BLOCK_SIZE, seq_q)
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

            q_block = np.arange(start_i, end_i)[:, None] + offset
            causal_mask = q_block >= k_block
            causal_mask = causal_mask[
                None, None, :, :
            ]  # Broadcast to (b, h, block_i, block_j)

            scale = 1 / np.sqrt(Q.shape[-1])
            Qi_scaled = Qi * scale
            S_ij = np.einsum("...id,...jd->...ij", Qi_scaled, Kj)

            S_ij = np.where(causal_mask, S_ij, NEG_INF)

            m_block_ij = np.max(S_ij, axis=-1, keepdims=True)
            P_ij = np.exp(S_ij - m_block_ij)
            P_ij = np.where(causal_mask, P_ij, 0.0)

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


def flash_attention_causal_backward(Q, K, V, O, l, m, dO):
    dQ = np.zeros_like(Q)
    dK = np.zeros_like(K)
    dV = np.zeros_like(V)

    Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[-1])
    KV_BLOCK_SIZE = BLOCK_SIZE
    seq_q, seq_k = Q.shape[2], K.shape[2]
    Tr = int(np.ceil(seq_q / Q_BLOCK_SIZE))
    Tc = int(np.ceil(seq_k / KV_BLOCK_SIZE))
    offset = seq_k - seq_q

    for j in range(Tc):
        start_j = j * KV_BLOCK_SIZE
        end_j = min(start_j + KV_BLOCK_SIZE, seq_k)
        Kj = K[:, :, start_j:end_j, :]
        Vj = V[:, :, start_j:end_j, :]

        k_block = np.arange(start_j, end_j)[None, :]

        dKj_block = np.zeros_like(Kj)
        dVj_block = np.zeros_like(Vj)

        for i in range(Tr):
            start_i = i * Q_BLOCK_SIZE
            end_i = min(start_i + Q_BLOCK_SIZE, seq_q)
            Qi = Q[:, :, start_i:end_i, :]
            Oi = O[:, :, start_i:end_i, :]
            dOi = dO[:, :, start_i:end_i, :]
            li = l[:, :, start_i:end_i, :]
            mi = m[:, :, start_i:end_i, :]

            q_block = np.arange(start_i, end_i)[:, None] + offset
            causal_mask = q_block >= k_block
            causal_mask = causal_mask[None, None, :, :]

            scale = 1 / np.sqrt(Q.shape[-1])
            Qi_scaled = Qi * scale
            S_ij = np.einsum("...id,...jd->...ij", Qi_scaled, Kj)

            S_ij = np.where(causal_mask, S_ij, NEG_INF)

            P_ij = (1 / li) * np.exp(S_ij - mi)
            P_ij = np.where(causal_mask, P_ij, 0.0)

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


def flash_attention_causal(Q, K, V):
    return flash_attention_causal_forward(Q, K, V)[0]


if __name__ == "__main__":
    Q = np.random.randn(1, 2, 4096, 1024)
    K = np.random.randn(1, 2, 4096, 1024)
    V = np.random.randn(1, 2, 4096, 1024)

    for _ in range(10):
        start1 = time.time()
        out1 = flash_attention_causal(Q, K, V)
        end1 = time.time()

        start2 = time.time()
        out2 = normal_attention_causal(Q, K, V)
        end2 = time.time()

        t1 = (end1 - start1) * 1000
        t2 = (end2 - start2) * 1000

        print(f"{t1:.2f}ms, {t2:.2f}ms")
        print(np.allclose(out1, out2, atol=1e-5))

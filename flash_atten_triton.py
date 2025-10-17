import pytest
import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_M = 64
BLOCK_N = 64
BLOCK_D = 128  # Example head dim, adjust as needed


@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, M_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    stride_mb, stride_mh, stride_mm,
    SEQLEN_Q, SEQLEN_K,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n_base = tl.arange(0, BLOCK_N)

    q_ptrs = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    o_ptrs = O_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    l_ptrs = L_ptr + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_lm
    m_ptrs = M_ptr + pid_b * stride_mb + pid_h * stride_mh + offs_m * stride_mm

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < SEQLEN_Q, other=0.0)

    for start_n in range(0, SEQLEN_K, BLOCK_N):
        offs_n = start_n + offs_n_base
        k_ptrs = K_ptr + pid_b * stride_kb + pid_h * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        v_ptrs = V_ptr + pid_b * stride_vb + pid_h * stride_vh + offs_n[None, :] * stride_vn + offs_d[:, None] * stride_vd

        k = tl.load(k_ptrs, mask=offs_n[None, :] < SEQLEN_K, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[None, :] < SEQLEN_K, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * SCALE

        if CAUSAL:
            mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(mask, qk, -float("inf"))

        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])

        alpha = tl.exp(m_i - tl.maximum(m_i, m_ij))
        l_ij = tl.sum(p, 1)

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        l_i = l_i * alpha + l_ij
        m_i = tl.maximum(m_i, m_ij)

    tl.store(o_ptrs, (acc / l_i[:, None]).to(tl.float16), mask=offs_m[:, None] < SEQLEN_Q)
    tl.store(l_ptrs, l_i, mask=offs_m < SEQLEN_Q)
    tl.store(m_ptrs, m_i, mask=offs_m < SEQLEN_Q)


@triton.jit
def flash_attn_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, DO_ptr, DQ_ptr, L_ptr, M_ptr, Delta_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_lb, stride_lh, stride_lm,
    stride_mb, stride_mh, stride_mm,
    stride_deltab, stride_deltah, stride_deltam,
    SEQLEN_Q, SEQLEN_K,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n_base = tl.arange(0, BLOCK_N)

    q_ptrs = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    do_ptrs = DO_ptr + pid_b * stride_dob + pid_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
    dq_ptrs = DQ_ptr + pid_b * stride_dqb + pid_h * stride_dqh + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
    l_ptrs = L_ptr + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_lm
    m_ptrs = M_ptr + pid_b * stride_mb + pid_h * stride_mh + offs_m * stride_mm
    delta_ptrs = Delta_ptr + pid_b * stride_deltab + pid_h * stride_deltah + offs_m * stride_deltam

    q = tl.load(q_ptrs, mask=offs_m[:, None] < SEQLEN_Q, other=0.0)
    do = tl.load(do_ptrs, mask=offs_m[:, None] < SEQLEN_Q, other=0.0)
    l_i = tl.load(l_ptrs, mask=offs_m < SEQLEN_Q, other=0.0)
    m_i = tl.load(m_ptrs, mask=offs_m < SEQLEN_Q, other=0.0)
    delta = tl.load(delta_ptrs, mask=offs_m < SEQLEN_Q, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, SEQLEN_K, BLOCK_N):
        offs_n = start_n + offs_n_base
        k_ptrs = K_ptr + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V_ptr + pid_b * stride_vb + pid_h * stride_vh + offs_n[None, :] * stride_vn + offs_d[:, None] * stride_vd

        k = tl.load(k_ptrs, mask=offs_n[:, None] < SEQLEN_K, other=0.0).trans()
        v = tl.load(v_ptrs, mask=offs_n[None, :] < SEQLEN_K, other=0.0)

        qk = tl.dot(q, k) * SCALE - m_i[:, None]
        p = tl.exp(qk)

        if CAUSAL:
            mask = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(mask, p, 0.0)

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - delta[:, None]) / l_i[:, None]

        dq += tl.dot(ds.to(tl.float16), tl.trans(k))

    tl.store(dq_ptrs, (dq * SCALE).to(tl.float16), mask=offs_m[:, None] < SEQLEN_Q)


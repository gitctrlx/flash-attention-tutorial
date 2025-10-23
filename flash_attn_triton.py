import math
import torch
import triton
import triton.language as tl

BLOCK_M = 128
BLOCK_N = 128


@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    M_ptr,
    Mask_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_maskb,
    stride_maskn,
    SEQLEN_Q,
    SEQLEN_K,
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

    q_ptrs = (
        Q_ptr
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    o_ptrs = (
        O_ptr
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    l_ptrs = L_ptr + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_lm
    m_ptrs = M_ptr + pid_b * stride_mb + pid_h * stride_mh + offs_m * stride_mm
    # mask ptr computed per tile below

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < SEQLEN_Q, other=0.0)

    for start_n in range(0, SEQLEN_K, BLOCK_N):
        offs_n = start_n + offs_n_base
        k_ptrs = (
            K_ptr
            + pid_b * stride_kb
            + pid_h * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            V_ptr
            + pid_b * stride_vb
            + pid_h * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )

        k = tl.load(k_ptrs, mask=offs_n[:, None] < SEQLEN_K, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < SEQLEN_K, other=0.0)
        mask_ptrs = Mask_ptr + pid_b * stride_maskb + offs_n * stride_maskn
        key_mask = tl.load(mask_ptrs, mask=offs_n < SEQLEN_K, other=0).to(tl.int1)

        qk = tl.dot(q, tl.trans(k)) * SCALE
        valid_k = offs_n[None, :] < SEQLEN_K
        qk = tl.where(valid_k & (key_mask[None, :] > 0), qk, -float("inf"))
        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, -float("inf"))

        m_ij = tl.max(qk, 1)
        p = tl.where(valid_k & (key_mask[None, :] > 0), tl.exp(qk - m_ij[:, None]), 0.0)

        m_new = tl.maximum(m_i, m_ij)
        alpha_old = tl.exp(m_i - m_new)
        alpha_blk = tl.exp(m_ij - m_new)
        l_ij = tl.sum(p, 1) + 1e-6

        acc = acc * alpha_old[:, None] + tl.dot(
            (p * alpha_blk[:, None]).to(tl.float32), v.to(tl.float32)
        )
        l_i = l_i * alpha_old + l_ij * alpha_blk
        m_i = m_new

    tl.store(
        o_ptrs, (acc / l_i[:, None]).to(tl.float16), mask=offs_m[:, None] < SEQLEN_Q
    )
    tl.store(l_ptrs, l_i, mask=offs_m < SEQLEN_Q)
    tl.store(m_ptrs, m_i, mask=offs_m < SEQLEN_Q)


@triton.jit
def flash_attn_bwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    DO_ptr,
    DQ_ptr,
    DK_ptr,
    DV_ptr,
    L_ptr,
    M_ptr,
    Mask_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dqd,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_dvd,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_maskb,
    stride_maskn,
    SEQLEN_Q,
    SEQLEN_K,
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

    q_ptrs = (
        Q_ptr
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    o_ptrs = (
        O_ptr
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    do_ptrs = (
        DO_ptr
        + pid_b * stride_dob
        + pid_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :] * stride_dod
    )
    dq_ptrs = (
        DQ_ptr
        + pid_b * stride_dqb
        + pid_h * stride_dqh
        + offs_m[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd
    )
    l_ptrs = L_ptr + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_lm
    m_ptrs = M_ptr + pid_b * stride_mb + pid_h * stride_mh + offs_m * stride_mm
    mask_ptrs = Mask_ptr + pid_b * stride_maskb + offs_n_base * stride_maskn

    q = tl.load(q_ptrs, mask=offs_m[:, None] < SEQLEN_Q, other=0.0)
    o = tl.load(o_ptrs, mask=offs_m[:, None] < SEQLEN_Q, other=0.0)
    do = tl.load(do_ptrs, mask=offs_m[:, None] < SEQLEN_Q, other=0.0)
    l_i = tl.load(l_ptrs, mask=offs_m < SEQLEN_Q, other=0.0)
    m_i = tl.load(m_ptrs, mask=offs_m < SEQLEN_Q, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, SEQLEN_K, BLOCK_N):
        offs_n = start_n + offs_n_base
        k_ptrs = (
            K_ptr
            + pid_b * stride_kb
            + pid_h * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            V_ptr
            + pid_b * stride_vb
            + pid_h * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        dk_ptrs = (
            DK_ptr
            + pid_b * stride_dkb
            + pid_h * stride_dkh
            + offs_n[:, None] * stride_dkn
            + offs_d[None, :] * stride_dkd
        )
        dv_ptrs = (
            DV_ptr
            + pid_b * stride_dvb
            + pid_h * stride_dvh
            + offs_n[:, None] * stride_dvn
            + offs_d[None, :] * stride_dvd
        )
        mask_ptrs = Mask_ptr + pid_b * stride_maskb + offs_n * stride_maskn

        k = tl.load(k_ptrs, mask=offs_n[:, None] < SEQLEN_K, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < SEQLEN_K, other=0.0)
        key_mask = tl.load(mask_ptrs, mask=offs_n < SEQLEN_K, other=0).to(tl.int1)

        qk = tl.dot(q, tl.trans(k)) * SCALE - m_i[:, None]
        p = tl.exp(qk)
        valid_k = offs_n[None, :] < SEQLEN_K
        p = tl.where(valid_k & (key_mask[None, :] > 0), p, 0.0)
        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(causal_mask, p, 0.0)

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        delta = tl.sum(do * o, 1)
        ds = p * (dp - delta[:, None]) / l_i[:, None]

        dq += tl.dot(ds.to(tl.float32), k.to(tl.float32))

        dK_tile = tl.dot(tl.trans(ds).to(tl.float32), q.to(tl.float32)) * SCALE
        probs = p / (l_i[:, None])
        dV_tile = tl.dot(tl.trans(probs).to(tl.float32), do.to(tl.float32))
        mask_rows = offs_n[:, None] < SEQLEN_K
        tl.atomic_add(dk_ptrs, tl.where(mask_rows, dK_tile, 0.0))
        tl.atomic_add(dv_ptrs, tl.where(mask_rows, dV_tile, 0.0))

    tl.store(dq_ptrs, (dq * SCALE).to(tl.float16), mask=offs_m[:, None] < SEQLEN_Q)


def _cdiv(a, b):
    return (a + b - 1) // b


def _ensure_contig_fp16(x):
    if x.dtype not in (torch.float16, torch.bfloat16):
        x = x.to(torch.float16)
    return x.contiguous()


def flash_attention_triton_forward(Q, K, V, key_mask=None, causal=False):
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    B, H, M, D = Q.shape
    N = K.shape[2]
    Q = _ensure_contig_fp16(Q)
    K = _ensure_contig_fp16(K)
    V = _ensure_contig_fp16(V)

    if key_mask is None:
        key_mask = torch.ones(B, N, dtype=torch.int32, device=Q.device)
    else:
        key_mask = key_mask.to(torch.int32).contiguous()

    O = torch.empty_like(Q)
    L = torch.empty((B, H, M), dtype=torch.float32, device=Q.device)
    Mmax = torch.empty((B, H, M), dtype=torch.float32, device=Q.device)

    BLOCK_D = D
    grid = (B, H, _cdiv(M, BLOCK_M))
    scale = 1.0 / math.sqrt(D)

    flash_attn_fwd_kernel[grid](
        Q,
        K,
        V,
        O,
        L,
        Mmax,
        key_mask,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        Q.stride(3),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        V.stride(3),
        O.stride(0),
        O.stride(1),
        O.stride(2),
        O.stride(3),
        L.stride(0),
        L.stride(1),
        L.stride(2),
        Mmax.stride(0),
        Mmax.stride(1),
        Mmax.stride(2),
        key_mask.stride(0),
        key_mask.stride(1),
        M,
        N,
        SCALE=scale,
        CAUSAL=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return O, L, Mmax


class FlashAttnTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, key_mask=None, causal=False):
        O, L, Mmax = flash_attention_triton_forward(Q, K, V, key_mask, causal)
        ctx.save_for_backward(
            Q,
            K,
            V,
            O,
            L,
            Mmax,
            (
                key_mask
                if key_mask is not None
                else torch.ones(
                    Q.shape[0], K.shape[2], dtype=torch.int32, device=Q.device
                )
            ),
        )
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L, Mmax, key_mask = ctx.saved_tensors
        B, H, M, D = Q.shape
        N = K.shape[2]
        dO = dO.contiguous()
        dQ = torch.empty_like(Q)
        dK_fp32 = torch.zeros_like(K, dtype=torch.float32)
        dV_fp32 = torch.zeros_like(V, dtype=torch.float32)

        grid = (B, H, _cdiv(M, BLOCK_M))
        scale = 1.0 / math.sqrt(D)

        flash_attn_bwd_kernel[grid](
            Q,
            K,
            V,
            O,
            dO,
            dQ,
            dK_fp32,
            dV_fp32,
            L,
            Mmax,
            key_mask,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            K.stride(3),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            V.stride(3),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            O.stride(3),
            dO.stride(0),
            dO.stride(1),
            dO.stride(2),
            dO.stride(3),
            dQ.stride(0),
            dQ.stride(1),
            dQ.stride(2),
            dQ.stride(3),
            dK_fp32.stride(0),
            dK_fp32.stride(1),
            dK_fp32.stride(2),
            dK_fp32.stride(3),
            dV_fp32.stride(0),
            dV_fp32.stride(1),
            dV_fp32.stride(2),
            dV_fp32.stride(3),
            L.stride(0),
            L.stride(1),
            L.stride(2),
            Mmax.stride(0),
            Mmax.stride(1),
            Mmax.stride(2),
            key_mask.stride(0),
            key_mask.stride(1),
            M,
            N,
            SCALE=scale,
            CAUSAL=ctx.causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=D,
            num_warps=4,
            num_stages=2,
        )
        dK = dK_fp32.to(K.dtype)
        dV = dV_fp32.to(V.dtype)
        return dQ, dK, dV, None, None


def flash_attention_triton(Q, K, V, key_mask=None):
    return FlashAttnTritonFn.apply(Q, K, V, key_mask, False)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, M, N, D = 1, 2, 1024, 1024, 64
    Q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    key_mask = torch.ones(B, N, device="cuda", dtype=torch.int32)
    O, L, Mmax = flash_attention_triton_forward(Q, K, V, key_mask, causal=False)
    print(O.shape, L.shape, Mmax.shape)

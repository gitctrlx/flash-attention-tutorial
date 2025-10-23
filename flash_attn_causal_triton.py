import torch
from flash_atten_triton import FlashAttnTritonFn, flash_attention_triton_forward


def flash_attention_triton_causal_forward(Q, K, V, key_mask=None):
    return flash_attention_triton_forward(Q, K, V, key_mask, causal=True)


def flash_attention_triton_causal(Q, K, V, key_mask=None):
    return FlashAttnTritonFn.apply(Q, K, V, key_mask, True)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, M, N, D = 1, 2, 256, 256, 64
    Q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    O, L, Mmax = flash_attention_triton_causal_forward(Q, K, V)
    print(O.shape, L.shape, Mmax.shape)

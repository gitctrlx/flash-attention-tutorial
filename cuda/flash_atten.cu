#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(e)                                                          \
    if (e != cudaSuccess) {                                                    \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__,    \
               __LINE__);                                                      \
        exit(1);                                                               \
    }

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

const int Br = 32;
const int Bc = 32;

__global__ void flash_attention_forward_kernel(
    const float *Q, const float *K, const float *V, float *O, float *l_arr,
    float *m_arr, int N, int d, float scale) {
    const float neg_inf = -1e10f;
    const float epsilon = 1e-10f;

    int i = blockIdx.x;
    int bh_idx = blockIdx.y;
    int tx = threadIdx.x;

    size_t offset = bh_idx * static_cast<size_t>(N) * d;
    size_t offset_lm = bh_idx * static_cast<size_t>(N);

    const float *Q_bh = Q + offset;
    const float *K_bh = K + offset;
    const float *V_bh = V + offset;
    float *O_bh = O + offset;
    float *l_bh = l_arr + offset_lm;
    float *m_bh = m_arr + offset_lm;

    extern __shared__ float smem[];
    float *Qi = smem;
    float *Kj = Qi + Br * d;
    float *Vj = Kj + Bc * d;
    float *S_ij = Vj + Bc * d;
    float *Oi = S_ij + Br * Bc;
    float *li = Oi + Br * d;
    float *mi = li + Br;
    float *mi_new = mi + Br;
    float *li_new = mi_new + Br;
    float *m_block_ij = li_new + Br;

    int eff_br = min(Br, N - i * Br);

    // Load Qi, Oi, li, mi for this query block i
    int global_row = i * Br + tx;
    if (tx < eff_br && global_row < N) {
        for (int col = 0; col < d; col++) {
            Qi[tx * d + col] = Q_bh[global_row * d + col];
            Oi[tx * d + col] = 0.0f;  // Initialize Oi to 0
        }
        li[tx] = 0.0f;
        mi[tx] = neg_inf;
    } else if (tx < Br) {
        // Zero-pad for safety
        for (int col = 0; col < d; col++) {
            Qi[tx * d + col] = 0.0f;
            Oi[tx * d + col] = 0.0f;
        }
        li[tx] = 0.0f;
        mi[tx] = neg_inf;
    }

    __syncthreads();

    int Tc = CEIL_DIV(N, Bc);

    for (int j = 0; j < Tc; j++) {
        int eff_bc = min(Bc, N - j * Bc);

        // Load Kj, Vj for this key block j
        int global_col = j * Bc + tx;
        if (tx < eff_bc && global_col < N) {
            for (int col = 0; col < d; col++) {
                Kj[tx * d + col] = K_bh[global_col * d + col];
                Vj[tx * d + col] = V_bh[global_col * d + col];
            }
        } else if (tx < Bc) {
            // Zero-pad
            for (int col = 0; col < d; col++) {
                Kj[tx * d + col] = 0.0f;
                Vj[tx * d + col] = 0.0f;
            }
        }

        __syncthreads();

        // Compute S_ij with causal mask
        if (tx < eff_br) {
            for (int s_col = 0; s_col < eff_bc; s_col++) {
                float acc = 0.0f;
                for (int k = 0; k < d; k++) {
                    acc += Qi[tx * d + k] * Kj[s_col * d + k];
                }
                acc *= scale;
                S_ij[tx * Bc + s_col] = acc;
            }
        }

        __syncthreads();

        // Compute block max, softmax P_ij (overwrite S_ij), block sum
        if (tx < eff_br) {
            float row_m = neg_inf;
            for (int c = 0; c < eff_bc; c++) {
                row_m = fmaxf(row_m, S_ij[tx * Bc + c]);
            }
            m_block_ij[tx] = row_m;

            float row_l = 0.0f;
            for (int c = 0; c < eff_bc; c++) {
                float exp_val = expf(S_ij[tx * Bc + c] - row_m);
                S_ij[tx * Bc + c] = exp_val;
                row_l += exp_val;
            }

            mi_new[tx] = fmaxf(mi[tx], row_m);
            li_new[tx] = expf(mi[tx] - mi_new[tx]) * li[tx] +
                         expf(row_m - mi_new[tx]) * row_l;
        }

        __syncthreads();

        // Update Oi with P_ij * Vj
        if (tx < eff_br) {
            for (int col = 0; col < d; col++) {
                float acc = 0.0f;
                for (int c = 0; c < eff_bc; c++) {
                    acc += S_ij[tx * Bc + c] * Vj[c * d + col];
                }
                Oi[tx * d + col] = (expf(mi[tx] - mi_new[tx]) * li[tx] /
                                    li_new[tx]) * Oi[tx * d + col] +
                                   (expf(m_block_ij[tx] - mi_new[tx]) /
                                    li_new[tx]) * acc;
            }
        }

        __syncthreads();

        // Update mi and li
        if (tx < eff_br) {
            mi[tx] = mi_new[tx];
            li[tx] = li_new[tx];
        }

        __syncthreads();
    }

    // Write back O, l, m
    global_row = i * Br + tx;
    if (tx < eff_br && global_row < N) {
        for (int col = 0; col < d; col++) {
            O_bh[global_row * d + col] = Oi[tx * d + col];
        }
        l_bh[global_row] = li[tx] + epsilon;
        m_bh[global_row] = mi[tx];
    }
}

__global__ void flash_attention_backward_kernel(
    const float *Q, const float *K, const float *V, const float *O,
    const float *l_arr, const float *m_arr, const float *dO, float *dQ,
    float *dK, float *dV, int N, int d, float scale) {
    const float neg_inf = -1e10f;

    int j = blockIdx.x;
    int bh_idx = blockIdx.y;
    int tx = threadIdx.x;

    size_t offset = bh_idx * static_cast<size_t>(N) * d;
    size_t offset_lm = bh_idx * static_cast<size_t>(N);

    const float *Q_bh = Q + offset;
    const float *K_bh = K + offset;
    const float *V_bh = V + offset;
    const float *O_bh = O + offset;
    const float *l_bh = l_arr + offset_lm;
    const float *m_bh = m_arr + offset_lm;
    const float *dO_bh = dO + offset;
    float *dQ_bh = dQ + offset;
    float *dK_bh = dK + offset;
    float *dV_bh = dV + offset;

    extern __shared__ float smem[];
    float *Qi = smem;
    float *Kj = Qi + Br * d;
    float *Vj = Kj + Bc * d;
    float *S_ij = Vj + Bc * d;
    float *P_ij = S_ij + Br * Bc;
    float *dOi = P_ij + Br * Bc;
    float *Oi = dOi + Br * d;
    float *li = Oi + Br * d;
    float *mi = li + Br;
    float *dP_ij = mi + Br;
    float *dS_ij = dP_ij + Br * Bc;
    float *Di = dS_ij + Br * Bc;
    float *dKj_block = Di + Br;
    float *dVj_block = dKj_block + Bc * d;

    int eff_bc = min(Bc, N - j * Bc);

    // Initialize dKj_block, dVj_block to 0
    if (tx < eff_bc) {
        for (int col = 0; col < d; col++) {
            dKj_block[tx * d + col] = 0.0f;
            dVj_block[tx * d + col] = 0.0f;
        }
    } else if (tx < Bc) {
        for (int col = 0; col < d; col++) {
            dKj_block[tx * d + col] = 0.0f;
            dVj_block[tx * d + col] = 0.0f;
        }
    }

    __syncthreads();

    // Load Kj, Vj for this key block j
    int global_col = j * Bc + tx;
    if (tx < eff_bc && global_col < N) {
        for (int col = 0; col < d; col++) {
            Kj[tx * d + col] = K_bh[global_col * d + col];
            Vj[tx * d + col] = V_bh[global_col * d + col];
        }
    } else if (tx < Bc) {
        for (int col = 0; col < d; col++) {
            Kj[tx * d + col] = 0.0f;
            Vj[tx * d + col] = 0.0f;
        }
    }

    __syncthreads();

    int Tr = CEIL_DIV(N, Br);

    for (int i = 0; i < Tr; i++) {
        int eff_br = min(Br, N - i * Br);

        // Load Qi, Oi, dOi, li, mi for this query block i
        int global_row = i * Br + tx;
        if (tx < eff_br && global_row < N) {
            for (int col = 0; col < d; col++) {
                Qi[tx * d + col] = Q_bh[global_row * d + col];
                Oi[tx * d + col] = O_bh[global_row * d + col];
                dOi[tx * d + col] = dO_bh[global_row * d + col];
            }
            li[tx] = l_bh[global_row];
            mi[tx] = m_bh[global_row];
        } else if (tx < Br) {
            // Pad with safe values
            for (int col = 0; col < d; col++) {
                Qi[tx * d + col] = 0.0f;
                Oi[tx * d + col] = 0.0f;
                dOi[tx * d + col] = 0.0f;
            }
            li[tx] = 1.0f;  // Avoid div0
            mi[tx] = 0.0f;
        }

        __syncthreads();

        // Compute S_ij with causal mask
        if (tx < eff_br) {
            for (int s_col = 0; s_col < eff_bc; s_col++) {
                float acc = 0.0f;
                for (int k = 0; k < d; k++) {
                    acc += Qi[tx * d + k] * Kj[s_col * d + k];
                }
                acc *= scale;
                int g_row = i * Br + tx;
                int g_col = j * Bc + s_col;
                S_ij[tx * Bc + s_col] = (g_row >= g_col) ? acc : neg_inf;
            }
        }

        __syncthreads();

        // Compute P_ij = exp(S_ij - mi) / li, with mask
        if (tx < eff_br) {
            for (int s_col = 0; s_col < eff_bc; s_col++) {
                float exp_val = expf(S_ij[tx * Bc + s_col] - mi[tx]);
                float p = exp_val / li[tx];
                P_ij[tx * Bc + s_col] = p;
            }
        }

        __syncthreads();

        // Accumulate dVj_block += P_ij^T * dOi
        if (tx < eff_bc) {
            for (int col = 0; col < d; col++) {
                float acc = 0.0f;
                for (int r = 0; r < eff_br; r++) {
                    acc += P_ij[r * Bc + tx] * dOi[r * d + col];
                }
                dVj_block[tx * d + col] += acc;
            }
        }

        __syncthreads();

        // Compute dP_ij = dOi * Vj^T
        if (tx < eff_br) {
            for (int s_col = 0; s_col < eff_bc; s_col++) {
                float acc = 0.0f;
                for (int k = 0; k < d; k++) {
                    acc += dOi[tx * d + k] * Vj[s_col * d + k];
                }
                dP_ij[tx * Bc + s_col] = acc;
            }
        }

        __syncthreads();

        // Compute Di = sum(dOi * Oi, dim=-1)
        if (tx < eff_br) {
            float acc = 0.0f;
            for (int k = 0; k < d; k++) {
                acc += dOi[tx * d + k] * Oi[tx * d + k];
            }
            Di[tx] = acc;
        }

        __syncthreads();

        // Compute dS_ij = P_ij * (dP_ij - Di)
        if (tx < eff_br) {
            for (int s_col = 0; s_col < eff_bc; s_col++) {
                dS_ij[tx * Bc + s_col] = P_ij[tx * Bc + s_col] *
                                         (dP_ij[tx * Bc + s_col] - Di[tx]);
            }
        }

        __syncthreads();

        // Accumulate dQ += scale * dS_ij * Kj
        if (tx < eff_br) {
            int global_row = i * Br + tx;
            for (int col = 0; col < d; col++) {
                float acc = 0.0f;
                for (int c = 0; c < eff_bc; c++) {
                    acc += dS_ij[tx * Bc + c] * Kj[c * d + col];
                }
                atomicAdd(&dQ_bh[global_row * d + col], scale * acc);
            }
        }

        __syncthreads();

        // Accumulate dKj_block += scale * Qi^T * dS_ij
        if (tx < eff_bc) {
            for (int col = 0; col < d; col++) {
                float acc = 0.0f;
                for (int r = 0; r < eff_br; r++) {
                    acc += Qi[r * d + col] * dS_ij[r * Bc + tx];
                }
                dKj_block[tx * d + col] += scale * acc;
            }
        }

        __syncthreads();
    }

    // Write back dK, dV for this j
    global_col = j * Bc + tx;
    if (tx < eff_bc && global_col < N) {
        for (int col = 0; col < d; col++) {
            dK_bh[global_col * d + col] = dKj_block[tx * d + col];
            dV_bh[global_col * d + col] = dVj_block[tx * d + col];
        }
    }
}

int main() {
    const float NEG_INF = -1e10f;

    int batch = 1;
    int heads = 2;
    int N = 128;  // Small for test
    int d = 32;

    float scale = 1.0f / sqrtf(static_cast<float>(d));

    size_t size = static_cast<size_t>(batch) * heads * N * d * sizeof(float);
    size_t lm_size = static_cast<size_t>(batch) * heads * N * sizeof(float);

    float *Q_h = (float *)malloc(size);
    float *K_h = (float *)malloc(size);
    float *V_h = (float *)malloc(size);
    float *O_h = (float *)malloc(size);
    float *l_h = (float *)malloc(lm_size);
    float *m_h = (float *)malloc(lm_size);
    float *dO_h = (float *)malloc(size);
    float *dQ_h = (float *)malloc(size);
    float *dK_h = (float *)malloc(size);
    float *dV_h = (float *)malloc(size);

    // Deterministic initialization for reproducibility
    for (size_t i = 0; i < batch * heads * N * d; i++) {
        Q_h[i] = static_cast<float>(i % 10) / 10.0f - 0.5f;
        K_h[i] = static_cast<float>(i % 10) / 10.0f - 0.5f;
        V_h[i] = static_cast<float>(i % 10) / 10.0f - 0.5f;
        dO_h[i] = static_cast<float>(i % 10) / 10.0f - 0.5f;
    }

    for (size_t i = 0; i < batch * heads * N; i++) {
        l_h[i] = 0.0f;
        m_h[i] = NEG_INF;
    }
    memset(O_h, 0, size);
    memset(dQ_h, 0, size);
    memset(dK_h, 0, size);
    memset(dV_h, 0, size);

    float *Q_d, *K_d, *V_d, *O_d, *l_d, *m_d, *dO_d, *dQ_d, *dK_d, *dV_d;

    CHECK_CUDA(cudaMalloc(&Q_d, size));
    CHECK_CUDA(cudaMalloc(&K_d, size));
    CHECK_CUDA(cudaMalloc(&V_d, size));
    CHECK_CUDA(cudaMalloc(&O_d, size));
    CHECK_CUDA(cudaMalloc(&l_d, lm_size));
    CHECK_CUDA(cudaMalloc(&m_d, lm_size));
    CHECK_CUDA(cudaMalloc(&dO_d, size));
    CHECK_CUDA(cudaMalloc(&dQ_d, size));
    CHECK_CUDA(cudaMalloc(&dK_d, size));
    CHECK_CUDA(cudaMalloc(&dV_d, size));

    CHECK_CUDA(cudaMemcpy(Q_d, Q_h, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(K_d, K_h, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(V_d, V_h, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(O_d, O_h, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(l_d, l_h, lm_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m_d, m_h, lm_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dO_d, dO_h, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dQ_d, dQ_h, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK_d, dK_h, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV_d, dV_h, size, cudaMemcpyHostToDevice));

    int Tr = CEIL_DIV(N, Br);
    int Tc = CEIL_DIV(N, Bc);

    dim3 grid_f(Tr, batch * heads);
    size_t shared_size_f = ((Br * d) + (Bc * d) + (Bc * d) + (Br * Bc) +
                            (Br * d) + (Br * 5)) * sizeof(float);

    flash_attention_forward_kernel<<<grid_f, Br, shared_size_f>>>(
        Q_d, K_d, V_d, O_d, l_d, m_d, N, d, scale);
    CHECK_CUDA(cudaDeviceSynchronize());

    dim3 grid_b(Tc, batch * heads);
    size_t shared_size_b = ((Br * d) + (Bc * d) + (Bc * d) + (Br * Bc * 4) +
                            (Br * d * 2) + (Br * 3) + (Bc * d * 2)) *
                           sizeof(float);

    flash_attention_backward_kernel<<<grid_b, Bc, shared_size_b>>>(
        Q_d, K_d, V_d, O_d, l_d, m_d, dO_d, dQ_d, dK_d, dV_d, N, d, scale);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("Execution completed successfully.\n");

    // Free device memory
    CHECK_CUDA(cudaFree(Q_d));
    CHECK_CUDA(cudaFree(K_d));
    CHECK_CUDA(cudaFree(V_d));
    CHECK_CUDA(cudaFree(O_d));
    CHECK_CUDA(cudaFree(l_d));
    CHECK_CUDA(cudaFree(m_d));
    CHECK_CUDA(cudaFree(dO_d));
    CHECK_CUDA(cudaFree(dQ_d));
    CHECK_CUDA(cudaFree(dK_d));
    CHECK_CUDA(cudaFree(dV_d));

    // Free host memory
    free(Q_h);
    free(K_h);
    free(V_h);
    free(O_h);
    free(l_h);
    free(m_h);
    free(dO_h);
    free(dQ_h);
    free(dK_h);
    free(dV_h);

    return 0;
}
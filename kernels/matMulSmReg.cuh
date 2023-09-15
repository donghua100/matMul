#ifndef MATMUL_SM_REG_CUH
#define MATMUL_SM_REG_CUH
#include <cuda_runtime.h>
#define OFFSET(i, lda, j) ((i)*(lda) + (j))

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matMulSmReg(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    float res[TM][TN] = {0.0};
    float regA[TM] = {0.0};
    float regB[TN] = {0.0};

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int by = blockIdx.y;

    const int THREAD_NUM_PER_BLOCK = blockDim.x * blockDim.y;
    const int tid = ty * blockDim.x + tx;

    const int strideA = THREAD_NUM_PER_BLOCK / BK;
    int rowA = tid / BK;
    int colA = tid % BK;

    const int strideB = THREAD_NUM_PER_BLOCK / BN;
    int rowB = tid / BN;
    int colB = tid % BN;
    for (int idx = 0; idx < K; idx += BK) {
        for (int off = 0; off < BM; off += strideA) {
            if (by*BM + off + rowA < M && idx + colA < K)
                As[off + rowA][colA] = A[OFFSET(by*BM + off + rowA, K, idx + colA)];
            else 
                As[off + rowA][colA] = .0f;
        }
        for (int off = 0; off < BK; off += strideB) {
            if (idx + off + rowB < K && bx * BN + colB < N)
                Bs[off + rowB][colB] = B[OFFSET( idx + off + rowB, N, bx*BN + colB)];
            else
                Bs[off + rowB][colB] = .0f;
        }

        __syncthreads();

        for (int k = 0; k < BK; k++) {
            for (int  i = 0; i < TM; i++) {
                regA[i] = As[ty*TM + i][k];
            }
            for (int i = 0; i < TN; i++) {
                regB[i] = Bs[k][tx*TN + i];
            }

            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    res[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            if (by * BM + ty*TM + i < M && bx*BN + tx*TN + j < N)
                C[OFFSET(by*BM + ty*TM + i, N, bx*BN + tx*TN + j)] = res[i][j];
        }
    }
}
#endif

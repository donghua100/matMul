#ifndef MATMUL_NAIVE_CUH
#define MATMUL_NAIVE_CUH

#include <cuda_runtime.h>
__global__ void matMulNaive(float *A, float *B, float *C, int M, int N, int K) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < M && j < N) {
        float val = 0.0;
        for (int k = 0; k < K; k++) {
            val += A[i*K + k]*B[k*N + j];
        }
        C[i*N + j] = val;
    }
}
#endif

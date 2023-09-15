#ifndef MATMUL_SM_CUH
#define MATMUL_SM_CUH
#include <cuda_runtime.h>

template<const int TILE_WIDTH = 32>
__global__ void matMulSm(float *A, float *B, float *C, int M, int N, int K) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int by = blockIdx.y;
    __shared__ float Ad[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bd[TILE_WIDTH][TILE_WIDTH];
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float val = 0.0;
    for (int j = 0; j < K ; j += TILE_WIDTH) {
        if (row < M && j + tx < K)
            Ad[ty][tx] = A[row * K + j + tx];
        else 
            Ad[ty][tx] = .0f;
        if (col < N && j + ty < K) 
            Bd[ty][tx] = B[(j + ty)*N + col];
        else 
            Bd[ty][tx] = .0f;
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++) {
            val +=  Ad[ty][k]*Bd[k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) 
        C[row*N + col] = val;
}
#endif

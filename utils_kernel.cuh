#ifndef MY_UTILS_KERNEL_CUH
#define MY_UTILS_KERNEL_CUH

#include "kernel.cuh"

#define blockThreadNum 1024
#define warpSize 32
#define CEIL(M, N) (((M)-1)/(N) + 1)
#define TILE_WIDTH 32

void randomMat(float *A, int n) {
    for (int i = 0; i < n; i++) {
        A[i] = (rand() % 100)*0.1;
    }
}

void cmpMat(float *A, float *B, int n) {
    float maxErr = .0f;
    float aveErr = .0f;
    for (int i = 0; i < n; i++) {
        if (A[i] != 0) {
            float err = fabs((A[i] - B[i])/A[i]);
            if (err > maxErr) maxErr = err;
            aveErr += err;
        }
    }
    aveErr /= n;
    printf("Max Relative Error: %g, Average Relative Error: %g\n", maxErr, aveErr);
}

void matMulCpu(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float val = 0.0;
            for (int k = 0; k < K; k++) {
                val += A[i*K + k]*B[k*N + j];
            }
            C[i*N + j] = val;
        }
    }
} 

void runCublasSgemm(float *A, float *B, float *C, int M, int N, int K, cublasHandle_t handle) {
    float alpha = 1.0;
    float beta = 0.0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, B, K, A, N, &beta, C, M);
}

void runMatMulNaive(float *A, float *B, float *C, int M, int N, int K) {
        dim3 blockSize(warpSize, warpSize);
        dim3 gridSize(CEIL(N, warpSize), CEIL(M, warpSize));
        matMulNaive<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

void runMatMulSm(float *A, float *B, float *C, int M, int N, int K) {
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize(CEIL(N, TILE_WIDTH), CEIL(M, TILE_WIDTH));
    matMulSm<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

template<const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8, const int TN = 8>
void runMatMulSmReg(float *A, float *B, float *C, int M, int N, int K) {
    dim3 gridSize(CEIL(N, BN), CEIL(M, BM));
    dim3 blockSize(CEIL(BN, TN), CEIL(BM, TM));
    matMulSmReg<BM, BN, BK, TM, TN><<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

void runKernel(int n, float *A, float *B, float *C, int M, int N, int K, cublasHandle_t handle) {
    if (n == 0) {
        runCublasSgemm(A, B, C, M, N, K, handle);
    }
    else if( n == 1) {
        runMatMulNaive(A, B, C, M, N, K);
    }
    else if (n == 2) {
        runMatMulSm(A, B, C, M, N, K);
    }
    else if (n == 3) {
        runMatMulSmReg(A, B, C, M, N, K);
    }
    else {
        printf("ERROR KERNEL NUM");
        exit(-1);
    }
}
#endif

#include <getopt.h>
#include "utils.cuh"

static struct option long_options[] = {
    {"help", no_argument, 0, 'h'},
    {"kernel", required_argument, 0, 'k'},
    {"size", required_argument, 0, 's'},
    {0, 0, 0, 0}
};

void usage(char *s) {
    printf("Uasge %s [OPTIONS]\n", s);
    printf("Options:\n");
    printf("  -h, --help        Display this help message\n");
    printf("  -k, --kernel      Specify a kernel\n");
    printf("  -s, --size        Specify size\n");

}

static char kernelName[][20] = {
    "cublas",
    "Naive",
    "Sm",
    "SmReg"
};

int main(int argc, char *argv[]) {
    if (argc == 1) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    int opt;
    int n = 0;
    int M, N, K;
    while ((opt = getopt_long(argc, argv, "hk:s:", long_options, NULL)) != -1) {
        switch(opt) {
            case 'h':
                usage(argv[0]);
                exit(EXIT_SUCCESS);
            case 'k':
                n = atoi(optarg);
                break;
            case 's':
                int x;
                if ((x = sscanf(optarg, "%d %d %d", &M, &N, &K)) != 3) {
                    printf("optarg = %s \n", optarg);
                    printf("x = %d\n", x);
                    fprintf(stderr, "Invalid format for -s option.\n");
                    exit(EXIT_FAILURE);
                }
                break;
            case '?':
                exit(EXIT_FAILURE);
            default:
                break;
                
        }
    }

    srand(time(NULL));
    float *Ah, *Bh, *Ch, *Dh, *Eh, *Ad, *Bd, *Cd;
    Ah = (float *)malloc(sizeof(float)*M*K);
    Bh = (float *)malloc(sizeof(float)*K*N);
    Ch = (float *)malloc(sizeof(float)*M*N);
    Dh = (float *)malloc(sizeof(float)*M*N);
    Eh = (float *)malloc(sizeof(float)*M*N);

    randomMat(Ah, M*K);
    randomMat(Bh, K*N);

    cudaMalloc((void **)&Ad, sizeof(float)*M*K);
    cudaMalloc((void **)&Bd, sizeof(float)*K*N);
    cudaMalloc((void **)&Cd, sizeof(float)*M*N);
    cudaMemcpy(Ad, Ah, sizeof(float)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, Bh, sizeof(float)*K*N, cudaMemcpyHostToDevice);
    cublasHandle_t handle;
    cublasCreate(&handle);

    if (n != 0) {
        runKernel(0, Ad, Bd, Cd, M, N, K, handle);
        cudaMemcpy(Ch, Cd, sizeof(float)*M*N, cudaMemcpyDeviceToHost); 
        runKernel(n, Ad, Bd, Cd, M, N, K, handle);
        cudaMemcpy(Dh, Cd, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
        cmpMat(Ch, Dh, M*N);
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    int repat = 10;
    for (int i = 0; i < repat; i++) {
        runKernel(0, Ad, Bd, Cd, M, N, K, handle);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float millisecond;
    cudaEventElapsedTime(&millisecond, start, end);
    printf("[%-10s]     Time    %.2f ms, Throughput     %.2f GFLOPs,    [100.00%%] cublas\n",
           kernelName[0], millisecond/repat, 2.0*M*N*K*repat/(millisecond*1e6));

    cudaEventRecord(start);
    for (int i = 0; i < repat; i++) {
        runKernel(n, Ad, Bd, Cd, M, N, K, handle);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float cublas_millsecond = millisecond;
    cudaEventElapsedTime(&millisecond, start, end);
    printf("[%-10s]     Time    %.2f ms, Throughput     %.2f GFLOPs,    [%.2f%%] cublas\n",
           kernelName[n], millisecond/repat, 2.0*M*N*K*repat/(millisecond*1e6), 100*cublas_millsecond/millisecond);

    free(Ah);
    free(Bh);
    free(Ch);
    free(Dh);
    free(Eh);
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
    cublasDestroy(handle);
    return 0;
}


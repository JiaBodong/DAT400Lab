#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
//CUDA RunTime API
#include <cuda_runtime.h>

#define MATRIX_SIZE 1024

void printDeviceProp(const cudaDeviceProp &prop)
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %d.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %d.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %d.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

//CUDA Initialization
bool InitCUDA()
{
    int count;
    cudaGetDeviceCount(&count);
    if (count == 0) 
    {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    int i;
    for (i = 0; i < count; i++) 
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printDeviceProp(prop);
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) 
        {
            if (prop.major >= 1) 
            {
            break;
            }
        }
    }
    if (i == count) 
    {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }
    cudaSetDevice(i);
    return true;
}

// Generate Random Matrix Elements
void matgen(float* a, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            a[i * n + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX * RAND_MAX);
        }
    }
}

/* Task 1 & 2: Implement Your Kernel Function Here */
__global__ static void matMultCUDA(const float* a, const float* b, float* c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if((row < n) && (col < n))
    {
        float localv = 0;
        for(int k = 0; k< n ; k++)//each thread compute a row of a and a col of b
        {
            localv += a[row *n + k] * b[k * n+ col];
        }
        c[row * n +col] = localv;
    }
}

int main()
{
    if (!InitCUDA()) return 0; 
    cudaError_t err;
    float *a, *b, *c, *d;

    int n = MATRIX_SIZE;

    a = (float*)malloc(sizeof(float)* n * n); 
    b = (float*)malloc(sizeof(float)* n * n); 
    c = (float*)malloc(sizeof(float)* n * n); 
    d = (float*)malloc(sizeof(float)* n * n);

    srand(0);

    matgen(a, n);
    matgen(b, n);

    float *cuda_a, *cuda_b, *cuda_c;

    /* Task: Memory Allocation */
    cudaMalloc(&cuda_a, n * n * sizeof(float));

	cudaMalloc(&cuda_b, n * n * sizeof(float));

	cudaMalloc(&cuda_c, n * n * sizeof(float));


    /* Task: CUDA Memory Copy from Host to Device */
    cudaMemcpy(cuda_a, a, n * n * sizeof(float), cudaMemcpyHostToDevice);


	cudaMemcpy(cuda_b, b, n * n * sizeof(float), cudaMemcpyHostToDevice);


    /* Task: Number of Blocks and Threads && Dimention*/

    // 
    int blockSize = 32; // 
    int numBlocks = (n + blockSize - 1) / blockSize; // 
    dim3 dimGrid(numBlocks,numBlocks,1);
    dim3 dimBlock(blockSize,blockSize,1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Kernel Execution
    matMultCUDA << < dimGrid, dimBlock >> >(cuda_a , cuda_b , cuda_c , n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("GPU Time elapsed: %f ms \n",milliseconds);

    /* Task: CUDA Memory Copy from Device to Host */
    cudaMemcpy(c, cuda_c, n * n* sizeof(float), cudaMemcpyDeviceToHost);
    
    //Free
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);


 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // CPU Implementation of MatMul
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        { 
            double t = 0;
            for (int k = 0; k < n; k++)
            { 
                t += a[i * n + k] * b[k * n + j]; 
            } 
            //printf(" %f",t);
            d[i * n + j] = t; 
            
        } 
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("CPU Time elapsed: %f ms \n",milliseconds);
    // printf(" CPU result  \n");

    // for (int i = 0; i < n; i++) 
    // {
    //     for (int j = 0; j < n; j++) 
    //     {
    //         float x = c[i * n + j];
    //         printf(" %f",x);
    //     } 
    // }
    // printf(" GPU result  \n");
    // Check the accuracy of GPU results with CPU results
    float max_err = 0;
    float average_err = 0; 
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            if (d[i * n + j] != 0)
            { 
                float err = fabs((c[i * n + j] - d[i * n + j]) / d[i * n + j]);
                if (max_err < err) max_err = err; 
                average_err += err; 
            } 
        } 
    }
    printf("Max error: %g Average error: %g\n",max_err, average_err / (n * n));

    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        return 1; // 
    }

    return 0;
}
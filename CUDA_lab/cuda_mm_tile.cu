#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
//CUDA RunTime API
#include <cuda_runtime.h>

#define MATRIX_SIZE 1024
#define BLOCK_DIM 16
// #define numBlocks 32
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
    __shared__ float shared_a[BLOCK_DIM][BLOCK_DIM];
    __shared__ float shared_b[BLOCK_DIM][BLOCK_DIM];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float temp = 0;
    for(int i = 0; i<(n / BLOCK_DIM); i++)
    {   
        if(i > numBlocks)
        {
            shared_a[ty][tx] = a[(row+i-numBlocks) * n + i*BLOCK_DIM + tx];
            shared_b[ty][tx] = b[(i * BLOCK_DIM + ty)*n + col + i -numBlocks];
            __syncthreads();
        }
      

        else
        {
            shared_a[ty][tx] = a[row * n + i *BLOCK_DIM + tx];
            shared_b[ty][tx] = b[(i * BLOCK_DIM + ty)*n + col];
            __syncthreads();//sync all the threads , but in this part of code, we just see the thread [ty][tx]
        }
      
        for(int j = 0; j < BLOCK_DIM ; j++)
        {
            temp += shared_a[ty][j] * shared_b[j][tx];
        }
        __syncthreads();
        
    }

    if (row < n && col < n)
    {
        for (int k = (n / BLOCK_DIM) * BLOCK_DIM; k < n; k++)//compute the index which the remaind part begin after divided region
        {
            // printf("index:  %d", k);
            temp += a[row * n + k] * b[k * n + col];//each thread conmpute the remain part of the row of a and the col of b
        }
        c[row * n + col] = temp;

    }
}

/* 1 dimension */
// __global__ static void matMultCUDA(const float* a, const float* b, float* c, int n)
// {
//     __shared__ float shared_a[BLOCK_DIM * BLOCK_DIM];
//     __shared__ float shared_b[BLOCK_DIM * BLOCK_DIM];

//     int bx = blockIdx.x;
//     int tx = threadIdx.x;

//     int row = bx * blockDim.x + (tx / BLOCK_DIM);
//     int col = bx * blockDim.x + (tx % BLOCK_DIM);

//     float temp = 0;
    
//     for (int i = 0; i < (n / BLOCK_DIM); i++)
//     {   
//         shared_a[tx] = a[row * n + i * BLOCK_DIM + (tx % BLOCK_DIM)];
//         shared_b[tx] = b[(i * BLOCK_DIM + (tx / BLOCK_DIM)) * n + col];
//         __syncthreads();

//         for (int j = 0; j < BLOCK_DIM; j++)
//         {
//             temp += shared_a[(tx / BLOCK_DIM) * BLOCK_DIM + j] * shared_b[j * BLOCK_DIM + (tx % BLOCK_DIM)];
//         }
//         __syncthreads();
//     }

//     if (row < n && col < n)
//     {
//         for (int k = (n / BLOCK_DIM) * BLOCK_DIM; k < n; k++)
//         {
//             temp += a[row * n + k] * b[k * n + col];
//         }
//         c[row * n + col] = temp;
//     }
// }




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
    //int numBlocks = (n + BLOCK_DIM  - 1) / (BLOCK_DIM); // 
    printf("numblocks = %d", numBlocks);
    dim3 dimGrid(numBlocks,numBlocks,1);
    dim3 dimBlock(BLOCK_DIM,BLOCK_DIM,1);

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
#include "Reduction_Functions.cuh"
#include <device_launch_parameters.h>
#include "GlobalDefines.cuh"
#include "Device_Globals_Constants.cuh"
#include "Host_Globals.cuh"
#include <helper_cuda.h>
#include <stdlib.h>
#include <math.h>
#include<device_atomic_functions.h>
#include <sm_60_atomic_functions.h>
#include <cuda.h>
#include "Reduction_Kernels.cuh"
#include "Reduce_Templates.cuh"
#include "Array_Indexing_Functions.cuh"

__host__ void ReductionArraysInit(MEMDATA DATA)
{
#pragma omp parallel for
    for (int i = 0; i < NumberofBlocksIntegrator.z; i++)
    {
        for (int j = 0; j < NumberofBlocksIntegrator.y; j++)
        {
            for (int k = 0; k < NumberofBlocksIntegrator.x; k++)
            {
                int ind = ((i * NumberofBlocksIntegrator.y + j) * NumberofBlocksIntegrator.x + k);
                DATA->MaxTorqueReduction[ind] = 0.0;
                DATA->StepReduction[ind] = 0.0;
                DATA->dE_Reduction[ind] = 0.0;
                DATA->NewtonStepsReduction[ind] = 0;

            }
        }
    }
    return;
}

__device__ void warpReduce(volatile double* sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}
__global__ void SumReduction(double* idata, double* odata, int N)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = threadIdx.x + (blockIdx.x) * (2 * blockDim.x);

    if ((i) < N && tid < N)
    {
        sdata[tid] = idata[i] + idata[i + blockDim.x];
    }
    else
    {
        sdata[tid] = 0.0;
    }

    __syncthreads();


    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
    }


    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0)
    {
        odata[blockIdx.x] = sdata[0];
    }
    return;
}
__global__ void SumReductionStrided(double* idata, double* odata, int N, int Stride)
{
    extern __shared__ double SumStridedDat[];

    int tid = threadIdx.x;
    int i = (threadIdx.x + (blockIdx.x) * (2 * blockDim.x)) * Stride;

    if ((i / Stride + blockDim.x * Stride) < N && tid < N)
    {
        SumStridedDat[tid] = idata[i] + idata[i + blockDim.x * Stride];
    }
    else
    {
        SumStridedDat[tid] = 0.0;
    }

    __syncthreads();


    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
        {
            SumStridedDat[tid] += SumStridedDat[tid + s];
            __syncthreads();
        }
    }


    if (tid < 32) warpReduce(SumStridedDat, tid);

    if (tid == 0)
    {
        odata[blockIdx.x] = SumStridedDat[0];
    }
    return;
}
__global__ void SumReductionSmallData(double* idata, double* odata, int N)
{
    extern __shared__ double SumSmallDat[];

    int tid = threadIdx.x;
    int i = threadIdx.x + (blockIdx.x) * (blockDim.x);

    if (i < N)
    {
        SumSmallDat[tid] = idata[i];
    }
    else
    {
        SumSmallDat[tid] = 0.0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
        {
            SumSmallDat[tid] += SumSmallDat[tid + s];
            __syncthreads();
        }
    }

    if (tid < 32) warpReduce(SumSmallDat, tid);

    if (tid == 0)
    {
        odata[blockIdx.x] = SumSmallDat[0];
    }
    return;
}
__host__ void testreduction()
{
    int Nx = 128;
    int Ny = 128;
    int N = Nx * Ny; //4096

   // int Numblocks = (N / 64)/2; //block size halved minimum size is 64
   // int Numthreads = 64;

    int Numblocks = (N / 64); //block size halved minimum size is 64
    int Numthreads = 256;

    int SHARED_1 = Numthreads * sizeof(double);
    int SHARED_2 = Numblocks * sizeof(double);
    double* input_h = (double*)malloc(N * sizeof(double));
    double result;

    double* input_d;
    double* output_d;

    //initialise array
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            input_h[i * Ny + j] = 1.0;
        }
    }
    printf("Sum of array elements reduction test \n");
    printf("Predicted answer is %d \n", N);
    checkCudaErrors(cudaMalloc((void**)&input_d, N * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&output_d, Numblocks * sizeof(double)));

    checkCudaErrors(cudaMemcpy(input_d, input_h, N * sizeof(double), cudaMemcpyHostToDevice));

    SumReduction << <Numblocks, Numthreads, SHARED_1 >> > (input_d, output_d, N);
    checkCudaErrors(cudaDeviceSynchronize());

    SumReduction << <1, Numblocks / 2, SHARED_2 >> > (output_d, output_d, Numblocks);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(&result, output_d, sizeof(double), cudaMemcpyDeviceToHost));

    printf("Parallel reduction result = %d \n", (int)result);


    return;
}
__host__ void TestMaxReduction()
{
    int Nx = 128;
    int Ny = 128;
    int N = Nx * Ny; //4096

   // int Numblocks = (N / 64)/2; //block size halved minimum size is 64
   // int Numthreads = 64;

    int Numblocks = (N / 64); //block size halved minimum size is 64
    int Numthreads = 256;

    int SHARED_1 = Numthreads * sizeof(double);
    int SHARED_2 = Numblocks * sizeof(double);
    double* input_h = (double*)malloc(N * sizeof(double));
    double result;

    double* input_d;
    double* output_d;

    //initialise array
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            input_h[i * Ny + j] = (i + 1);
        }
    }
    printf("Max Value of array reduction test \n");
    printf("Predicted answer is %d \n", Nx);
    checkCudaErrors(cudaMalloc((void**)&input_d, N * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&output_d, Numblocks * sizeof(double)));

    checkCudaErrors(cudaMemcpy(input_d, input_h, N * sizeof(double), cudaMemcpyHostToDevice));

    MaxReduction << <Numblocks, Numthreads, SHARED_1 >> > (input_d, output_d, N);
    checkCudaErrors(cudaDeviceSynchronize());

    MaxReduction << <1, Numblocks / 2, SHARED_2 >> > (output_d, output_d, Numblocks);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(&result, output_d, sizeof(double), cudaMemcpyDeviceToHost));

    printf("Parallel reduction result = %d \n", (int)result);


    return;
}
__device__ void warpMinReduce(volatile double* sdata, int tid)
{
    sdata[tid] = fmin(sdata[tid], sdata[tid + 32]);
    sdata[tid] = fmin(sdata[tid], sdata[tid + 16]);
    sdata[tid] = fmin(sdata[tid], sdata[tid + 8]);
    sdata[tid] = fmin(sdata[tid], sdata[tid + 4]);
    sdata[tid] = fmin(sdata[tid], sdata[tid + 2]);
    sdata[tid] = fmin(sdata[tid], sdata[tid + 1]);
}
__device__ void warpMaxReduce(volatile double* sdata, int tid)
{
    sdata[tid] = fmax(sdata[tid], sdata[tid + 32]);
    sdata[tid] = fmax(sdata[tid], sdata[tid + 16]);
    sdata[tid] = fmax(sdata[tid], sdata[tid + 8]);
    sdata[tid] = fmax(sdata[tid], sdata[tid + 4]);
    sdata[tid] = fmax(sdata[tid], sdata[tid + 2]);
    sdata[tid] = fmax(sdata[tid], sdata[tid + 1]);
}
__global__ void MaxReduction(double* idata, double* odata, int N)
{
    extern __shared__ double MaxDat[];

    int tid = threadIdx.x;
    int i = threadIdx.x + (blockIdx.x) * (2 * blockDim.x);

    if ((i) < N && tid < N)
    {
        MaxDat[tid] = fmax(idata[i], idata[i + blockDim.x]);
    }
    else
    {
        MaxDat[tid] = 0.0;
    }

    __syncthreads();


    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
        {
            MaxDat[tid] = fmax(MaxDat[tid], MaxDat[tid + s]);
            __syncthreads();
        }
    }


    if (tid < 32) warpMaxReduce(MaxDat, tid);

    if (tid == 0)
    {
        odata[blockIdx.x] = MaxDat[0];
    }
    return;
}
__global__ void MinReduction(double* idata, double* odata, int N)
{
    extern __shared__ double MinDat[];

    int tid = threadIdx.x;
    int i = threadIdx.x + (blockIdx.x) * (2 * blockDim.x);

    if (i < N && tid < N)
    {
        MinDat[tid] = fmin(idata[i], idata[i + blockDim.x]);
    }
    else
    {
        MinDat[tid] = 0.0;
    }

    __syncthreads();


    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
        {
            MinDat[tid] = fmin(MinDat[tid], MinDat[tid + s]);
            __syncthreads();
        }
    }


    if (tid < 32) warpMinReduce(MinDat, tid);

    if (tid == 0)
    {
        odata[blockIdx.x] = MinDat[0];
    }
    return;
}
__global__ void SumEnergyReduction(double* idata_M, double* idata_H, double* odata, int N)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = threadIdx.x + (blockIdx.x) * (2 * blockDim.x);

    if ((i) < N && tid < N)
    {
        sdata[tid] = idata_M[i] * idata_H[i] + idata_M[i + blockDim.x] * idata_H[i + blockDim.x];
    }
    else
    {
        sdata[tid] = 0.0;
    }

    __syncthreads();


    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
    }


    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0)
    {
        odata[blockIdx.x] = sdata[0];
    }
    return;
}
__global__ void MaxReductionSmallN(double* idata, double* odata, int N)
{
    extern __shared__ double MaxDat[];

    int tid = threadIdx.x;

    if (tid == 0)
    {
        double max = idata[0];
        for (int n = 0; n < N; n++)
        {
            max = fmax(max, idata[n]);
        }
        odata[0] = max;
    }
    /* int i = threadIdx.x + (blockIdx.x) * (2 * blockDim.x);

     if ((i) < N && tid < N)
     {
         MaxDat[tid] =idata[i];
     }
     else
     {
         MaxDat[tid] = 0.0;
     }

     __syncthreads();

     if (tid < 32) warpMaxReduce(MaxDat, tid);

     if (tid == 0)
     {
         odata[blockIdx.x] = MaxDat[0];
     }*/
    return;
}
__global__ void MinReductionSmallN(double* idata, double* odata, int N)
{
    extern __shared__ double MinDat[];

    int tid = threadIdx.x;

    if (tid == 0)
    {
        double min = idata[0];
        for (int n = 0; n < N; n++)
        {
            min = fmin(min, idata[n]);
        }
        odata[0] = min;
    }
    /* int i = threadIdx.x + (blockIdx.x) * (2 * blockDim.x);

     if ((i) < N && tid < N)
     {
         MinDat[tid] = idata[i];
     }
     else
     {
         MinDat[tid] = 0.0;
     }

     __syncthreads();

     if (tid < 32) warpMinReduce(MinDat, tid);

     if (tid == 0)
     {
         odata[blockIdx.x] = MinDat[0];
     }*/
    return;
}
__global__ void SumReductionSmallN(double* idata, double* odata, int N)
{
    extern __shared__ double SumSmallDat[];

    int tid = threadIdx.x;

    if (tid == 0)
    {
        double Sum = 0;
        for (int n = 0; n < N; n++)
        {
            Sum += idata[n];
        }
        odata[0] = Sum;
    }

    /* if (tid < N)
     {
         SumSmallDat[tid] = idata[tid];
     }
     else
     {
         SumSmallDat[tid] = 0.0;
     }

     __syncthreads();

     if (tid < 32) warpReduce(SumSmallDat, tid);

     if (tid == 0)
     {
         odata[blockIdx.x] = SumSmallDat[0];
     }*/
    return;
}
__global__ void SumEnergyReductionSmallN(double* idata_M, double* idata_H, double* odata, int N)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;

    if (tid == 0)
    {
        double SumEnergy = idata_M[0] * idata_H[0];

        for (int n = 1; n < N; n++)
        {
            SumEnergy += idata_M[n] * idata_H[n];
        }
        odata[0] = SumEnergy;
    }

    /* if ((i) < N && tid < N)
     {
         sdata[tid] = idata_M[i] * idata_H[i];
     }
     else
     {
         sdata[tid] = 0.0;
     }

     __syncthreads();

     if (tid < 32) warpReduce(sdata, tid);

     if (tid == 0)
     {
         odata[blockIdx.x] = sdata[0];
     }*/
    return;
}
__global__ void SumZeemanEnergyReduction_x(double* idata_M, double* idata_H, double* odata, int N)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = threadIdx.x + (blockIdx.x) * (2 * blockDim.x);

    if ((i) < N && tid < N)
    {
        sdata[tid] = idata_M[i] * AMPx + idata_M[i + blockDim.x] * AMPx;
    }
    else
    {
        sdata[tid] = 0.0;
    }

    __syncthreads();


    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
    }


    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0)
    {
        odata[blockIdx.x] = sdata[0];
    }
    return;
}
__global__ void SumZeemanEnergyReduction_y(double* idata_M, double* idata_H, double* odata, int N)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = threadIdx.x + (blockIdx.x) * (2 * blockDim.x);

    if ((i) < N && tid < N)
    {
        sdata[tid] = idata_M[i] * AMPy + idata_M[i + blockDim.x] * AMPy;
    }
    else
    {
        sdata[tid] = 0.0;
    }

    __syncthreads();


    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
    }


    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0)
    {
        odata[blockIdx.x] = sdata[0];
    }
    return;
}
__global__ void SumZeemanEnergyReduction_z(double* idata_M, double* idata_H, double* odata, int N)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = threadIdx.x + (blockIdx.x) * (2 * blockDim.x);

    if ((i) < N && tid < N)
    {
        sdata[tid] = idata_M[i] * AMPz + idata_M[i + blockDim.x] * AMPz;
    }
    else
    {
        sdata[tid] = 0.0;
    }

    __syncthreads();


    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
    }


    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0)
    {
        odata[blockIdx.x] = sdata[0];
    }
    return;
}
__global__ void SumZeemanEnergyReductionSmallN_x(double* idata_M, double* idata_H, double* odata, int N)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;

    if (tid == 0)
    {
        double SumEnergy = idata_M[0] * AMPx;

        for (int n = 1; n < N; n++)
        {
            SumEnergy += idata_M[n] * idata_H[n];
        }
        odata[0] = SumEnergy;
    }

    return;
}
__global__ void SumZeemanEnergyReductionSmallN_y(double* idata_M, double* idata_H, double* odata, int N)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;

    if (tid == 0)
    {
        double SumEnergy = idata_M[0] * AMPy;

        for (int n = 1; n < N; n++)
        {
            SumEnergy += idata_M[n] * idata_H[n];
        }
        odata[0] = SumEnergy;
    }

    return;
}
__global__ void SumZeemanEnergyReductionSmallN_z(double* idata_M, double* idata_H, double* odata, int N)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;

    if (tid == 0)
    {
        double SumEnergy = idata_M[0] * AMPz;

        for (int n = 1; n < N; n++)
        {
            SumEnergy += idata_M[n] * idata_H[n];
        }
        odata[0] = SumEnergy;
    }

    return;
}

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}
////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction
// kernel For the kernels >= 3, we set threads / block to the minimum of
// maxThreads and n/2. For kernels < 3, we set to the minimum of maxThreads and
// n.  For kernel 6, we observe the maximum specified number of blocks, because
// each thread in that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks,
    int maxThreads, int& blocks, int& threads) {
    // get device capability, to avoid block/grid size exceed the upper bound
    cudaDeviceProp prop;
    int device;
    checkCudaErrors(cudaGetDevice(&device));
    checkCudaErrors(cudaGetDeviceProperties(&prop, device));

    if (whichKernel < 3) {
        threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
        blocks = (n + threads - 1) / threads;
    }
    else {
        threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }

    if ((float)threads * blocks >
        (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0]) {
        printf(
            "Grid size <%d> exceeds the device capability <%d>, set block size as "
            "%d (original %d)\n",
            blocks, prop.maxGridSize[0], threads * 2, threads);

        blocks /= 2;
        threads *= 2;
    }

    if (whichKernel >= 6) {
        blocks = MIN(maxBlocks, blocks);
    }
}

double Reduce_Sum(int  n, int  numThreads, int  numBlocks, int  maxThreads, int  maxBlocks,
    double* d_idata, double* d_odata) 
{
    double gpu_result = 0;

    // execute the kernel, first pass
    reduce<double>(n, numThreads, numBlocks,6, d_idata, d_odata);

    // check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    int s = numBlocks;
    int kernel = 6;
    int threads = 0, blocks = 0;

    while (s > 1) 
    {
     threads = 0, blocks = 0;

        getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks,threads);
  
        reduce<double>(s, threads, blocks, kernel, d_odata, d_odata);

            s = (s + (threads * 2 - 1)) / (threads * 2);
        
    }
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(&gpu_result, d_odata, sizeof(double), cudaMemcpyDeviceToHost));

return gpu_result;
}

double Reduce_Max(int  n, int  numThreads, int  numBlocks, int  maxThreads, int  maxBlocks,
    double* d_idata, double* d_odata)
{
    double gpu_result = 0;

    // execute the kernel, first pass
    reduce_Max<double>(n, numThreads, numBlocks, 6, d_idata, d_odata);

    // check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    int s = numBlocks;
    int kernel = 6;
    int threads = 0, blocks = 0;

    while (s > 1)
    {
        threads = 0, blocks = 0;

        getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);

        reduce_Max<double>(s, threads, blocks, kernel, d_odata, d_odata);

        s = (s + (threads * 2 - 1)) / (threads * 2);

    }
    cudaDeviceSynchronize();

        checkCudaErrors(cudaMemcpy(&gpu_result, d_odata, sizeof(double), cudaMemcpyDeviceToHost));

    return gpu_result;
}

double Reduce_Min(int  n, int  numThreads, int  numBlocks, int  maxThreads, int  maxBlocks,
    double* d_idata, double* d_odata)
{
    double gpu_result = 0;

    // execute the kernel, first pass
    reduce_Min<double>(n, numThreads, numBlocks, 6, d_idata, d_odata);

    // check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    int s = numBlocks;
    int kernel = 6;
    int threads = 0, blocks = 0;

    while (s > 1)
    {
        threads = 0, blocks = 0;

        getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);

        reduce_Min<double>(s, threads, blocks, kernel, d_odata, d_odata);

        s = (s + (threads * 2 - 1)) / (threads * 2);

    }
    cudaDeviceSynchronize();

        checkCudaErrors(cudaMemcpy(&gpu_result, d_odata, sizeof(double), cudaMemcpyDeviceToHost));

    return gpu_result;
}

double Reduce_Energy(int  n, int  numThreads, int  numBlocks, int  maxThreads, int  maxBlocks,
    double* M_idata,double* H_idata, double* E_odata)
{
    double gpu_result = 0;
    // execute the kernel, first pass
    reduce_Energy<double>(n, numThreads, numBlocks, 6, M_idata,H_idata,E_odata);

    // check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    int s = numBlocks;
    int kernel = 6;
    int threads = 0, blocks = 0;

    while (s > 1)
    {
        threads = 0, blocks = 0;

        getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);
        reduce<double>(s, threads, blocks, kernel, E_odata, E_odata);

        s = (s + (threads * 2 - 1)) / (threads * 2);

    }
    cudaDeviceSynchronize();

        checkCudaErrors(cudaMemcpy(&gpu_result, E_odata, sizeof(double), cudaMemcpyDeviceToHost));

    return gpu_result;
}

double StepError_Reduction()
{
    int maxThreads = 256;  // number of threads per block
    int whichKernel = 6;
    int maxBlocks = 64;
    int size = NUM_h * NUMY_h * NUMZ_h;
    int numBlocks = 0;
    int numThreads = 0;

    getNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks,
        numThreads);

    double result = Reduce_Max(size, numThreads, numBlocks, maxThreads, maxBlocks,
                        &(DEVICE_PTR_STRUCT.M->M[mind_h(8,0,0,0,0)]),
                        (DEVICE_PTR_STRUCT.DATA->StepReduction));

    return (result);
}
double MaxTorque_Reduction()
{
    int maxThreads = 256;  // number of threads per block
    int whichKernel = 6;
    int maxBlocks = 64;
    int size = NUM_h * NUMY_h * NUMZ_h;
    int numBlocks = 0;
    int numThreads = 0;

    getNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks,
        numThreads);

    double result = Reduce_Max(size, numThreads, numBlocks, maxThreads, maxBlocks,
        &(DEVICE_PTR_STRUCT.M->M[mind_h(8, 0, 0, 0, 1)]),
        (DEVICE_PTR_STRUCT.DATA->MaxTorqueReduction));

    return (result);
}
double MinTorque_Reduction()
{
    int maxThreads = 256;  // number of threads per block
    int whichKernel = 6;
    int maxBlocks = 64;
    int size = NUM_h * NUMY_h * NUMZ_h;
    int numBlocks = 0;
    int numThreads = 0;

    getNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks,
        numThreads);

    double result = Reduce_Min(size, numThreads, numBlocks, maxThreads, maxBlocks,
        &(DEVICE_PTR_STRUCT.M->M[mind_h(8, 0, 0, 0, 2)]),
        (DEVICE_PTR_STRUCT.DATA->dE_Reduction));

    return (result);
}
#ifndef REDUCTION_KERNELS_CUH
#define REDUCTION_KERNELS_CUH

#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "Reduce_Templates.cuh"
#include <helper_cuda.h>
#include "GlobalDefines.cuh"

namespace cg = cooperative_groups;

extern "C" bool isPow2(unsigned int x);

// Utility class used to avoid linker errors with extern
//unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator T* ()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T* () const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

// specialize for double to avoid unaligned memory access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator double* ()
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }

    __device__ inline operator const double* () const
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }
};

//Template class for general sum
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(T* g_idata, T* g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T* sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i + blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    cg::sync(cta);


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    cg::sync(cta);

    if ((blockSize >= 256) && (tid < 128))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

    cg::sync(cta);

    if ((blockSize >= 128) && (tid < 64))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 64];
    }

    cg::sync(cta);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    if (cta.thread_rank() < 32)
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >= 64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
        {
            mySum += tile32.shfl_down(mySum, offset);
        }
    }

    // write result for this block to global mem
    if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}

//Template class for general Max
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6_Max(T* g_idata, T* g_odata, unsigned int n)
{
    // Handle to thread block group 
    T* sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum = MAX(g_idata[i],mySum);

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) {
            mySum = MAX(g_idata[i + blockSize], mySum);
        }

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
  
    __syncthreads();

    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = MAX(sdata[tid + 256],mySum);
    }

    __syncthreads();

    if ((blockSize >= 256) && (tid < 128))
    {
        sdata[tid] = mySum = MAX(sdata[tid + 128],mySum);
    }

    __syncthreads();

    if ((blockSize >= 128) && (tid < 64))
    {
        sdata[tid] = mySum =MAX(sdata[tid + 64],mySum);
    }

    __syncthreads();

    if ((blockSize >= 64) && (tid < 32))
    {     
        sdata[tid] = mySum = MAX(sdata[tid + 32], mySum);
    }

    __syncthreads();

    if ((blockSize >= 32) && (tid < 16))
    {
        sdata[tid] = mySum = MAX(sdata[tid + 16], mySum);
    }

    __syncthreads();

    if ((blockSize >= 16) && (tid < 8))
    {
        sdata[tid] = mySum = MAX(sdata[tid + 8], mySum);
    }

    __syncthreads();

    if ((blockSize >= 8) && (tid < 4))
    {
        sdata[tid] = mySum = MAX(sdata[tid + 4], mySum);
    }

    __syncthreads();

    if ((blockSize >= 4) && (tid < 2))
    {
        sdata[tid] = mySum = MAX(sdata[tid + 2], mySum);
    }

    __syncthreads();

    if ((blockSize >= 2) && (tid < 1))
    {
        sdata[tid] = mySum = MAX(sdata[tid + 1], mySum);
    }
    __syncthreads();
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

//Template class for general Max
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6_Min(T* g_idata, T* g_odata, unsigned int n)
{
    // Handle to thread block group 
    T* sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    T mySum = INT_MAX;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum = MIN(g_idata[i], mySum);

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) {
            mySum = MIN(g_idata[i + blockSize], mySum);
        }

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;

    __syncthreads();

    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = MIN(sdata[tid + 256], mySum);
    }

    __syncthreads();

    if ((blockSize >= 256) && (tid < 128))
    {
        sdata[tid] = mySum = MIN(sdata[tid + 128], mySum);
    }

    __syncthreads();

    if ((blockSize >= 128) && (tid < 64))
    {
        sdata[tid] = mySum = MIN(sdata[tid + 64], mySum);
    }

    __syncthreads();

    if ((blockSize >= 64) && (tid < 32))
    {
        sdata[tid] = mySum = MIN(sdata[tid + 32], mySum);
    }

    __syncthreads();

    if ((blockSize >= 32) && (tid < 16))
    {
        sdata[tid] = mySum = MIN(sdata[tid + 16], mySum);
    }

    __syncthreads();

    if ((blockSize >= 16) && (tid < 8))
    {
        sdata[tid] = mySum = MIN(sdata[tid + 8], mySum);
    }

    __syncthreads();

    if ((blockSize >= 8) && (tid < 4))
    {
        sdata[tid] = mySum = MIN(sdata[tid + 4], mySum);
    }

    __syncthreads();

    if ((blockSize >= 4) && (tid < 2))
    {
        sdata[tid] = mySum = MIN(sdata[tid + 2], mySum);
    }

    __syncthreads();

    if ((blockSize >= 2) && (tid < 1))
    {
        sdata[tid] = mySum = MIN(sdata[tid + 1], mySum);
    }
    __syncthreads();
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

//Template class for sum of M.H
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce_6_ProdSum(T* M_idata, T* H_idata, T* E_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T* sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += M_idata[i]+ H_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += M_idata[i + blockSize]+ H_idata[i + blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    cg::sync(cta);


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    cg::sync(cta);

    if ((blockSize >= 256) && (tid < 128))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

    cg::sync(cta);

    if ((blockSize >= 128) && (tid < 64))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 64];
    }

    cg::sync(cta);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    if (cta.thread_rank() < 32)
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >= 64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
        {
            mySum += tile32.shfl_down(mySum, offset);
        }
    }

    // write result for this block to global mem
    if (cta.thread_rank() == 0) E_odata[blockIdx.x] = mySum;
}


// Wrapper function for kernel launch
template <class T>
void
reduce(int size, int threads, int blocks,
    int whichKernel, T* d_idata, T* d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);


    if (isPow2(size))
    {
        switch (threads)
        {
        case 512:
            reduce6<T, 512, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 256:
            reduce6<T, 256, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 128:
            reduce6<T, 128, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 64:
            reduce6<T, 64, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 32:
            reduce6<T, 32, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 16:
            reduce6<T, 16, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  8:
            reduce6<T, 8, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  4:
            reduce6<T, 4, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  2:
            reduce6<T, 2, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  1:
            reduce6<T, 1, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;
        }
    }
    else
    {
        switch (threads)
        {
        case 512:
            reduce6<T, 512, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 256:
            reduce6<T, 256, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 128:
            reduce6<T, 128, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 64:
            reduce6<T, 64, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 32:
            reduce6<T, 32, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 16:
            reduce6<T, 16, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  8:
            reduce6<T, 8, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  4:
            reduce6<T, 4, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  2:
            reduce6<T, 2, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  1:
            reduce6<T, 1, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;
        }
    }
}

// Wrapper function for kernel launch
template <class T>
void
reduce_Max(int size, int threads, int blocks,
    int whichKernel, T* d_idata, T* d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);


    if (isPow2(size))
    {
        switch (threads)
        {
        case 512:
            reduce6_Max<T, 512, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 256:
            reduce6_Max<T, 256, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 128:
            reduce6_Max<T, 128, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 64:
            reduce6_Max<T, 64, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 32:
            reduce6_Max<T, 32, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 16:
            reduce6_Max<T, 16, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  8:
            reduce6_Max<T, 8, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  4:
            reduce6_Max<T, 4, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  2:
            reduce6_Max<T, 2, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  1:
            reduce6_Max<T, 1, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;
        }
    }
    else
    {
        switch (threads)
        {
        case 512:
            reduce6_Max<T, 512, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 256:
            reduce6_Max<T, 256, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 128:
            reduce6_Max<T, 128, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 64:
            reduce6_Max<T, 64, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 32:
            reduce6_Max<T, 32, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 16:
            reduce6_Max<T, 16, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  8:
            reduce6_Max<T, 8, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  4:
            reduce6_Max<T, 4, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  2:
            reduce6_Max<T, 2, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  1:
            reduce6_Max<T, 1, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;
        }
    }
}

// Wrapper function for kernel launch
template <class T>
void
reduce_Min(int size, int threads, int blocks,
    int whichKernel, T* d_idata, T* d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);


    if (isPow2(size))
    {
        switch (threads)
        {
        case 512:
            reduce6_Min<T, 512, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 256:
            reduce6_Min<T, 256, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 128:
            reduce6_Min<T, 128, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 64:
            reduce6_Min<T, 64, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 32:
            reduce6_Min<T, 32, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 16:
            reduce6_Min<T, 16, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  8:
            reduce6_Min<T, 8, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  4:
            reduce6_Min<T, 4, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  2:
            reduce6_Min<T, 2, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  1:
            reduce6_Min<T, 1, true> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;
        }
    }
    else
    {
        switch (threads)
        {
        case 512:
            reduce6_Min<T, 512, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 256:
            reduce6_Min<T, 256, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 128:
            reduce6_Min<T, 128, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 64:
            reduce6_Min<T, 64, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 32:
            reduce6_Min<T, 32, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case 16:
            reduce6_Min<T, 16, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  8:
            reduce6_Min<T, 8, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  4:
            reduce6_Min<T, 4, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  2:
            reduce6_Min<T, 2, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;

        case  1:
            reduce6_Min<T, 1, false> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, size);
            break;
        }
    }
}

// Wrapper function for kernel launch
template <class T>
void
reduce_Energy(int size, int threads, int blocks,
    int whichKernel, T* M_idata, T* H_idata, T* E_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);


    if (isPow2(size))
    {
        switch (threads)
        {
        case 512:
            reduce_6_ProdSum<T, 512, true> << < dimGrid, dimBlock, smemSize >> > (M_idata,H_idata,E_odata, size);
            break;

        case 256:
            reduce_6_ProdSum<T, 256, true> << < dimGrid, dimBlock, smemSize >> > (M_idata, H_idata, E_odata, size);
            break;

        case 128:
            reduce_6_ProdSum<T, 128, true> << < dimGrid, dimBlock, smemSize >> > (M_idata, H_idata, E_odata, size);
            break;

        case 64:
            reduce_6_ProdSum<T, 64, true> << < dimGrid, dimBlock, smemSize >> > (M_idata, H_idata, E_odata, size);
            break;

        case 32:
            reduce_6_ProdSum<T, 32, true> << < dimGrid, dimBlock, smemSize >> > (M_idata, H_idata, E_odata, size);
            break;

        case 16:
            reduce_6_ProdSum<T, 16, true> << < dimGrid, dimBlock, smemSize >> > (M_idata, H_idata, E_odata, size);
            break;

        case  8:
            reduce_6_ProdSum<T, 8, true> << < dimGrid, dimBlock, smemSize >> > (M_idata, H_idata, E_odata, size);
            break;

        case  4:
            reduce_6_ProdSum<T, 4, true> << < dimGrid, dimBlock, smemSize >> > (M_idata, H_idata, E_odata, size);
            break;

        case  2:
            reduce_6_ProdSum<T, 2, true> << < dimGrid, dimBlock, smemSize >> > (M_idata, H_idata, E_odata, size);
            break;

        case  1:
            reduce_6_ProdSum<T, 1, true> << < dimGrid, dimBlock, smemSize >> > (M_idata, H_idata, E_odata, size);
            break;
        }
    }
    else
    {
        switch (threads)
        {
        case 512:
            reduce_6_ProdSum<T, 512, false> << < dimGrid, dimBlock, smemSize >> > (M_idata, H_idata, E_odata, size);
            break;

        case 256:
            reduce_6_ProdSum<T, 256, false> << < dimGrid, dimBlock, smemSize >> > (M_idata, H_idata, E_odata, size);
            break;

        case 128:
            reduce_6_ProdSum<T, 128, false> << < dimGrid, dimBlock, smemSize >> > (M_idata, H_idata, E_odata, size);
            break;

        case 64:
            reduce_6_ProdSum<T, 64, false> << < dimGrid, dimBlock, smemSize >> > (M_idata, H_idata, E_odata, size);
            break;

        case 32:
            reduce_6_ProdSum<T, 32, false> << < dimGrid, dimBlock, smemSize >> > (M_idata, H_idata, E_odata, size);
            break;

        case 16:
            reduce_6_ProdSum<T, 16, false> << < dimGrid, dimBlock, smemSize >> > (M_idata, H_idata, E_odata, size);
            break;

        case  8:
            reduce_6_ProdSum<T, 8, false> << < dimGrid, dimBlock, smemSize >> > (M_idata, H_idata, E_odata, size);
            break;

        case  4:
            reduce_6_ProdSum<T, 4, false> << < dimGrid, dimBlock, smemSize >> > (M_idata, H_idata, E_odata, size);
            break;

        case  2:
            reduce_6_ProdSum<T, 2, false> << < dimGrid, dimBlock, smemSize >> > (M_idata, H_idata, E_odata, size);
            break;

        case  1:
            reduce_6_ProdSum<T, 1, false> << < dimGrid, dimBlock, smemSize >> > (M_idata, H_idata, E_odata, size);
            break;
        }
    }
}

// Instantiate the reduction functions
template void
reduce<double>(int size, int threads, int blocks,
    int whichKernel, double* d_idata, double* d_odata);

template void
reduce_Max<double>(int size, int threads, int blocks,
    int whichKernel, double* d_idata, double* d_odata);

template void
reduce_Min<double>(int size, int threads, int blocks,
    int whichKernel, double* d_idata, double* d_odata);

// Instantiate the reduction function
template void
reduce_Energy<double>(int size, int threads, int blocks,
    int whichKernel, double* M_idata,double* H_idata,double* E_odata);

#endif // !REDUCTION_KERNELS_CUH

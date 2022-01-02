#include <cuda_runtime.h>
#include "DataTypes.cuh"
#ifndef THREADSANDBLOCKS_FUNCTIONS_CUH
#define THREADSANDBLOCKS_FUNCTIONS_CUH

__host__ int Factorise(int N, int n);
bool IsPowerOfTwo(int x);
__device__ bool IsPowerOfTwo_d(int x);
unsigned int nextPowerOfTwo(unsigned int n);

__host__ void CalculateThreadsAndBlocksReductions_1D();
__host__ void Fields_GetThreadsAndBlocks();
__host__ void Integration_GetThreadsAndBlocks();

#endif // !THREADSANDBLOCKS_FUNCTIONS_CUH

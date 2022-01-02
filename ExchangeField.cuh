#include "DataTypes.cuh"
#include <cuda_runtime.h>
#ifndef EXCHANGEFIELD_CUH
#define EXCHANGEFIELD_CUH

__host__ void ExchangeStencilParameters();
__global__ void ExchangeField_FullGridBoundaries(MAG M, FIELD H);
__global__ void ExchangeField_FullGridBoundaries_PBC(MAG M, FIELD H);
__global__ void ExchangeField_FullGridBoundaries_Mn1(MAG M, FIELD H);
__global__ void ExchangeField_FullGridBoundaries_PBC_Mn1(MAG M, FIELD H);

__device__ int ExWrapX(int x);
__device__ int ExWrapY(int y);
__device__ int ExWrapZ(int z);

__device__ int PBCWrapLeftX(int index);
__device__ int PBCWrapLeftY(int index);
__device__ int PBCWrapRightX(int index);
__device__ int PBCWrapRightY(int index);

#endif // !EXCHANGEFIELD_CUH

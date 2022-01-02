#include <cuda_runtime.h>
#include "DataTypes.cuh"
#include "GlobalDefines.cuh"
#include "Device_Globals_Constants.cuh"
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#ifndef ZHANGLI_STT_CUH
#define ZHANGLI_STT_CUH

__global__ void Compute_ZhangLi_STT(MEMDATA DATA, FIELD H, MAG M);
__global__ void Compute_ZhangLi_STT_Mn1(MEMDATA DATA, FIELD H, MAG M);

__device__ int inline lclampX(int x)
{
	return MAX(x,0);
}
__device__ int inline lclampY(int y)
{
	return MAX(y,0);
}

__device__ int inline lclampZ(int z)
{
	return MAX(z,0);
}

__device__ int inline hclampX(int x)
{
	return MIN(x, NUM-1);
}
__device__ int inline hclampY(int y)
{
	return MIN(y, NUMY-1);
}

__device__ int inline hclampZ(int z)
{
	return MIN(z, NUMZ-1);
}

#endif // !ZHANGLI_STT_CUH

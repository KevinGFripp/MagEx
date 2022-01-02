#include <cuda_runtime.h>
#include "DataTypes.cuh"

#ifndef UNIAXIALANISOTROPY
#define UNIAXIALANISOTROPY

__global__ void UniaxialAnisotropy(MAG M, FIELD H);
__global__ void UniaxialAnisotropy_Mn1(MAG M, FIELD H);

#endif // !UNIAXIALANISOTROPY

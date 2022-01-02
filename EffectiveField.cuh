#include <cuda_runtime.h>
#include "DataTypes.cuh"

#ifndef EFFECTIVEFIELD_CUH
#define EFFECTIVEFIELD_CUH

__global__ void ComputeEffectiveField(MEMDATA DATA, FIELD H,MAG M);
__global__ void ComputeEffectiveField_SinglePrecision_R2C(MEMDATA DATA, FIELD H,MAG M);
__host__ void ComputeFields(MEMDATA DATA, MAG M, FIELD H, PLANS P, int Flag);
__host__ void ComputeFields_RKStageEvaluation(MEMDATA DATA, MAG M, FIELD H, PLANS P, int Flag);

#endif // !EFFECTIVEFIELD_CUH

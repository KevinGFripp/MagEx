#include <cuda_runtime.h>
#include "DataTypes.cuh"

#ifndef HOST_ENGINE_CUH
#define HOST_ENGINE_CUH

__host__ void Simulate(MEMDATA DATA, MAG M, FIELD H, PLANS P,
    MEMDATA DATA_d, MAG M_d, FIELD H_d, OutputFormat Out);

#endif // !HOST_ENGINE_CUH

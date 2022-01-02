#include <cuda_runtime.h>
#include "DataTypes.cuh"

#ifndef MEMORYALLOCATION_CUH
#define MEMORYALLOCATION_CUH
__host__ void EstimateDeviceMemoryRequirements();
__host__ void GlobalInitialise(MEMDATA* DATA_h, MAG* M_h, FIELD* H_h, PLANS* P_h,
    MEMDATA* DATA_temp, MAG* M_temp, FIELD* H_temp,
    MEMDATA* DATA_d, MAG* M_d, FIELD* H_d);
__host__  void MemInitialise_d(MEMDATA* DATA, MAG* M, FIELD* H,
    MEMDATA* DATA_d, MAG* M_d, FIELD* H_d);
__host__  void MemInitialise_h(MEMDATA* DATA, MAG* M, FIELD* H, PLANS* P);
__host__ void AllocateSharedMemorySize();

__host__ void MemoryClear(MEMDATA* DATA, MAG* M, FIELD* H, PLANS* P,
    MEMDATA* DATA_d, MAG* M_d, FIELD* H_d);
#endif // !MEMORYALLOCATION_CUH

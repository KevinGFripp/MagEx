#include <cuda_runtime.h>
#include "DataTypes.cuh"
#ifndef EXTERNALFIELDS_CUH
#define EXTERNALFIELDS_CUH

__device__ Vector ExcitationFunc_CW(double t, double x, double y, double z);
__device__ Vector Excitation_TemporalSinc(double t, double x, double y, double z);

#endif // !EXTERNALFIELDS_CUH

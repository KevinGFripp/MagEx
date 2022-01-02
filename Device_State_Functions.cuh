#include <cuda_runtime.h>
#include "DataTypes.cuh"

#ifndef DEVICE_STATE_FUNCTIONS_CUH
#define DEVICE_STATE_FUNCTIONS_CUH

__host__ void UpdateDeviceTime(double time);
__host__ void SetCurrentTime(double time);

#endif // !DEVICE_STATE_FUNCTIONS_CUH

#include <cuda_runtime.h>
#ifndef DEVICE_MISC_MATH_CUH
#define DEVICE_MISC_MATH_CUH

__device__ double sinc(double x);
__device__ double erf(double x);

#endif // !DEVICE_MISC_MATH_CUH

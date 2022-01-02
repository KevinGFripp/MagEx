#include <cuda_runtime.h>
#include "DataTypes.cuh"
#ifndef DEVICE_VECTORMATH_CUH
#define DEVICE_VECTORMATH_CUH

__device__ inline Vector CrossProduct(double* a, double* b)
{
    Vector result;
    result.X[0] = a[1] * b[2] - a[2] * b[1];
    result.X[1] = a[2] * b[0] - a[0] * b[2];
    result.X[2] = a[0] * b[1] - a[1] * b[0];

    return result;
}
__device__ inline void VectorNormalise(double* X)
{
    double Norm2 = X[0] * X[0] + X[1] * X[1] + X[2] * X[2];
    X[0] /= sqrt(Norm2);
    X[1] /= sqrt(Norm2);
    X[2] /= sqrt(Norm2);
    return;
}
__device__ void VectorMultiplyScalar(double* X, double a);
__device__ Vector VectorAdd(double* X, double* Y);

#endif // !DEVICE_VECTORMATH_CUH

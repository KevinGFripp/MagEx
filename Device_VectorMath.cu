#include "Device_VectorMath.cuh"

__device__ void VectorMultiplyScalar(double* X, double a)
{
    X[0] *= a;
    X[1] *= a;
    X[2] *= a;

    return;
}
__device__ Vector VectorAdd(double* X, double* Y)
{
    Vector Sum;
    Sum.X[0] = X[0] + Y[0];
    Sum.X[1] = X[1] + Y[1];
    Sum.X[2] = X[2] + Y[2];
    return Sum;
}

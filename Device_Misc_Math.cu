#include "Device_Misc_Math.cuh"
#include <math.h>
#include "GlobalDefines.cuh"
__device__ double sinc(double x)
{
    if (fabs(x) <= eps)
    {
        return 1.0;
    }
    else {
        return (sin(x) / (x));
    }
}
__device__ double erf(double x)
{
    return 2. * ((x - x * x * x / 3.) + 
                (x * x * x * x * x / 10.) - 
                (x * x * x * x * x * x * x / 42.)) / sqrt(M_PI);
}
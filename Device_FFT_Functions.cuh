#include <cuda_runtime.h>
#include "DataTypes.cuh"
#ifndef DEVICE_FFT_FUNCTIONS_CUH
#define DEVICE_FFT_FUNCTIONS_CUH

__host__ void FFTPlansInitialise(PLANS P);
__host__ void FFTPlansInitialise_SinglePrecision_R2C(PLANS P);
__host__ void MagnetisationFFT(PLANS P);
__host__ void MagnetisationFFT_SinglePrecision_R2C(PLANS P);
__host__ void DemagFieldInverseFFT(PLANS P);
__host__ void DemagFieldInverseFFT_SinglePrecision_R2C(PLANS P);
__host__ void DemagTensorFFT_Symmetries(MEMDATA DATA_h, MEMDATA DATA_d);
#endif // !DEVICE_FFT_FUNCTIONS_CUH

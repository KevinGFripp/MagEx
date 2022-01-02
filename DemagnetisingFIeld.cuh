#include <cuda_runtime.h>
#include "DataTypes.cuh"

#ifndef DEMAGNETISINGFIELD_CUH
#define DEMAGNETISINGFIELD_CUH

__global__ void MagnetisationFFTCompute3DInitialised(MEMDATA DATA, MAG M);
__global__ void MagnetisationFFTCompute3DInitialised_SinglePrecision_R2C(MEMDATA DATA, MAG M);
__global__ void MagnetisationFFTCompute3DInitialised_Mn1(MEMDATA DATA, MAG M);
__global__ void MagnetisationFFTCompute3DInitialised_Mn1_SinglePrecision_R2C(MEMDATA DATA, MAG M);
__global__ void ComputeDemagField3DConvolution_Symmetries(MEMDATA DATA);
__global__ void ComputeDemagField3DConvolution_Symmetries_SinglePrecision_R2C(MEMDATA DATA);

#endif // !DEMAGNETISINGFIELD_CUH

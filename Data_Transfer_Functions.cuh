#include <cuda_runtime.h>
#include "DataTypes.cuh"
#include "Host_Globals.cuh"
#include <stdlib.h>
#include "Magnetisation_Functions.cuh"
#include "GlobalDefines.cuh"
#include "Pointer_Functions.cuh"
#include <helper_cuda.h>
#ifndef DATA_TRANSFER_FUNCTIONS_CUH
#define DATA_TRANSFER_FUNCTIONS_CUH

//
// Data Transfer
//
__host__ void CopyDevicePointers(MEMDATA DATA, MAG M, FIELD H);
__host__ void CopyMagToDevice(MAG M_h, MAG M_d);
__host__ void CopyFieldToDevice(FIELD F_h, FIELD F_d);
__host__ void CopyMemDataToDevice(MEMDATA DATA_h, MEMDATA DATA_d);
__host__ void CopyEffectiveFieldFromDevice(FIELD F_h, FIELD F_d);
__host__ void CopyMemDataFromDevice(MEMDATA DATA_h, MEMDATA DATA_d);
__host__ void CopyExchangeFieldFromDevice(FIELD F_h, FIELD F_d);
__host__ void CopyDemagFieldFromDevice(FIELD F_h, FIELD F_d);
__host__ void CopyDemagComponentsFromDevice(FIELD H_h, FIELD H_d);
__host__ void CopyMagFromDevice(MAG M_h, MAG M_d);
__host__ void CopyMagComponentsFromDevice(MAG M_h, MAG M_d);

#endif // !DATA_TRANSFER_FUNCTIONS_CUH

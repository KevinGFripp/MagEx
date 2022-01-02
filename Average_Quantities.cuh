#include <cuda_runtime.h>
#include "DataTypes.cuh"

#ifndef AVERAGE_QUANTITIES_CUH
#define AVERAGE_QUANTITIES_CUH

//Averaged Quantities

__host__ double UniAnisotropyEnergy(MAG M, FIELD H, MEMDATA DATA);
__host__ double ExchangeEnergy(MAG M, FIELD H, MEMDATA DATA);
__host__ double DemagEnergy(MAG M, FIELD H, MEMDATA DATA);
__host__ double ZeemanEnergy(MAG M, FIELD H, MEMDATA DATA);

__host__ Vector AverageExchange_Reduction();
__host__ Vector AverageMag_Reduction();
__host__ Vector AverageDemag_Reduction();
__host__ Vector AverageUniAnisotropy_Reduction();
__host__ Vector AverageZeeman_Reduction();

#endif // !AVERAGE_QUANTITIES_CUH

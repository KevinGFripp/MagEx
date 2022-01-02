#include <cuda_runtime.h>
#include "DataTypes.cuh"

#ifndef ODE_INTEGRATOR_ESDIRK547_CUH
#define ODE_INTEGRATOR_ESDIRK547_CUH

__global__ void RungeKuttaStage_2_ESDIRK547(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_3_ESDIRK547(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_4_ESDIRK547(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_4_ESDIRK547_TorqueAndField(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_5_ESDIRK547(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_6_ESDIRK547(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_7_ESDIRK547(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaFinalSolution_ESDIRK547(MAG M, FIELD H, MEMDATA DATA);
__host__ void RungeKuttaESDIRK547_StageEvaluations(MAG M_d, FIELD H_d, MEMDATA DATA, PLANS P, double h,
    int* ResetFlag, int AllocSize);

#endif // !ODE_INTEGRATOR_ESDIRK547_CUH

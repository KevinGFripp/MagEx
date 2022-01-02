#include <cuda_runtime.h>
#include "DataTypes.cuh"

#ifndef ODE_INTEGRATOR_STT_ESDIRK54_CUH
#define ODE_INTEGRATOR_STT_ESDIRK54_CUH

__global__ void RungeKuttaStage_2_STT_ESDIRK54a(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_3_STT_ESDIRK54a(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_4_STT_ESDIRK54a(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_4_STT_ESDIRK54a_TorqueAndField(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_5_STT_ESDIRK54a(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_6_STT_ESDIRK54a(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_7_STT_ESDIRK54a(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaFinalSolution_STT_ESDIRK54a(MAG M, FIELD H, MEMDATA DATA);
__host__ void RungeKuttaESDIRK54a_STT_StageEvaluations(MAG M_d, FIELD H_d, MEMDATA DATA, PLANS P, double h,
    int* ResetFlag, int AllocSize);

#endif // !ODE_INTEGRATOR_STT_ESDIRK54_CUH
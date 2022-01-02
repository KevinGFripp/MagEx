#include <cuda_runtime.h>
#include "DataTypes.cuh"


#ifndef ODE_INTEGRATOR_ESDIRK659_CUH
#define ODE_INTEGRATOR_ESDIRK659_CUH

__global__ void RungeKuttaStage_2_ESDIRK659(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_3_ESDIRK659(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_4_ESDIRK659(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_4_ESDIRK659_TorqueAndField(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_5_ESDIRK659(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_6_ESDIRK659(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_7_ESDIRK659(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_8_ESDIRK659(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_9_ESDIRK659(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaFinalSolution_ESDIRK659(MAG M, FIELD H, MEMDATA DATA);
__host__ void RungeKuttaESDIRK659_StageEvaluations(MAG M_d, FIELD H_d, MEMDATA DATA, PLANS P, double h,
    int* ResetFlag, int AllocSize);

#endif // !ODE_INTEGRATOR_ESDIRK659_CUH

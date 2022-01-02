#include <cuda_runtime.h>
#include "DataTypes.cuh"

#ifndef ODE_INTEGRATOR_ESDIRK548_LSA_CUH
#define ODE_INTEGRATOR_ESDIRK548_LSA_CUH

__global__ void RungeKuttaStage_2_ESDIRK548_L_SA(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_3_ESDIRK548_L_SA(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_4_ESDIRK548_L_SA(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_5_ESDIRK548_L_SA(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_6_ESDIRK548_L_SA(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_7_ESDIRK548_L_SA(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_8_ESDIRK548_L_SA(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaFinalSolution_ESDIRK548_L_SA(MAG M, FIELD H, MEMDATA DATA);
__host__ void RungeKuttaESDIRK548_L_SA_StageEvaluations(MAG M_d, FIELD H_d, MEMDATA DATA, PLANS P, double h,
    int* ResetFlag, int AllocSize);

#endif // !ODE_INTEGRATOR_ESDIRK548_LSA_CUH

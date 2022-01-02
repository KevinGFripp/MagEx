#include <cuda_runtime.h>
#include "DataTypes.cuh"

#ifndef ODE_INTEGRATOR_RK54DP_CUH
#define ODE_INTEGRATOR_RK54DP_CUH

__host__ void RungeKutta54DP_StageEvaluations(MAG M_d, FIELD H_d,
    MEMDATA DATA, PLANS P, double h, int* ResetFlag, int AllocSize);

__global__ void RungeKuttaStage_2_RK54DP(MAG M, FIELD H, MEMDATA DATA, int Flag);
__global__ void RungeKuttaStage_3_RK54DP(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_4_RK54DP(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_5_RK54DP(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_6_RK54DP(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_7_RK54DP(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaFinalSolution_RK54DP(MAG M, FIELD H, MEMDATA DATA);

__device__ Vector rk54dp_stage_1(double* Mn, double* H);
__device__ Vector rk54dp_stage_2(double* Mn1, double* Mn, double* H, double* k1);
__device__ Vector rk54dp_stage_3(double* Mn1, double* Mn, double* H, double* k1, double* k2);
__device__ Vector rk54dp_stage_4(double* Mn1, double* Mn, double* H, double* k1, double* k2,
                                 double* k3);
__device__ Vector rk54dp_stage_5(double* Mn1, double* Mn, double* H, double* k1, double* k2,
                                 double* k3, double* k4);
__device__ Vector rk54dp_stage_6(double* Mn1, double* Mn, double* H, double* k1, double* k2,
                                 double* k3, double* k4, double* k5);
__device__ Vector rk54dp_stage_7(double* Mn1, double* Mn, double* H, double* k1, double* k2,
                                 double* k3, double* k4, double* k5, double* k6);

#endif // !ODE_INTEGRATOR_RK54DP_CUH

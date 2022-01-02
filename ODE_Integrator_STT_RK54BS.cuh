#include <cuda_runtime.h>
#include "DataTypes.cuh"

#ifndef ODE_INTEGRATOR_STT_RK54BS_CUH
#define ODE_INTEGRATOR_STT_RK54BS_CUH

__host__ void   RungeKutta54BS_STT_StageEvaluations(MAG M_d, FIELD H_d,
    MEMDATA DATA, PLANS P, double h, int* ResetFlag, int AllocSize);
__global__ void RungeKuttaStage_2_STT_RK54BS(MAG M, FIELD H, MEMDATA DATA, int Flag);
__global__ void RungeKuttaStage_3_STT_RK54BS(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_4_STT_RK54BS(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_5_STT_RK54BS(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_6_STT_RK54BS(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_7_STT_RK54BS(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaStage_8_STT_RK54BS(MAG M, FIELD H, MEMDATA DATA);
__global__ void RungeKuttaFinalSolution_STT_RK54BS(MAG M, FIELD H, MEMDATA DATA);

__device__ Vector rk54bs_STT_stage_1(double* Mn, double* H, double* Hstt, MaterialHandle index);
__device__ Vector rk54bs_STT_stage_2(double* Mn1, double* Mn, double* H, double* Hstt, MaterialHandle index,
                                     double* k1);
__device__ Vector rk54bs_STT_stage_3(double* Mn1, double* Mn, double* H, double* Hstt, MaterialHandle index,
                                     double* k1, double* k2);
__device__ Vector rk54bs_STT_stage_4(double* Mn1, double* Mn, double* H, double* Hstt, MaterialHandle index,
                                     double* k1, double* k2, double* k3);
__device__ Vector rk54bs_STT_stage_5(double* Mn1, double* Mn, double* H, double* Hstt, MaterialHandle index,
                                     double* k1, double* k2, double* k3, double* k4);
__device__ Vector rk54bs_STT_stage_6(double* Mn1, double* Mn, double* H, double* Hstt, MaterialHandle index,
                                     double* k1, double* k2,double* k3, double* k4, double* k5);
__device__ Vector rk54bs_STT_stage_7(double* Mn1, double* Mn, double* H, double* Hstt, MaterialHandle index,
                                     double* k1, double* k2,double* k3, double* k4, double* k5, double* k6);
__device__ Vector rk54bs_STT_stage_8(double* Mn1, double* Mn, double* H, double* Hstt, MaterialHandle index,
                                     double* k1, double* k2,double* k3, double* k4, double* k5, double* k6,
                                     double* k7);

#endif // !ODE_INTEGRATOR_STT_RK54BS_CUH

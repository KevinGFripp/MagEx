#include <cuda_runtime.h>
#include "DataTypes.cuh"

#ifndef ODE_INTEGRATOR_CUH
#define ODE_INTEGRATOR_CUH

__host__ void ComputeInitialStep(MAG M, FIELD H, MEMDATA DATA, PLANS P, double* step);
__host__ void MakeStep_RK(Vector* Stepdata, double* h, MAG M, FIELD H, MEMDATA DATA, PLANS P,
    int* Flag);
__host__ Vector LandauLifshiftzIntegration_RKStagesFieldEvaluation(MAG M_d, FIELD H_d,
    MEMDATA DATA, PLANS P, double h, int* ResetFlag);
__host__ Vector LandauLifshiftzIntegration_RKStagesFieldEvaluation_STT(MAG M_d, FIELD H_d,
    MEMDATA DATA, PLANS P, double h, int* ResetFlag);
__host__ void ResetIntegrationStepFlag(int Flag);
__global__ void RungeKuttaStage_CopyPreviousStep(MAG M, FIELD H, MEMDATA DATA, int Flag);

__host__ int AdaptiveStepSizeControl(double* ERR, double* h, double MaxTorque,
    double MinTorque, double TOL);
__host__ int AdaptiveStepSizeControl_ESDIRK54(double* ERR, double* h, double MaxTorque,
    double MinTorque, double TOL, int Flag);
__host__ int AdaptiveStepSizeControl_DIRK65(double* ERR, double* h, double MaxTorque,
    double MinTorque, double TOL, int Flag);
__host__ int AdaptiveStepSizeControl_DormandPrince54(double* ERR, double* h, double MaxTorque,
    double MinTorque, double TOL, int Flag);


#endif // !ODE_INTEGRATOR_CUH

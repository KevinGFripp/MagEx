#include "ODE_Integrator.cuh"
#include <device_launch_parameters.h>
#include "Device_Globals_Constants.cuh"
#include "Host_Globals.cuh"
#include "GlobalDefines.cuh"
#include "Array_Indexing_Functions.cuh"
#include <helper_cuda.h>
#include "Reduction_Functions.cuh"
#include "ODE_Integrator_RK54DP.cuh"
#include "ODE_Integrator_RK54BS.cuh"
#include "ODE_Integrator_ESDIRK54.cuh"
#include "ODE_Integrator_ESDIRK659.cuh"
#include "ODE_Integrator_STT_ESDIRK54.cuh"
#include "ODE_Integrator_STT_RK54BS.cuh"

__host__ void ComputeInitialStep(MAG M, FIELD H, MEMDATA DATA, PLANS P, double* step)
{
    /* Vector Heff;
     ComputeFields(DATA, M, H, P,0);
     int MSIZE = (NumberofThreadsIntegrator.x * NumberofThreadsIntegrator.y * NumberofThreadsIntegrator.z);
     int ReductionSize = (NumberofBlocksIntegrator.x * NumberofBlocksIntegrator.y * NumberofBlocksIntegrator.z);

     int blocksize_2ndstage = 0;

     if (ReductionSize > 1024)
     {
         while (ReductionSize > 1024)
         {
             ReductionSize /= 2;
             blocksize_2ndstage += 2;
         }
         blocksize_2ndstage /= 2;
     }

     for (int n = 0;n < DIM; n++)
     {
         if (blocksize_2ndstage == 0)
         {
             MaxReduction << <1, ReductionSize, (ReductionSize * sizeof(double)) >> >
                 (&(DEVICE_PTR_STRUCT.H)->H_eff[find_h(0, 0, 0, n)], (DEVICE_PTR_STRUCT.DATA->StepReduction), ReductionSize);
             checkCudaErrors(cudaDeviceSynchronize());
         }
         else {
             MaxReduction << <blocksize_2ndstage, ReductionSize, (ReductionSize * sizeof(double)) >> >
                 (&(DEVICE_PTR_STRUCT.H)->H_eff[find_h(0, 0, 0, n)], (DEVICE_PTR_STRUCT.DATA->StepReduction), ReductionSize);

             checkCudaErrors(cudaDeviceSynchronize());
             MaxReduction << <1, 64, (64 * sizeof(double)) >> >
                 ((DEVICE_PTR_STRUCT.DATA->StepReduction), (DEVICE_PTR_STRUCT.DATA->StepReduction), blocksize_2ndstage);
         }

         checkCudaErrors(cudaMemcpy(&Heff.X[n], &(DEVICE_PTR_STRUCT.DATA)->StepReduction[0],
             sizeof(double), cudaMemcpyDeviceToHost));
         checkCudaErrors(cudaDeviceSynchronize());
     }

     double MaxH = fabs((Heff.X[0] + Heff.X[1] + Heff.X[2]) / 3);
     *step = sqrt(alpha_h)*0.2*(MaxH) / (Gamma * 1e6);*/

    * step = 0.5 * sqrt(alpha_h) * 1e-3;


}
__host__ void MakeStep_RK(Vector* Stepdata, double* h, MAG M, FIELD H, MEMDATA DATA, PLANS P,
    int* Flag)
{
    //Stepdata : MaxError|MaxTorque|MinTorque
    if (SpinTransferTorque_h == 0)
    {
        *Stepdata = LandauLifshiftzIntegration_RKStagesFieldEvaluation(M, H, DATA, P, *h, Flag);
    }
    else 
    {
        *Stepdata = LandauLifshiftzIntegration_RKStagesFieldEvaluation_STT(M, H, DATA, P, *h, Flag);
    }

    if (FixedTimeStep == true)
    {
        return;
    }

    if (METHOD_h == TRAPEZOIDAL_FE)
    {
        *Flag = AdaptiveStepSizeControl(&((*Stepdata).X[0]), h, Stepdata->X[1], Stepdata->X[2], RelTol);
    }

    if (METHOD_h == ESDIRK54)
    {
        *Flag = AdaptiveStepSizeControl_ESDIRK54(&((*Stepdata).X[0]), h, Stepdata->X[1],
            Stepdata->X[2], RelTol, *Flag);
    }

    if (METHOD_h == ESDIRK65)
    {
        *Flag = AdaptiveStepSizeControl_DIRK65(&((*Stepdata).X[0]), h, Stepdata->X[1],
            Stepdata->X[2], RelTol, *Flag);
    }

    if (METHOD_h == RK54DP || METHOD_h == RK54BS)
    {
        *Flag = AdaptiveStepSizeControl_DormandPrince54(&((*Stepdata).X[0]), h, Stepdata->X[1],
            Stepdata->X[2], RelTol, *Flag);
    }
    return;
}
__host__ Vector LandauLifshiftzIntegration_RKStagesFieldEvaluation(MAG M_d, FIELD H_d, MEMDATA DATA, PLANS P, double h, int* ResetFlag)
{
    Vector stepdata;
    stepdata.X[0] = 0.0; //LTE Error
    stepdata.X[1] = 0.0; //Max torque 
    stepdata.X[2] = 0.0; //Min Torque

    int MSIZE = (NumberofThreadsIntegrator.x * NumberofThreadsIntegrator.y * NumberofThreadsIntegrator.z);
    int ReductionSize = (NumberofBlocksIntegrator.x * NumberofBlocksIntegrator.y * NumberofBlocksIntegrator.z);
    int ALLOCSIZE = 3 * (MSIZE * DIM * sizeof(double)) + MSIZE * sizeof(int);


    int blocksize_2ndstage = 0;

    ResetIntegrationStepFlag(*ResetFlag);

    if (METHOD_h == RK54DP)
    {
        RungeKutta54DP_StageEvaluations(M_d, H_d, DATA, P, h, ResetFlag, ALLOCSIZE);
    }

    if (METHOD_h == RK54BS)
    {
        RungeKutta54BS_StageEvaluations(M_d, H_d, DATA, P, h, ResetFlag, ALLOCSIZE);
    }

    if (METHOD_h == ESDIRK54)
    {
        RungeKuttaESDIRK54a_StageEvaluations(M_d, H_d, DATA, P, h, ResetFlag, ALLOCSIZE);
    }

    if (METHOD_h == ESDIRK65)
    {
        RungeKuttaESDIRK659_StageEvaluations(M_d, H_d, DATA, P, h, ResetFlag, ALLOCSIZE);
    }

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stepdata.X[0] = StepError_Reduction();
    stepdata.X[1] = MaxTorque_Reduction();
    stepdata.X[2] = MinTorque_Reduction();

    cudaDeviceSynchronize();

    return stepdata;
}
__host__ Vector LandauLifshiftzIntegration_RKStagesFieldEvaluation_STT(MAG M_d, FIELD H_d, MEMDATA DATA, PLANS P, double h, int* ResetFlag)
{
    Vector stepdata;
    stepdata.X[0] = 0.0; //LTE Error
    stepdata.X[1] = 0.0; //Max torque 
    stepdata.X[2] = 0.0; //Min Torque

    int MSIZE = (NumberofThreadsIntegrator.x * NumberofThreadsIntegrator.y * NumberofThreadsIntegrator.z);
    int ReductionSize = (NumberofBlocksIntegrator.x * NumberofBlocksIntegrator.y * NumberofBlocksIntegrator.z);
    int ALLOCSIZE = 3 * (MSIZE * DIM * sizeof(double)) + MSIZE * sizeof(int);


    int blocksize_2ndstage = 0;

    ResetIntegrationStepFlag(*ResetFlag);

    if (METHOD_h == ESDIRK54)
    {
        RungeKuttaESDIRK54a_STT_StageEvaluations(M_d, H_d, DATA, P, h, ResetFlag, ALLOCSIZE);
    }

    if (METHOD_h == RK54BS)
    {
        RungeKutta54BS_STT_StageEvaluations(M_d, H_d, DATA, P, h, ResetFlag, ALLOCSIZE);
    }


    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stepdata.X[0] = StepError_Reduction();
    stepdata.X[1] = MaxTorque_Reduction();
    stepdata.X[2] = MinTorque_Reduction();

    cudaDeviceSynchronize();

    return stepdata;
}
__global__ void RungeKuttaStage_CopyPreviousStep(MAG M, FIELD H, MEMDATA DATA, int Flag)
{
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    if ( (i < NUM) && (j < NUMY) && (k < NUMZ))
    {
        if (M->Mat[ind(i, j, k)] == 1)
        {
            if (ResetFlag_d == 0)
            {
                M->M[mind(2, i, j, k, 0)] = M->M[mind(0, i, j, k, 0)]; //Backup last step only if last step was good
                M->M[mind(2, i, j, k, 1)] = M->M[mind(0, i, j, k, 1)];
                M->M[mind(2, i, j, k, 2)] = M->M[mind(0, i, j, k, 2)];
            }
            else {
                M->M[mind(0, i, j, k, 0)] = M->M[mind(2, i, j, k, 0)];
                M->M[mind(0, i, j, k, 1)] = M->M[mind(2, i, j, k, 1)];
                M->M[mind(0, i, j, k, 2)] = M->M[mind(2, i, j, k, 2)];
            }
        }
    }
    return;
}
__host__ void ResetIntegrationStepFlag(int Flag)
{
    checkCudaErrors(cudaMemcpyToSymbol(ResetFlag_d, &Flag, sizeof(int)));
    return;
}

__host__ int AdaptiveStepSizeControl(double* ERR, double* h, double MaxTorque,
    double MinTorque, double TOL)
{
    double ErrorRatio, Err;
    double MAXSTEP = 1.5e-3;

    Err = *ERR;


    if (Err > TOL * 5)
    {
        ErrorRatio = 0.95 * cbrt(TOL * 5 / Err);
        *h *= ErrorRatio;
        return 1;
    }

    else if (Err < 0.95 * TOL)
    {
        ErrorRatio = 0.9 * sqrt(TOL * 5 / Err);
        if (ErrorRatio >= 1.001)
        {
            *h *= 1.001;
            if (*h > MAXSTEP)
            {
                *h = MAXSTEP;
            }
        }
        else
        {
            *h *= ErrorRatio;
            if (*h > MAXSTEP)
            {
                *h = MAXSTEP;
            }
        }
        return 0;
    }
    return 0;
}
__host__ int AdaptiveStepSizeControl_ESDIRK54(double* ERR, double* h, double MaxTorque,
    double MinTorque, double TOL, int Flag)
{
    double ErrorRatio;

    double MAXSTEP = 2e-3;
    double MINSTEP = 1e-6;

    double fractionalTol = TOL;
    if (*ERR > fractionalTol)
    {
        ErrorRatio = 0.84 * pow(fractionalTol / *ERR, 1. / 6.);


        if (ErrorRatio <= 0.5)
        {
            *h *= 0.5;
        }
        else
        {
            *h *= ErrorRatio;
        }

        if (*h < MINSTEP)
        {
            *h = MINSTEP;
        }

        return 1;
    }
    else if (*ERR < fractionalTol && Flag == 0)
    {
        ErrorRatio = 0.84 * pow(fractionalTol / *ERR, 1. / 6.);
        if (ErrorRatio >= 1.25)
        {
            *h *= 1.25;

            if (*h > MAXSTEP)
            {
                *h = MAXSTEP;
            }
        }
        else
        {
            *h *= ErrorRatio;

            if (*h > MAXSTEP)
            {
                *h = MAXSTEP;
            }
        }
        return 0;
    }
    return 0;
}
__host__ int AdaptiveStepSizeControl_DIRK65(double* ERR, double* h, double MaxTorque,
    double MinTorque, double TOL, int Flag)
{
    double ErrorRatio;

    double MAXSTEP = 2e-3;
    double MINSTEP = 1e-6;

    double fractionalTol = TOL;
    if (*ERR > fractionalTol)
    {
        ErrorRatio = 0.80 * pow(fractionalTol / *ERR, 1. / 7.);


        if (ErrorRatio <= 0.5)
        {
            *h *= 0.5;
        }
        else
        {
            *h *= ErrorRatio;
        }

        if (*h < MINSTEP)
        {
            *h = MINSTEP;
        }

        return 1;
    }
    else if (*ERR < fractionalTol && Flag == 0)
    {
        ErrorRatio = 0.80 * pow(fractionalTol / *ERR, 1. / 7.);
        if (ErrorRatio >= 1.25)
        {
            *h *= 1.25;

            if (*h > MAXSTEP)
            {
                *h = MAXSTEP;
            }
        }
        else
        {
            *h *= ErrorRatio;

            if (*h > MAXSTEP)
            {
                *h = MAXSTEP;
            }
        }
        return 0;
    }
    return 0;
}
__host__ int AdaptiveStepSizeControl_DormandPrince54(double* ERR, double* h, double MaxTorque,
    double MinTorque, double TOL, int Flag)
{
    double ErrorRatio;

    double MAXSTEP = 2.0e-3;
    double MINSTEP = 1e-6;

    double fractionalTol = TOL;
    if (*ERR > fractionalTol)
    {
        ErrorRatio = 0.8 * pow(fractionalTol / *ERR, 1. / 6.);


        if (ErrorRatio <= 0.5)
        {
            *h *= 0.5;
        }
        else
        {
            *h *= ErrorRatio;
        }

        if (*h < MINSTEP)
        {
            *h = MINSTEP;
        }

        return 1;
    }
    else if (*ERR < fractionalTol && Flag == 0)
    {
        ErrorRatio = 0.8 * pow(fractionalTol / *ERR, 1. / 5.);
        if (ErrorRatio >= 1.25)
        {
            *h *= 1.25;

            if (*h > MAXSTEP)
            {
                *h = MAXSTEP;
            }
        }
        else
        {
            *h *= ErrorRatio;

            if (*h > MAXSTEP)
            {
                *h = MAXSTEP;
            }
        }
        return 0;
    }
    return 0;
}
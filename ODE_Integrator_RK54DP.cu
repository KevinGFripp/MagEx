#include "ODE_Integrator_RK54DP.cuh"
#include "Host_Globals.cuh"
#include "Device_Globals_Constants.cuh"
#include "LandauLifshitz.cuh"
#include "GlobalDefines.cuh"
#include <device_launch_parameters.h>
#include "Array_Indexing_Functions.cuh"
#include "Device_VectorMath.cuh"
#include "Reduction_Functions.cuh"
#include "ODE_Integrator.cuh"
#include "Device_State_Functions.cuh"
#include <helper_cuda.h>
#include "EffectiveField.cuh"

__host__ void RungeKutta54DP_StageEvaluations(MAG M_d, FIELD H_d, MEMDATA DATA, PLANS P, double h, int* ResetFlag, int AllocSize)
{

    double c2 = 1. / 5., c3 = 3. / 10., c4 = 4. / 5., c5 = 8. / 9., c6 = 1.;

    double tn = t_h;

    RungeKuttaStage_CopyPreviousStep << <NumberofBlocksIntegrator, NumberofThreadsIntegrator>> >
        (M_d, H_d, DATA, *ResetFlag);

    t_h = tn + c2 * h;
    UpdateDeviceTime(t_h);

    if (*ResetFlag == 1)
    {
        ComputeFields(DATA, M_d, H_d, P, 0);
    }


    RungeKuttaStage_2_RK54DP << <NumberofBlocksIntegrator, NumberofThreadsIntegrator>> >
        (M_d, H_d, DATA, *ResetFlag);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c3 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_3_RK54DP << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c4 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_4_RK54DP << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c5 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_5_RK54DP << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c6 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_6_RK54DP << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_7_RK54DP << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);


    RungeKuttaFinalSolution_RK54DP << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    return;
}

__device__ Vector rk54dp_stage_1(double* Mn, double* H)
{
    Vector Data = EvaluateLandauLifshitz_GPU(Mn, H);

    return Data;
}
__device__ Vector rk54dp_stage_2(double* Mn1, double* Mn, double* H, double* k1)
{
    Vector Data;
    double a21 = 1. / 5.;

#pragma unroll DIM
    for (int n = 0; n < DIM; n++)
    {
        Mn1[n] = fma(a21 * h_d, k1[n], Mn[n]);
    }

    Data = EvaluateLandauLifshitz_GPU(Mn1, H);

    return Data;
}
__device__ Vector rk54dp_stage_3(double* Mn1, double* Mn, double* H, double* k1, double* k2)
{
    Vector Data;
    double a31 = 3. / 40., a32 = 9. / 40.;

#pragma unroll DIM
    for (int n = 0; n < DIM; n++)
    {
        Mn1[n] = fma(a31 * h_d, k1[n], Mn[n]);
        Mn1[n] = fma(a32 * h_d, k2[n], Mn1[n]);
    }

    Data = EvaluateLandauLifshitz_GPU(Mn1, H);

    return Data;
}
__device__ Vector rk54dp_stage_4(double* Mn1, double* Mn, double* H, double* k1, double* k2,
    double* k3)
{
    Vector Data;
    double a41 = 44. / 45., a42 = -56. / 15., a43 = 32. / 9.;

#pragma unroll DIM
    for (int n = 0; n < DIM; n++)
    {
        Mn1[n] = fma(a41 * h_d, k1[n], Mn[n]);
        Mn1[n] = fma(a42 * h_d, k2[n], Mn1[n]);
        Mn1[n] = fma(a43 * h_d, k3[n], Mn1[n]);
    }

    Data = EvaluateLandauLifshitz_GPU(Mn1, H);

    return Data;
}
__device__ Vector rk54dp_stage_5(double* Mn1, double* Mn, double* H, double* k1, double* k2,
    double* k3, double* k4)
{
    Vector Data;
    double a51 = 19372. / 6561., a52 = -25360. / 2187., a53 = 64448. / 6561.,
        a54 = -212. / 729.;

#pragma unroll DIM
    for (int n = 0; n < DIM; n++)
    {
        Mn1[n] = fma(a51 * h_d, k1[n], Mn[n]);
        Mn1[n] = fma(a52 * h_d, k2[n], Mn1[n]);
        Mn1[n] = fma(a53 * h_d, k3[n], Mn1[n]);
        Mn1[n] = fma(a54 * h_d, k4[n], Mn1[n]);
    }

    Data = EvaluateLandauLifshitz_GPU(Mn1, H);

    return Data;
}
__device__ Vector rk54dp_stage_6(double* Mn1, double* Mn, double* H, double* k1, double* k2,
    double* k3, double* k4, double* k5)
{
    Vector Data;
    double a61 = 9017. / 3168., a62 = -355. / 33.,
        a63 = 46732. / 5247., a64 = 49. / 176., a65 = -5103. / 18656.;

#pragma unroll DIM
    for (int n = 0; n < DIM; n++)
    {
        Mn1[n] = fma(a61 * h_d, k1[n], Mn[n]);
        Mn1[n] = fma(a62 * h_d, k2[n], Mn1[n]);
        Mn1[n] = fma(a63 * h_d, k3[n], Mn1[n]);
        Mn1[n] = fma(a64 * h_d, k4[n], Mn1[n]);
        Mn1[n] = fma(a65 * h_d, k5[n], Mn1[n]);
    }

    Data = EvaluateLandauLifshitz_GPU(Mn1, H);

    return Data;
}
__device__ Vector rk54dp_stage_7(double* Mn1, double* Mn, double* H, double* k1, double* k2,
    double* k3, double* k4, double* k5, double* k6)
{
    Vector Data;
    double a71 = 35. / 384., a72 = 0., a73 = 500. / 1113.,
        a74 = 125. / 192., a75 = -2187. / 6784., a76 = 11. / 84.;

#pragma unroll DIM
    for (int n = 0; n < DIM; n++)
    {
        Mn1[n] = fma(a71 * h_d, k1[n], Mn[n]);
        Mn1[n] = fma(a72 * h_d, k2[n], Mn1[n]);
        Mn1[n] = fma(a73 * h_d, k3[n], Mn1[n]);
        Mn1[n] = fma(a74 * h_d, k4[n], Mn1[n]);
        Mn1[n] = fma(a75 * h_d, k5[n], Mn1[n]);
        Mn1[n] = fma(a76 * h_d, k6[n], Mn1[n]);
    }

    Data = EvaluateLandauLifshitz_GPU(Mn1, H);

    return Data;
}

__global__ void RungeKuttaStage_2_RK54DP(MAG M, FIELD H, MEMDATA DATA, int Flag)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;


    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {

            double Mn1[3], Mn[3], Heff[3];
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Mn1[n] = Mn[n];
            }
            Vector stage = rk54dp_stage_1(Mn, Heff);
            Vector k2 = rk54dp_stage_2(Mn1, Mn, Heff, &(stage.X[0]));

            VectorNormalise(Mn1);
            //Update Mn1
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                M->M[mind(1, i, j, k, n)] = Mn1[n];
                M->M[mind(3, i, j, k, n)] = stage.X[n];

            }
        }
    }

    return;

}
__global__ void RungeKuttaStage_3_RK54DP(MAG M, FIELD H, MEMDATA DATA)
{
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

   
    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
            double Mn1[3], Mn[3], Heff[3], k1[3];

#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Mn1[n] = M->M[mind(1, i, j, k, n)];
                k1[n] = M->M[mind(3, i, j, k, n)];
            }
            Vector k2 = EvaluateLandauLifshitz_GPU(Mn1, Heff);
            Vector stage = rk54dp_stage_3(Mn1, Mn, Heff, k1, &(k2.X[0]));

            VectorNormalise(Mn1);
            //Update Mn1
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                M->M[mind(1, i, j, k, n)] = Mn1[n];
                M->M[mind(4, i, j, k, n)] = k2.X[n];
            }

        }

    }

    return;
}
__global__ void RungeKuttaStage_4_RK54DP(MAG M, FIELD H, MEMDATA DATA)
{
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

  
    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
            double Mn1[3], Mn[3], Heff[3], k1[3], k2[3];
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Mn1[n] = M->M[mind(1, i, j, k, n)];
                k1[n] = M->M[mind(3, i, j, k, n)];
                k2[n] = M->M[mind(4, i, j, k, n)];
            }
            Vector k3 = EvaluateLandauLifshitz_GPU(Mn1, Heff);
            Vector stage = rk54dp_stage_4(Mn1, Mn, Heff, k1, k2, &(k3.X[0]));

            VectorNormalise(Mn1);
            //Update Mn1
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                M->M[mind(1, i, j, k, n)] = Mn1[n];
                M->M[mind(5, i, j, k, n)] = k3.X[n];
            }
        }

    }

    return;
}
__global__ void RungeKuttaStage_5_RK54DP(MAG M, FIELD H, MEMDATA DATA)
{
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

  
    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
            double Mn1[3], Mn[3], Heff[3], k1[3], k2[3], k3[3];
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Mn1[n] = M->M[mind(1, i, j, k, n)];
                k1[n] = M->M[mind(3, i, j, k, n)];
                k2[n] = M->M[mind(4, i, j, k, n)];
                k3[n] = M->M[mind(5, i, j, k, n)];
            }
            Vector k4 = EvaluateLandauLifshitz_GPU(Mn1, Heff);
            Vector stage = rk54dp_stage_5(Mn1, Mn, Heff, k1, k2, k3, &(k4.X[0]));

            VectorNormalise(Mn1);
            //Update Mn1
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                M->M[mind(1, i, j, k, n)] = Mn1[n];
                M->M[mind(6, i, j, k, n)] = k4.X[n];
            }
        }

    }

    return;
}
__global__ void RungeKuttaStage_6_RK54DP(MAG M, FIELD H, MEMDATA DATA)
{
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

  
    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
            double Mn1[3], Mn[3], Heff[3], k1[3], k2[3], k3[3], k4[3];
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Mn1[n] = M->M[mind(1, i, j, k, n)];
                k1[n] = M->M[mind(3, i, j, k, n)];
                k2[n] = M->M[mind(4, i, j, k, n)];
                k3[n] = M->M[mind(5, i, j, k, n)];
                k4[n] = M->M[mind(6, i, j, k, n)];
            }
            Vector k5 = EvaluateLandauLifshitz_GPU(Mn1, Heff);
            Vector stage = rk54dp_stage_6(Mn1, Mn, Heff, k1, k2, k3, k4, &(k5.X[0]));

            VectorNormalise(Mn1);
            //Update Mn1
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                M->M[mind(1, i, j, k, n)] = Mn1[n];
                M->M[mind(7, i, j, k, n)] = k5.X[n];
            }
        }

    }
    return;
}
__global__ void RungeKuttaStage_7_RK54DP(MAG M, FIELD H, MEMDATA DATA)
{
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

  
    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
            double Mn1[3], Mn[3], Heff[3], k1[3], k2[3], k3[3], k4[3], k5[3];
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Mn1[n] = M->M[mind(1, i, j, k, n)];
                k1[n] = M->M[mind(3, i, j, k, n)];
                k2[n] = M->M[mind(4, i, j, k, n)];
                k3[n] = M->M[mind(5, i, j, k, n)];
                k4[n] = M->M[mind(6, i, j, k, n)];
                k5[n] = M->M[mind(7, i, j, k, n)];
            }
            Vector k6 = EvaluateLandauLifshitz_GPU(Mn1, Heff);
            Vector stage = rk54dp_stage_7(Mn1, Mn, Heff, k1, k2, k3, k4, k5, &(k6.X[0]));

            VectorNormalise(Mn1);
            //Update Mn1
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                M->M[mind(1, i, j, k, n)] = Mn1[n];
                M->M[mind(8, i, j, k, n)] = k6.X[n];
            }
        }
    }
    return;
}
__global__ void RungeKuttaFinalSolution_RK54DP(MAG M, FIELD H, MEMDATA DATA)
{
    
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

  
    double Mn1[3], Mnk[3],Heff[3];
    Vector Data;
    Data.X[0] = 0.0;
    Data.X[1] = 0.0;
    Data.X[2] = 0.0;

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
            //RK stage 7

            double b1 = 35. / 384.,
                b2 = 0., b3 = 500. / 1113.,
                b4 = 125. / 192., b5 = -2187. / 6784., b6 = 11. / 84., b7 = 0.;

            double b1p = 5179. / 57600., b2p = 0.,
                b3p = 7571. / 16695., b4p = 393. / 640., b5p = -92097. / 339200.,
                b6p = 187. / 2100., b7p = 1. / 40.;

         //   double Mn1[3], Heff[3];
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Mn1[n] = M->M[mind(1, i, j, k, n)];
            }

            Vector dmdt = EvaluateLandauLifshitz_GPU(Mn1, Heff);


            //Error Estimate
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mnk[n] = fma((b1 - b1p) * h_d, M->M[mind(3, i, j, k, n)], 0.0);
                Mnk[n] = fma((b3 - b3p) * h_d, M->M[mind(5, i, j, k, n)], Mnk[n]);
                Mnk[n] = fma((b4 - b4p) * h_d, M->M[mind(6, i, j, k, n)], Mnk[n]);
                Mnk[n] = fma((b5 - b5p) * h_d, M->M[mind(7, i, j, k, n)], Mnk[n]);
                Mnk[n] = fma((b6 - b6p) * h_d, M->M[mind(8, i, j, k, n)], Mnk[n]);
                Mnk[n] = fma((b7 - b7p) * h_d, dmdt.X[n], Mnk[n]);
            }

            Data.X[0] = sqrt(Mnk[0] * Mnk[0] + Mnk[1] * Mnk[1]
                + Mnk[2] * Mnk[2]);


            Mnk[0] = fmax(dmdt.X[0], dmdt.X[1]);
            Mnk[0] = fmax(Mnk[0], dmdt.X[2]);

            Mnk[1] = fmin(dmdt.X[0], dmdt.X[1]);
            Mnk[1] = fmin(Mnk[1], dmdt.X[2]);

            Data.X[1] = Mnk[0]; //Max Torque
            Data.X[2] = Mnk[1]; //Min Torque
        }

        if (M->Mat[ind(i, j, k)] != 0) //Only write to global memory if integration took place
        {
            M->M[mind(0, i, j, k, 0)] = M->M[mind(1, i, j, k, 0)];
            M->M[mind(0, i, j, k, 1)] = M->M[mind(1, i, j, k, 1)];
            M->M[mind(0, i, j, k, 2)] = M->M[mind(1, i, j, k, 2)];

        }
        else
        {
        }
        //Store LTE in index 8,xcomp, Max torque in index 8,ycomp 
        //Min torque in index 8,zcomp
        M->M[mind(8, i, j, k, 0)] = Data.X[0];
        M->M[mind(8, i, j, k, 1)] = Data.X[1];
        M->M[mind(8, i, j, k, 2)] = Data.X[2];
    }
    return;
}
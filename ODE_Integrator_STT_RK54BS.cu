#include "ODE_Integrator_STT_RK54BS.cuh"

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

__device__ Vector rk54bs_STT_stage_1(double* Mn, double* H,double* Hstt, MaterialHandle index)
{
    Vector Data = EvaluateLandauLifshitz_STT_GPU(Mn,H,Hstt,index);
    return Data;
}
__device__ Vector rk54bs_STT_stage_2(double* Mn1, double* Mn, double* H,double* Hstt,MaterialHandle index,
                                     double* k1)
{
    Vector Data;
    double a21 = 1. / 6.;

#pragma unroll DIM
    for (int n = 0; n < DIM; n++)
    {
        Mn1[n] = fma(a21 * h_d, k1[n], Mn[n]);
    }

    Data = EvaluateLandauLifshitz_STT_GPU(Mn1, H, Hstt, index);

    return Data;
}
__device__ Vector rk54bs_STT_stage_3(double* Mn1, double* Mn, double* H,double* Hstt,
                                     MaterialHandle index, double* k1, double* k2)
{
    Vector Data;
    double a31 = 2. / 27., a32 = 4. / 27.;

#pragma unroll DIM
    for (int n = 0; n < DIM; n++)
    {
        Mn1[n] = fma(a31 * h_d, k1[n], Mn[n]);
        Mn1[n] = fma(a32 * h_d, k2[n], Mn1[n]);
    }

    Data = EvaluateLandauLifshitz_STT_GPU(Mn1, H, Hstt, index);

    return Data;
}
__device__ Vector rk54bs_STT_stage_4(double* Mn1, double* Mn, double* H,double* Hstt,
                                     MaterialHandle index, double* k1, double* k2,
    double* k3)
{
    Vector Data;
    double a41 = 183. / 1372., a42 = -162. / 343., a43 = 1053. / 1372.;

#pragma unroll DIM
    for (int n = 0; n < DIM; n++)
    {
        Mn1[n] = fma(a41 * h_d, k1[n], Mn[n]);
        Mn1[n] = fma(a42 * h_d, k2[n], Mn1[n]);
        Mn1[n] = fma(a43 * h_d, k3[n], Mn1[n]);
    }

    Data = EvaluateLandauLifshitz_STT_GPU(Mn1, H, Hstt, index);

    return Data;
}
__device__ Vector rk54bs_STT_stage_5(double* Mn1, double* Mn, double* H,double* Hstt,MaterialHandle index,
                                     double* k1, double* k2,double* k3, double* k4)
{
    Vector Data;
    double a51 = 68. / 297., a52 = -4. / 11., a53 = 42. / 143.,
        a54 = 1960. / 3861.;

#pragma unroll DIM
    for (int n = 0; n < DIM; n++)
    {
        Mn1[n] = fma(a51 * h_d, k1[n], Mn[n]);
        Mn1[n] = fma(a52 * h_d, k2[n], Mn1[n]);
        Mn1[n] = fma(a53 * h_d, k3[n], Mn1[n]);
        Mn1[n] = fma(a54 * h_d, k4[n], Mn1[n]);
    }

    Data = EvaluateLandauLifshitz_STT_GPU(Mn1, H, Hstt, index);

    return Data;
}
__device__ Vector rk54bs_STT_stage_6(double* Mn1, double* Mn, double* H,double* Hstt, MaterialHandle index,
    double* k1, double* k2,double* k3, double* k4, double* k5)
{
    Vector Data;
    double a61 = 597. / 22528., a62 = 81. / 352.,
        a63 = 63099. / 585728., a64 = 58653. / 366080., a65 = 4617. / 20480.;

#pragma unroll DIM
    for (int n = 0; n < DIM; n++)
    {
        Mn1[n] = fma(a61 * h_d, k1[n], Mn[n]);
        Mn1[n] = fma(a62 * h_d, k2[n], Mn1[n]);
        Mn1[n] = fma(a63 * h_d, k3[n], Mn1[n]);
        Mn1[n] = fma(a64 * h_d, k4[n], Mn1[n]);
        Mn1[n] = fma(a65 * h_d, k5[n], Mn1[n]);
    }

    Data = EvaluateLandauLifshitz_STT_GPU(Mn1, H, Hstt, index);

    return Data;
}
__device__ Vector rk54bs_STT_stage_7(double* Mn1, double* Mn, double* H,double* Hstt,MaterialHandle index,
    double* k1, double* k2, double* k3, double* k4, double* k5, double* k6)
{
    Vector Data;
    double a71 = 174197. / 959244., a72 = -30942. / 79937., a73 = 8152137. / 19744439.,
        a74 = 666106. / 1039181., a75 = -29421. / 29068., a76 = 482048. / 414219.;

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

    Data = EvaluateLandauLifshitz_STT_GPU(Mn1, H, Hstt, index);

    return Data;
}
__device__ Vector rk54bs_STT_stage_8(double* Mn1, double* Mn, double* H,double* Hstt, MaterialHandle index,
    double* k1, double* k2,double* k3, double* k4, double* k5, double* k6, double* k7)
{
    Vector Data;
    double a81 = 587. / 8064., a82 = 0., a83 = 4440339. / 15491840.,
        a84 = 24353. / 124800., a85 = 387. / 44800., a86 = 2152. / 5985., a87 = 7267. / 94080.;

#pragma unroll DIM
    for (int n = 0; n < DIM; n++)
    {
        Mn1[n] = fma(a81 * h_d, k1[n], Mn[n]);
        Mn1[n] = fma(a82 * h_d, k2[n], Mn1[n]);
        Mn1[n] = fma(a83 * h_d, k3[n], Mn1[n]);
        Mn1[n] = fma(a84 * h_d, k4[n], Mn1[n]);
        Mn1[n] = fma(a85 * h_d, k5[n], Mn1[n]);
        Mn1[n] = fma(a86 * h_d, k6[n], Mn1[n]);
        Mn1[n] = fma(a87 * h_d, k7[n], Mn1[n]);
    }

    Data = EvaluateLandauLifshitz_STT_GPU(Mn1, H, Hstt, index);

    return Data;
}
__global__ void RungeKuttaStage_2_STT_RK54BS(MAG M, FIELD H, MEMDATA DATA, int Flag)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {

            double Mn1[3], Mn[3], Heff[3],Hstt[3];
            MaterialHandle MatIndex = M->Mat[ind(i, j, k)];
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Hstt[n] = H->H_STT[find(i, j, k, n)];
                Mn1[n] = Mn[n];
            }
            Vector stage = rk54bs_STT_stage_1(Mn, Heff,Hstt,MatIndex);
            Vector k2 = rk54bs_STT_stage_2(Mn1, Mn, Heff, Hstt, MatIndex,&(stage.X[0]));

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
__global__ void RungeKuttaStage_3_STT_RK54BS(MAG M, FIELD H, MEMDATA DATA)
{
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
            double Mn1[3], Mn[3], Heff[3], k1[3], Hstt[3];
            MaterialHandle MatIndex = M->Mat[ind(i, j, k)];

#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Hstt[n] = H->H_STT[find(i, j, k, n)];
                Mn1[n] = M->M[mind(1, i, j, k, n)];
                k1[n] = M->M[mind(3, i, j, k, n)];
            }
            Vector k2 = EvaluateLandauLifshitz_STT_GPU(Mn1, Heff, Hstt, MatIndex);
            Vector stage = rk54bs_STT_stage_3(Mn1, Mn, Heff, Hstt, MatIndex, k1, &(k2.X[0]));

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
__global__ void RungeKuttaStage_4_STT_RK54BS(MAG M, FIELD H, MEMDATA DATA)
{
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
            double Mn1[3], Mn[3], Heff[3], k1[3], k2[3], Hstt[3];
            MaterialHandle MatIndex = M->Mat[ind(i, j, k)];
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Hstt[n] = H->H_STT[find(i, j, k, n)];
                Mn1[n] = M->M[mind(1, i, j, k, n)];
                k1[n] = M->M[mind(3, i, j, k, n)];
                k2[n] = M->M[mind(4, i, j, k, n)];
            }
            Vector k3 = EvaluateLandauLifshitz_STT_GPU(Mn1, Heff, Hstt, MatIndex);
            Vector stage = rk54bs_STT_stage_4(Mn1, Mn, Heff, Hstt, MatIndex, k1, k2, &(k3.X[0]));

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
__global__ void RungeKuttaStage_5_STT_RK54BS(MAG M, FIELD H, MEMDATA DATA)
{
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;


    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
            double Mn1[3], Mn[3], Heff[3], k1[3], k2[3], k3[3], Hstt[3];
            MaterialHandle MatIndex = M->Mat[ind(i, j, k)];
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Hstt[n] = H->H_STT[find(i, j, k, n)];
                Mn1[n] = M->M[mind(1, i, j, k, n)];
                k1[n] = M->M[mind(3, i, j, k, n)];
                k2[n] = M->M[mind(4, i, j, k, n)];
                k3[n] = M->M[mind(5, i, j, k, n)];
            }
            Vector k4 = EvaluateLandauLifshitz_STT_GPU(Mn1, Heff, Hstt, MatIndex);
            Vector stage = rk54bs_STT_stage_5(Mn1, Mn, Heff, Hstt, MatIndex, k1, k2, k3, &(k4.X[0]));

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
__global__ void RungeKuttaStage_6_STT_RK54BS(MAG M, FIELD H, MEMDATA DATA)
{
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
            double Mn1[3], Mn[3], Heff[3], k1[3], k2[3], k3[3], k4[3], Hstt[3];
            MaterialHandle MatIndex = M->Mat[ind(i, j, k)];
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Hstt[n] = H->H_STT[find(i, j, k, n)];
                Mn1[n] = M->M[mind(1, i, j, k, n)];
                k1[n] = M->M[mind(3, i, j, k, n)];
                k2[n] = M->M[mind(4, i, j, k, n)];
                k3[n] = M->M[mind(5, i, j, k, n)];
                k4[n] = M->M[mind(6, i, j, k, n)];
            }
            Vector k5 = EvaluateLandauLifshitz_STT_GPU(Mn1, Heff, Hstt, MatIndex);
            Vector stage = rk54bs_STT_stage_6(Mn1, Mn, Heff, Hstt, MatIndex, k1, k2, k3, k4, &(k5.X[0]));

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
__global__ void RungeKuttaStage_7_STT_RK54BS(MAG M, FIELD H, MEMDATA DATA)
{
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
            double Mn1[3], Mn[3], Heff[3], k1[3], k2[3], k3[3], k4[3], k5[3], Hstt[3];
            MaterialHandle MatIndex = M->Mat[ind(i, j, k)];
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Hstt[n] = H->H_STT[find(i, j, k, n)];
                Mn1[n] = M->M[mind(1, i, j, k, n)];
                k1[n] = M->M[mind(3, i, j, k, n)];
                k2[n] = M->M[mind(4, i, j, k, n)];
                k3[n] = M->M[mind(5, i, j, k, n)];
                k4[n] = M->M[mind(6, i, j, k, n)];
                k5[n] = M->M[mind(7, i, j, k, n)];
            }
            Vector k6 = EvaluateLandauLifshitz_STT_GPU(Mn1, Heff, Hstt, MatIndex);
            Vector stage = rk54bs_STT_stage_7(Mn1, Mn, Heff, Hstt, MatIndex, k1, k2, k3, k4, k5, &(k6.X[0]));

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
__global__ void RungeKuttaStage_8_STT_RK54BS(MAG M, FIELD H, MEMDATA DATA)
{
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
            double Mn1[3], Mn[3], Heff[3], k1[3], k2[3], k3[3], k4[3], k5[3], k6[3], Hstt[3];
            MaterialHandle MatIndex = M->Mat[ind(i, j, k)];
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Hstt[n] = H->H_STT[find(i, j, k, n)];
                Mn1[n] = M->M[mind(1, i, j, k, n)];
                k1[n] = M->M[mind(3, i, j, k, n)];
                k2[n] = M->M[mind(4, i, j, k, n)];
                k3[n] = M->M[mind(5, i, j, k, n)];
                k4[n] = M->M[mind(6, i, j, k, n)];
                k5[n] = M->M[mind(7, i, j, k, n)];
                k6[n] = M->M[mind(8, i, j, k, n)];
            }
            Vector k7 = EvaluateLandauLifshitz_STT_GPU(Mn1, Heff, Hstt, MatIndex);
            Vector stage = rk54bs_STT_stage_8(Mn1, Mn, Heff, Hstt, MatIndex,
                                              k1, k2, k3, k4, k5, k6, &(k7.X[0]));

            VectorNormalise(Mn1);
            //Update Mn1
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                M->M[mind(1, i, j, k, n)] = Mn1[n];
                M->M[mind(9, i, j, k, n)] = k7.X[n];
            }
        }
    }
    return;
}
__global__ void RungeKuttaFinalSolution_STT_RK54BS(MAG M, FIELD H, MEMDATA DATA)
{
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Mn1[3], Mnk[3], Heff[3];
    Vector Data;
    Data.X[0] = 0.0;
    Data.X[1] = 0.0;
    Data.X[2] = 0.0;

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
            double Hstt[3];
            MaterialHandle MatIndex = M->Mat[ind(i, j, k)];

            //RK stage 7
            double b1 = 587. / 8064., b2 = 0., b3 = 4440339. / 15491840.,
                b4 = 24353. / 124800., b5 = 387. / 44800., b6 = 2152. / 5985., b7 = 7267. / 94080.;

            double b1p = 6059. / 80640., b2p = 0.,
                b3p = 8559189. / 30983680., b4p = 26411. / 124800., b5p = -927. / 89600.,
                b6p = 443. / 1197., b7p = 7267. / 94080.;

#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Hstt[n] = H->H_STT[find(i, j, k, n)];
                Mn1[n] = M->M[mind(1, i, j, k, n)];
            }

            Vector dmdt = EvaluateLandauLifshitz_STT_GPU(Mn1, Heff, Hstt, MatIndex);


            //Error Estimate
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mnk[n] = fma((b1 - b1p) * h_d, M->M[mind(3, i, j, k, n)], 0.0);
                Mnk[n] = fma((b3 - b3p) * h_d, M->M[mind(5, i, j, k, n)], Mnk[n]);
                Mnk[n] = fma((b4 - b4p) * h_d, M->M[mind(6, i, j, k, n)], Mnk[n]);
                Mnk[n] = fma((b5 - b5p) * h_d, M->M[mind(7, i, j, k, n)], Mnk[n]);
                Mnk[n] = fma((b6 - b6p) * h_d, M->M[mind(8, i, j, k, n)], Mnk[n]);
                Mnk[n] = fma((b7 - b7p) * h_d, M->M[mind(9, i, j, k, n)], Mnk[n]);
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
__host__ void   RungeKutta54BS_STT_StageEvaluations(MAG M_d, FIELD H_d, MEMDATA DATA, PLANS P, double h, int* ResetFlag, int AllocSize)
{

    double c2 = 1. / 6., c3 = 2. / 9., c4 = 3. / 7., c5 = 2. / 3., c6 = 3. / 4., c7 = 1.0;

    double tn = t_h;

    RungeKuttaStage_CopyPreviousStep << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA, *ResetFlag);

    t_h = tn + c2 * h;
    UpdateDeviceTime(t_h);

    if (*ResetFlag == 1)
    {
        ComputeFields(DATA, M_d, H_d, P, 0);
    }


    RungeKuttaStage_2_STT_RK54BS << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA, *ResetFlag);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c3 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_3_STT_RK54BS << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c4 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_4_STT_RK54BS << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c5 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_5_STT_RK54BS << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c6 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_6_STT_RK54BS << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c7 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_7_STT_RK54BS << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_8_STT_RK54BS << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);


    RungeKuttaFinalSolution_STT_RK54BS << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    return;
}
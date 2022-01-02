#include "ODE_Integrator_STT_ESDIRK54.cuh"

#include "Host_Globals.cuh"
#include "Device_Globals_Constants.cuh"
#include "LandauLifshitz.cuh"
#include "GlobalDefines.cuh"
#include <device_launch_parameters.h>
#include "Array_Indexing_Functions.cuh"
#include "Device_VectorMath.cuh"
#include "Reduction_Functions.cuh"
#include "ODE_Integrator.cuh"
#include "ODE_LinAlg.cuh"
#include "Device_State_Functions.cuh"
#include <helper_cuda.h>
#include "EffectiveField.cuh"

__global__ void RungeKuttaStage_2_STT_ESDIRK54a(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Da = 0.260000000;
    double a21 = Da;

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {

            double Mk[3],Mn[3], Heff[3],Hstt[3];

            int Pivot_s[3], Flag;
            Jacobian J_s;
            Vector LL, dzi, F_0;
            double z2[3];

            MaterialHandle MatIndex = M->Mat[ind(i, j, k)];


#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Hstt[n] = H->H_STT[find(i, j, k, n)];
                Mk[n] = Mn[n];

            }

            z2[0] = Mn[0],
            z2[1] = Mn[1],
            z2[2] = Mn[2];
            dzi.X[0] = 0.0;
            dzi.X[1] = 0.0;
            dzi.X[2] = 0.0;

            //Compute LU factorisation
            LLJacobian_ESDIRK_GPU(&J_s, Mk, Heff, Da);
            LLJacobian_STT_GPU(&J_s, Mk, Hstt, Da);

            Flag = Crout_LU_Decomposition_with_Pivoting(&J_s.J[0][0], &(Pivot_s[0]), DIM);

            Vector LLn = EvaluateLandauLifshitz_STT_GPU(Mk, Heff,Hstt,MatIndex);

            //k2 system  
            for (int n = 0; n < NEWTONITERATIONS; n++)
            {
                F_0.X[0] = Mn[0];
                F_0.X[1] = Mn[1];
                F_0.X[2] = Mn[2];

                F_0.X[0] = fma(h_d, a21 * LLn.X[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a21 * LLn.X[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a21 * LLn.X[2], F_0.X[2]);
                
                LL = EvaluateLandauLifshitz_STT_GPU(z2, Heff,Hstt,MatIndex);

                F_0.X[0] = fma(h_d, Da * LL.X[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, Da * LL.X[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, Da * LL.X[2], F_0.X[2]);

                if (n == 0) //initial guess
                {
#pragma unroll DIM
                    for (int i = 0; i < DIM; i++)
                    {
                        z2[i] += (2 * Da) * h_d * F_0.X[i];
                    }
                }

                F_0.X[0] -= z2[0];
                F_0.X[1] -= z2[1];
                F_0.X[2] -= z2[2];

                Flag = Crout_LU_with_Pivoting_Solve(&J_s.J[0][0], &F_0.X[0], Pivot_s, &dzi.X[0], DIM);

                z2[0] += dzi.X[0], z2[1] += dzi.X[1], z2[2] += dzi.X[2];

                if ((fabs(dzi.X[0]) + fabs(dzi.X[1]) + fabs(dzi.X[2])) <= AbsTol)
                {
                    break;
                }
            }
            VectorNormalise(z2);
            //Update Mn1
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                M->M[mind(1, i, j, k, n)] = z2[n];
                M->M[mind(3, i, j, k, n)] = LLn.X[n];
            }
        }
    }
    return;

}
__global__ void RungeKuttaStage_3_STT_ESDIRK54a(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Da = 0.260000000;
    double a31 = 0.13000000, a32 = 0.84033320996790809;

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
            MaterialHandle MatIndex = M->Mat[ind(i, j, k)];

            double Mk[3], Mn[3], Heff[3],Hstt[3];

            int Pivot_s[3], Flag;
            Jacobian J_s;
            Vector LL, dzi, F_0;
            double z3[3], k1[3], k2[3];



#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Mk[n] = M->M[mind(1, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Hstt[n] = H->H_STT[find(i, j, k, n)];
                k1[n] = M->M[mind(3, i, j, k, n)];

            }

            //Compute LU factorisation
            LLJacobian_ESDIRK_GPU(&J_s, Mk, Heff, Da);
            LLJacobian_STT_GPU(&J_s, Mk, Hstt, Da);
            Flag = Crout_LU_Decomposition_with_Pivoting(&J_s.J[0][0], &(Pivot_s[0]), DIM);

            LL = EvaluateLandauLifshitz_STT_GPU(Mk, Heff,Hstt,MatIndex);
            k2[0] = LL.X[0];
            k2[1] = LL.X[1];
            k2[2] = LL.X[2];

            double Heff_stage3[3];
            //store stage 2 field
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                H->H_stage[find(i, j, k, n)] = Heff[n];
            }

            z3[0] = Mk[0],
            z3[1] = Mk[1],
            z3[2] = Mk[2];
            dzi.X[0] = 0.0;
            dzi.X[1] = 0.0;
            dzi.X[2] = 0.0;


            //k3 system  
            for (int n = 0; n < NEWTONITERATIONS; n++)
            {
                F_0.X[0] = Mn[0];
                F_0.X[1] = Mn[1];
                F_0.X[2] = Mn[2];

                F_0.X[0] = fma(h_d, a31 * k1[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a31 * k1[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a31 * k1[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a32 * k2[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a32 * k2[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a32 * k2[2], F_0.X[2]);

                LL = EvaluateLandauLifshitz_STT_GPU(z3, Heff,Hstt,MatIndex);

                F_0.X[0] = fma(h_d, Da * LL.X[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, Da * LL.X[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, Da * LL.X[2], F_0.X[2]);

                if (n == 0) //initial guess
                {
#pragma unroll DIM
                    for (int i = 0; i < DIM; i++)
                    {
                        z3[i] = Mn[i] + (1.230333209967908) * h_d * F_0.X[i];
                    }
                }

                F_0.X[0] -= z3[0];
                F_0.X[1] -= z3[1];
                F_0.X[2] -= z3[2];

                Flag = Crout_LU_with_Pivoting_Solve(&J_s.J[0][0], &F_0.X[0], Pivot_s, &dzi.X[0], DIM);

                z3[0] += dzi.X[0], z3[1] += dzi.X[1], z3[2] += dzi.X[2];

                if ((fabs(dzi.X[0]) + fabs(dzi.X[1]) + fabs(dzi.X[2])) <= AbsTol)
                {
                    break;
                }
            }
            VectorNormalise(z3);
            //Update Mn1
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                M->M[mind(1, i, j, k, n)] = z3[n];
                M->M[mind(4, i, j, k, n)] = k2[n];
            }
        }
    }
    return;

}
__global__ void RungeKuttaStage_4_STT_ESDIRK54a_TorqueAndField(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
            MaterialHandle MatIndex = M->Mat[ind(i, j, k)];
            double Mk[3], Heff[3], Hstt[3];
            Vector LL;

#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mk[n] = M->M[mind(1, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Hstt[n] = H->H_STT[find(i, j, k, n)];
            }

            LL = EvaluateLandauLifshitz_STT_GPU(Mk, Heff,Hstt,MatIndex);


            //Update k3 torque
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                M->M[mind(5, i, j, k, n)] = LL.X[n];
            }

            double Heff_stage3[3];
            //Store and swap fields for stage 4
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Heff_stage3[n] = Heff[n];
                H->H_stage_1[find(i, j, k, n)] = Heff[n];
                H->H_eff[find(i, j, k, n)] = H->H_stage[find(i, j, k, n)];
                H->H_stage[find(i, j, k, n)] = Heff_stage3[n];
            }

        }
    }
    return;

}
__global__ void RungeKuttaStage_4_STT_ESDIRK54a(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Da = 0.260000000;
    double a41 = 0.22371961478320505, a42 = 0.47675532319799699, a43 = -0.06470895363112615;

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
           
            MaterialHandle MatIndex = M->Mat[ind(i, j, k)];
            double Mk[3], Mn[3], Heff[3],Hstt[3];

            int Pivot_s[3], Flag;
            Jacobian J_s;
            Vector LL, dzi, F_0;
            double z4[3], k1[3], k2[3], k3[3];



#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Mk[n] = M->M[mind(1, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Hstt[n] = H->H_STT[find(i, j, k, n)];

                k1[n] = M->M[mind(3, i, j, k, n)];
                k2[n] = M->M[mind(4, i, j, k, n)];
                k3[n] = M->M[mind(5, i, j, k, n)];
            }

            //Compute LU factorisation
            LLJacobian_ESDIRK_GPU(&J_s, Mk, Heff, Da);
            LLJacobian_STT_GPU(&J_s, Mk, Hstt, Da);
            Flag = Crout_LU_Decomposition_with_Pivoting(&J_s.J[0][0], &(Pivot_s[0]), DIM);

            z4[0] = Mk[0],
            z4[1] = Mk[1],
            z4[2] = Mk[2];
            dzi.X[0] = 0.0;
            dzi.X[1] = 0.0;
            dzi.X[2] = 0.0;

            //k4 system  
            for (int n = 0; n < NEWTONITERATIONS; n++)
            {
                F_0.X[0] = Mn[0];
                F_0.X[1] = Mn[1];
                F_0.X[2] = Mn[2];

                F_0.X[0] = fma(h_d, a41 * k1[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a41 * k1[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a41 * k1[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a42 * k2[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a42 * k2[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a42 * k2[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a43 * k3[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a43 * k3[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a43 * k3[2], F_0.X[2]);

                LL = EvaluateLandauLifshitz_STT_GPU(z4, Heff,Hstt,MatIndex);

                F_0.X[0] = fma(h_d, Da * LL.X[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, Da * LL.X[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, Da * LL.X[2], F_0.X[2]);

                if (n == 0) //initial guess
                {
#pragma unroll DIM
                    for (int i = 0; i < DIM; i++)
                    {
                        z4[i] = Mn[i] + (0.895765984350076) * h_d * F_0.X[i];
                    }
                }

                F_0.X[0] -= z4[0];
                F_0.X[1] -= z4[1];
                F_0.X[2] -= z4[2];

                Flag = Crout_LU_with_Pivoting_Solve(&J_s.J[0][0], &F_0.X[0], Pivot_s, &dzi.X[0], DIM);

                z4[0] += dzi.X[0], z4[1] += dzi.X[1], z4[2] += dzi.X[2];

                if ((fabs(dzi.X[0]) + fabs(dzi.X[1]) + fabs(dzi.X[2])) <= AbsTol)
                {
                    break;
                }
            }

            VectorNormalise(z4);
            //Update Mn1
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                M->M[mind(1, i, j, k, n)] = z4[n];
            }
        }
    }
    return;

}
__global__ void RungeKuttaStage_5_STT_ESDIRK54a(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Da = 0.260000000;
    double a51 = 0.16648564323248321, a52 = 0.10450018841591720, a53 = 0.03631482272098715,
        a54 = -0.13090704451073998;

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
            MaterialHandle MatIndex = M->Mat[ind(i, j, k)];
            double Mk[3], Mn[3], Heff[3],Hstt[3];

            int Pivot_s[3], Flag;
            Jacobian J_s;
            Vector LL, dzi, F_0;
            double z5[3], k1[3], k2[3], k3[3], k4[3];


#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Mk[n] = M->M[mind(1, i, j, k, n)];

                Heff[n] = H->H_eff[find(i, j, k, n)];
                Hstt[n] = H->H_STT[find(i, j, k, n)];

                k1[n] = M->M[mind(3, i, j, k, n)];
                k2[n] = M->M[mind(4, i, j, k, n)];
                k3[n] = M->M[mind(5, i, j, k, n)];

            }

            //Compute LU factorisation
            LLJacobian_ESDIRK_GPU(&J_s, Mk, Heff, Da);
            LLJacobian_STT_GPU(&J_s, Mk, Hstt, Da);
            Flag = Crout_LU_Decomposition_with_Pivoting(&J_s.J[0][0], &(Pivot_s[0]), DIM);

            LL = EvaluateLandauLifshitz_STT_GPU(Mk, Heff,Hstt,MatIndex);
            k4[0] = LL.X[0];
            k4[1] = LL.X[1];
            k4[2] = LL.X[2];

            z5[0] = Mn[0],
            z5[1] = Mn[1],
            z5[2] = Mn[2];

            dzi.X[0] = 0.0;
            dzi.X[1] = 0.0;
            dzi.X[2] = 0.0;

            //k5 system  
            for (int n = 0; n < NEWTONITERATIONS; n++)
            {
                F_0.X[0] = Mn[0];
                F_0.X[1] = Mn[1];
                F_0.X[2] = Mn[2];

                F_0.X[0] = fma(h_d, a51 * k1[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a51 * k1[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a51 * k1[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a52 * k2[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a52 * k2[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a52 * k2[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a53 * k3[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a53 * k3[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a53 * k3[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a54 * k4[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a54 * k4[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a54 * k4[2], F_0.X[2]);

                LL = EvaluateLandauLifshitz_STT_GPU(z5, Heff,Hstt,MatIndex);

                F_0.X[0] = fma(h_d, Da * LL.X[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, Da * LL.X[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, Da * LL.X[2], F_0.X[2]);

                if (n == 0) //initial guess
                {
#pragma unroll DIM
                    for (int i = 0; i < DIM; i++)
                    {
                        z5[i] = Mn[i] + (0.436393609858648) * h_d * F_0.X[i];
                    }
                }

                F_0.X[0] -= z5[0];
                F_0.X[1] -= z5[1];
                F_0.X[2] -= z5[2];

                Flag = Crout_LU_with_Pivoting_Solve(&J_s.J[0][0], &F_0.X[0], Pivot_s, &dzi.X[0], DIM);

                z5[0] += dzi.X[0], z5[1] += dzi.X[1], z5[2] += dzi.X[2];

                if ((fabs(dzi.X[0]) + fabs(dzi.X[1]) + fabs(dzi.X[2])) <= AbsTol)
                {
                    break;
                }
            }

            VectorNormalise(z5);
            //Update Mn1
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                M->M[mind(1, i, j, k, n)] = z5[n];
                M->M[mind(6, i, j, k, n)] = k4[n];
            }
        }
    }
    return;
}
__global__ void RungeKuttaStage_6_STT_ESDIRK54a(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Da = 0.260000000;
    double a61 = 0.13855640231268224, a62 = 0.0, a63 = -0.04245337201752043,
        a64 = 0.02446657898003141, a65 = 0.61943039072480676;

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
            MaterialHandle MatIndex = M->Mat[ind(i, j, k)];
            double Mk[3], Mn[3], Heff[3],Hstt[3];

            int Pivot_s[3], Flag;
            Jacobian J_s;
            Vector LL, dzi, F_0;
            double z6[3], k1[3], k2[3], k3[3], k4[3], k5[3];


#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Mk[n] = M->M[mind(1, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Hstt[n] = H->H_STT[find(i, j, k, n)];

                k1[n] = M->M[mind(3, i, j, k, n)];
                k2[n] = M->M[mind(4, i, j, k, n)];
                k3[n] = M->M[mind(5, i, j, k, n)];
                k4[n] = M->M[mind(6, i, j, k, n)];
            }

            //Compute LU factorisation
            LLJacobian_ESDIRK_GPU(&J_s, Mk, Heff, Da);
            LLJacobian_STT_GPU(&J_s, Mk, Hstt, Da);
            Flag = Crout_LU_Decomposition_with_Pivoting(&J_s.J[0][0], &(Pivot_s[0]), DIM);

            LL = EvaluateLandauLifshitz_STT_GPU(Mk, Heff,Hstt,MatIndex);
            k5[0] = LL.X[0];
            k5[1] = LL.X[1];
            k5[2] = LL.X[2];

            z6[0] = Mk[0],
            z6[1] = Mk[1],
            z6[2] = Mk[2];
            dzi.X[0] = 0.0;
            dzi.X[1] = 0.0;
            dzi.X[2] = 0.0;


            //k6 system  
            for (int n = 0; n < NEWTONITERATIONS; n++)
            {
                F_0.X[0] = Mn[0];
                F_0.X[1] = Mn[1];
                F_0.X[2] = Mn[2];

                F_0.X[0] = fma(h_d, a61 * k1[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a61 * k1[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a61 * k1[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a62 * k2[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a62 * k2[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a62 * k2[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a63 * k3[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a63 * k3[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a63 * k3[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a64 * k4[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a64 * k4[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a64 * k4[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a65 * k5[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a65 * k5[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a65 * k5[2], F_0.X[2]);

                LL = EvaluateLandauLifshitz_STT_GPU(z6, Heff,Hstt,MatIndex);

                F_0.X[0] = fma(h_d, Da * LL.X[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, Da * LL.X[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, Da * LL.X[2], F_0.X[2]);

                if (n == 0) //initial guess
                {
#pragma unroll DIM
                    for (int i = 0; i < DIM; i++)
                    {
                        z6[i] = Mn[i] + h_d * F_0.X[i];
                    }
                }

                F_0.X[0] -= z6[0];
                F_0.X[1] -= z6[1];
                F_0.X[2] -= z6[2];

                Flag = Crout_LU_with_Pivoting_Solve(&J_s.J[0][0], &F_0.X[0], Pivot_s, &dzi.X[0], DIM);

                z6[0] += dzi.X[0], z6[1] += dzi.X[1], z6[2] += dzi.X[2];

                if ((fabs(dzi.X[0]) + fabs(dzi.X[1]) + fabs(dzi.X[2])) <= AbsTol)
                {
                    break;
                }
            }

            VectorNormalise(z6);
            //Update Mn1
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                M->M[mind(1, i, j, k, n)] = z6[n];
                M->M[mind(7, i, j, k, n)] = k5[n];
            }
        }
    }
    return;
}
__global__ void RungeKuttaStage_7_STT_ESDIRK54a(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Da = 0.260000000;
    double a71 = 0.13659751177640291, a72 = 0.0, a73 = -0.05496908796538376, a74 = -0.04118626728321046,
        a75 = 0.62993304899016403, a76 = 0.06962479448202728;

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
            MaterialHandle MatIndex = M->Mat[ind(i, j, k)];
            double Mk[3], Mn[3], Heff[3],Hstt[3];

            int Pivot_s[3], Flag;
            Jacobian J_s;
            Vector LL, dzi, F_0;
            double z7[3], k1[3], k2[3], k3[3], k4[3], k5[3], k6[3];


#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Mk[n] = M->M[mind(1, i, j, k, n)];

                //Error method stage 6
                M->M[mind(9, i, j, k, n)] = Mk[n];

                Heff[n] = H->H_eff[find(i, j, k, n)];
                Hstt[n] = H->H_STT[find(i, j, k, n)];

                k1[n] = M->M[mind(3, i, j, k, n)];
                k2[n] = M->M[mind(4, i, j, k, n)];
                k3[n] = M->M[mind(5, i, j, k, n)];
                k4[n] = M->M[mind(6, i, j, k, n)];
                k5[n] = M->M[mind(7, i, j, k, n)];
            }

            //Compute LU factorisation
            LLJacobian_ESDIRK_GPU(&J_s, Mk, Heff, Da);
            LLJacobian_STT_GPU(&J_s, Mk, Hstt, Da);
            Flag = Crout_LU_Decomposition_with_Pivoting(&J_s.J[0][0], &(Pivot_s[0]), DIM);

            LL = EvaluateLandauLifshitz_STT_GPU(Mk, Heff,Hstt,MatIndex);
            k6[0] = LL.X[0];
            k6[1] = LL.X[1];
            k6[2] = LL.X[2];


            z7[0] = Mk[0],
            z7[1] = Mk[1],
            z7[2] = Mk[2];
            dzi.X[0] = 0.0;
            dzi.X[1] = 0.0;
            dzi.X[2] = 0.0;


            //k7 system  
            for (int n = 0; n < NEWTONITERATIONS; n++)
            {
                F_0.X[0] = Mn[0];
                F_0.X[1] = Mn[1];
                F_0.X[2] = Mn[2];

                F_0.X[0] = fma(h_d, a71 * k1[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a71 * k1[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a71 * k1[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a72 * k2[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a72 * k2[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a72 * k2[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a73 * k3[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a73 * k3[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a73 * k3[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a74 * k4[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a74 * k4[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a74 * k4[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a75 * k5[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a75 * k5[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a75 * k5[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a76 * k6[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a76 * k6[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a76 * k6[2], F_0.X[2]);

                LL = EvaluateLandauLifshitz_STT_GPU(z7, Heff,Hstt,MatIndex);

                F_0.X[0] = fma(h_d, Da * LL.X[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, Da * LL.X[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, Da * LL.X[2], F_0.X[2]);

                F_0.X[0] -= z7[0];
                F_0.X[1] -= z7[1];
                F_0.X[2] -= z7[2];

                Flag = Crout_LU_with_Pivoting_Solve(&J_s.J[0][0], &F_0.X[0], Pivot_s, &dzi.X[0], DIM);

                z7[0] += dzi.X[0], z7[1] += dzi.X[1], z7[2] += dzi.X[2];

                if ((fabs(dzi.X[0]) + fabs(dzi.X[1]) + fabs(dzi.X[2])) <= AbsTol)
                {
                    break;
                }
            }

            VectorNormalise(z7);

            //Update Mn1
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                M->M[mind(1, i, j, k, n)] = z7[n];
                M->M[mind(8, i, j, k, n)] = k6[n];
            }
        }
    }
    return;
}
__global__ void RungeKuttaFinalSolution_STT_ESDIRK54a(MAG M, FIELD H, MEMDATA DATA)
{
    
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    Vector Data;
    Data.X[0] = 0.0;
    Data.X[1] = 0.0;
    Data.X[2] = 0.0;

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
            MaterialHandle MatIndex = M->Mat[ind(i, j, k)];
            //RK Error Stage
            double Da = 0.260000000;
            double a61 = 0.13855640231268224, a62 = 0.0, a63 = -0.04245337201752043,
                a64 = 0.02446657898003141, a65 = 0.61943039072480676;
            double a71 = 0.13659751177640291, a72 = 0.0, a73 = -0.05496908796538376, a74 = -0.04118626728321046,
                a75 = 0.62993304899016403, a76 = 0.06962479448202728;

            double Mn[3], Mn1[3],Mnk[3], Heff[3],Hstt[3], Mns[3];

#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Hstt[n] = H->H_STT[find(i, j, k, n)];
                Mn1[n] = M->M[mind(1, i, j, k, n)];
            }

            Vector dmdt = EvaluateLandauLifshitz_STT_GPU(Mn1, Heff,Hstt,MatIndex);

            for (int n = 0; n < DIM; n++)
            {
                Mns[n] = fma((a71 - a61) * h_d, M->M[mind(3, i, j, k, n)], 0.0);

                Mns[n] = fma((a73 - a63) * h_d, M->M[mind(5, i, j, k, n)], Mns[n]);
                Mns[n] = fma((a74 - a64) * h_d, M->M[mind(6, i, j, k, n)], Mns[n]);
                Mns[n] = fma((a75 - a65) * h_d, M->M[mind(7, i, j, k, n)], Mns[n]);
                Mns[n] = fma((a76 - Da) * h_d, M->M[mind(8, i, j, k, n)], Mns[n]);
                Mns[n] = fma(Da * h_d, dmdt.X[n], Mns[n]);
            }

            M->M[mind(0, i, j, k, 0)] = Mn1[0];
            M->M[mind(0, i, j, k, 1)] = Mn1[1];
            M->M[mind(0, i, j, k, 2)] = Mn1[2];

            Data.X[0] = sqrt(Mns[0] * Mns[0] + Mns[1] * Mns[1]
                + Mns[2] * Mns[2]);


            Mns[0] = fmax(dmdt.X[0], dmdt.X[1]);
            Mns[0] = fmax(Mns[0], dmdt.X[2]);

            Mns[1] = fmin(dmdt.X[0], dmdt.X[1]);
            Mns[1] = fmin(Mns[1], dmdt.X[2]);

            Data.X[1] = Mns[0]; //Max Torque
            Data.X[2] = Mns[1]; //Min Torque

            //Store LTE in index 8,xcomp, Max torque in index 8,ycomp 
            //Min torque in index 8,zcomp
            M->M[mind(8, i, j, k, 0)] = Data.X[0];
            M->M[mind(8, i, j, k, 1)] = Data.X[1];
            M->M[mind(8, i, j, k, 2)] = Data.X[2];
        }
    }
    return;
}
__host__ void RungeKuttaESDIRK54a_STT_StageEvaluations(MAG M_d, FIELD H_d, MEMDATA DATA, PLANS P, double h,
    int* ResetFlag, int AllocSize)
{

    double c2 = 0.520000000000000, c3 = 1.230333209967908,
        c4 = 0.895765984350076, c5 = 0.436393609858648,
        c6 = 1.;

    double tn = t_h;

    RungeKuttaStage_CopyPreviousStep << <NumberofBlocksIntegrator, NumberofThreadsIntegrator>> >
        (M_d, H_d, DATA, *ResetFlag);

    t_h = tn + c2 * h;
    UpdateDeviceTime(t_h);

    if (*ResetFlag == 1)
    {
        ComputeFields(DATA, M_d, H_d, P, 0);
    }


    RungeKuttaStage_2_STT_ESDIRK54a << <NumberofBlocksIntegrator, NumberofThreadsIntegrator>> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c3 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_3_STT_ESDIRK54a << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c4 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_4_STT_ESDIRK54a_TorqueAndField << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    RungeKuttaStage_4_STT_ESDIRK54a << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c5 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_5_STT_ESDIRK54a << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c6 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_6_STT_ESDIRK54a << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_7_STT_ESDIRK54a << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);


    RungeKuttaFinalSolution_STT_ESDIRK54a << <NumberofBlocksIntegrator, NumberofThreadsIntegrator>> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    return;
}
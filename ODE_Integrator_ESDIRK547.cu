#include "ODE_Integrator_ESDIRK547.cuh"
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

__global__ void RungeKuttaStage_2_ESDIRK547(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Da = 23. / 125.;
    double a21 = Da;

    int Index = (i * NUMY + j) * NUMZ + k;

    if (Index < (NUM * NUMY * NUMZ))
    {
        if (M->Mat[ind(i, j, k)] == 1)
        {

            double Mk[3], Mn[3], Heff[3];

            int Pivot_s[3], Flag;
            Jacobian J_s;
            Vector LL, dzi, F_0;
            double z2[3];


#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Mk[n] = Mn[n];
                //Store Mn field term
                H->H_stage[find(i, j, k, n)] = Heff[n];
            }
            z2[0] = Mn[0],
                z2[1] = Mn[1],
                z2[2] = Mn[2];
            dzi.X[0] = 0.0;
            dzi.X[1] = 0.0;
            dzi.X[2] = 0.0;

            //Compute LU factorisation
            LLJacobian_DIRKEvaluate_GPU(&J_s, Mk, Heff, Da);
            Flag = Crout_LU_Decomposition_with_Pivoting(&J_s.J[0][0], &(Pivot_s[0]), DIM);

            //Store
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                M->J[ind(i, j, k)].J[0][n] = J_s.J[0][n];
                M->J[ind(i, j, k)].J[1][n] = J_s.J[1][n];
                M->J[ind(i, j, k)].J[2][n] = J_s.J[2][n];
                M->Pivot[find(i, j, k, n)] = Pivot_s[n];
            }

            Vector LLn = EvaluateLandauLifshitz_GPU(Mk, Heff);

            //k2 system  
            for (int n = 0; n < NEWTONITERATIONS; n++)
            {
                F_0.X[0] = Mn[0];
                F_0.X[1] = Mn[1];
                F_0.X[2] = Mn[2];

                F_0.X[0] = fma(h_d, a21 * LLn.X[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a21 * LLn.X[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a21 * LLn.X[2], F_0.X[2]);

                LL = EvaluateLandauLifshitz_GPU(z2, Heff);

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
__global__ void RungeKuttaStage_3_ESDIRK547(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Da = 23. / 125.;
    double a31 = 791020047304. / 3561426431547.,
        a32 = a31;

    int Index = (i * NUMY + j) * NUMZ + k;

    if (Index < (NUM * NUMY * NUMZ))
    {
        if (M->Mat[ind(i, j, k)] == 1)
        {

            double Mk[3], Mn[3], Heff[3];

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

                k1[n] = M->M[mind(3, i, j, k, n)];

                J_s.J[0][n] = M->J[ind(i, j, k)].J[0][n];
                J_s.J[1][n] = M->J[ind(i, j, k)].J[1][n];
                J_s.J[2][n] = M->J[ind(i, j, k)].J[2][n];

                Pivot_s[n] = M->Pivot[find(i, j, k, n)];

            }

            LL = EvaluateLandauLifshitz_GPU(Mk, Heff);
            k2[0] = LL.X[0];
            k2[1] = LL.X[1];
            k2[2] = LL.X[2];

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

                LL = EvaluateLandauLifshitz_GPU(z3, Heff);

                F_0.X[0] = fma(h_d, Da * LL.X[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, Da * LL.X[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, Da * LL.X[2], F_0.X[2]);

                if (n == 0) //initial guess
                {
#pragma unroll DIM
                    for (int i = 0; i < DIM; i++)
                    {
                        z3[i] = Mn[i] + (7121331996143. / 11335814405378.) * h_d * F_0.X[i];
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
__global__ void RungeKuttaStage_4_ESDIRK547_TorqueAndField(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    int Index = (i * NUMY + j) * NUMZ + k;

    if (Index < (NUM * NUMY * NUMZ))
    {
        if (M->Mat[ind(i, j, k)] == 1)
        {

            double Mk[3], Heff[3];
            Vector LL;

#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mk[n] = M->M[mind(1, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
            }

            LL = EvaluateLandauLifshitz_GPU(Mk, Heff);


            //Update torque
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
                H->H_eff[find(i, j, k, n)] = H->H_stage[find(i, j, k, n)];
                // H->H_stage[find(i, j, k, n)] = Heff_stage3[n];
            }
        }
    }
    return;

}
__global__ void RungeKuttaStage_4_ESDIRK547(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Da = 23. / 125.;
    double a41 = -158159076358. / 11257294102345.,
        a42 = a41,
        a43 = -85517644447. / 5003708988389.;

    int Index = (i * NUMY + j) * NUMZ + k;

    if (Index < (NUM * NUMY * NUMZ))
    {
        if (M->Mat[ind(i, j, k)] == 1)
        {

            double Mk[3], Mn[3], Heff[3];

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

                k1[n] = M->M[mind(3, i, j, k, n)];
                k2[n] = M->M[mind(4, i, j, k, n)];
                k3[n] = M->M[mind(5, i, j, k, n)];

                J_s.J[0][n] = M->J[ind(i, j, k)].J[0][n];
                J_s.J[1][n] = M->J[ind(i, j, k)].J[1][n];
                J_s.J[2][n] = M->J[ind(i, j, k)].J[2][n];

                Pivot_s[n] = M->Pivot[find(i, j, k, n)];

            }

            z4[0] = Mn[0],
                z4[1] = Mn[1],
                z4[2] = Mn[2];
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

                LL = EvaluateLandauLifshitz_GPU(z4, Heff);

                F_0.X[0] = fma(h_d, Da * LL.X[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, Da * LL.X[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, Da * LL.X[2], F_0.X[2]);

                if (n == 0) //initial guess
                {
#pragma unroll DIM
                    for (int i = 0; i < DIM; i++)
                    {
                        z4[i] = Mn[i] + (49. / 353.) * h_d * F_0.X[i];
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

            //Update Mn1
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                M->M[mind(1, i, j, k, n)] = z4[n];
                // M->M[mind(5, i, j, k, n)] = k3[n];
            }
        }
    }
    return;

}
__global__ void RungeKuttaStage_5_ESDIRK547(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Da = 23. / 125.;
    double a51 = -1653327111580. / 4048416487981.,
        a52 = a51,
        a53 = 1514767744496. / 9099671765375.,
        a54 = 14283835447591. / 12247432691556.;

    int Index = (i * NUMY + j) * NUMZ + k;

    if (Index < (NUM * NUMY * NUMZ))
    {
        if (M->Mat[ind(i, j, k)] == 1)
        {

            double Mk[3], Mn[3], Heff[3];

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

                k1[n] = M->M[mind(3, i, j, k, n)];
                k2[n] = M->M[mind(4, i, j, k, n)];
                k3[n] = M->M[mind(5, i, j, k, n)];

                J_s.J[0][n] = M->J[ind(i, j, k)].J[0][n];
                J_s.J[1][n] = M->J[ind(i, j, k)].J[1][n];
                J_s.J[2][n] = M->J[ind(i, j, k)].J[2][n];

                Pivot_s[n] = M->Pivot[find(i, j, k, n)];

            }


            LL = EvaluateLandauLifshitz_GPU(Mk, Heff);
            k4[0] = LL.X[0];
            k4[1] = LL.X[1];
            k4[2] = LL.X[2];

            z5[0] = Mk[0],
                z5[1] = Mk[1],
                z5[2] = Mk[2];
            dzi.X[0] = 0.0;
            dzi.X[1] = 0.0;
            dzi.X[2] = 0.0;


            //swap fields for stage 5 back from stage 3
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                H->H_eff[find(i, j, k, n)] = H->H_stage[find(i, j, k, n)];
            }


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

                LL = EvaluateLandauLifshitz_GPU(z5, Heff);

                F_0.X[0] = fma(h_d, Da * LL.X[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, Da * LL.X[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, Da * LL.X[2], F_0.X[2]);

                if (n == 0) //initial guess
                {
#pragma unroll DIM
                    for (int i = 0; i < DIM; i++)
                    {
                        z5[i] = Mn[i] + (3706679970760. / 5295570149437.) * h_d * F_0.X[i];
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
__global__ void RungeKuttaStage_6_ESDIRK547(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Da = 23. / 125.;
    double a61 = -4540011970825. / 8418487046959.,
        a62 = a61,
        a63 = -1790937573418. / 7393406387169.,
        a64 = 10819093665085. / 7266595846747.,
        a65 = 4109463131231. / 7386972500302.;

    int Index = (i * NUMY + j) * NUMZ + k;

    if (Index < (NUM * NUMY * NUMZ))
    {
        if (M->Mat[ind(i, j, k)] == 1)
        {

            double Mk[3], Mn[3], Heff[3];

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

                k1[n] = M->M[mind(3, i, j, k, n)];
                k2[n] = M->M[mind(4, i, j, k, n)];
                k3[n] = M->M[mind(5, i, j, k, n)];
                k4[n] = M->M[mind(6, i, j, k, n)];

                J_s.J[0][n] = M->J[ind(i, j, k)].J[0][n];
                J_s.J[1][n] = M->J[ind(i, j, k)].J[1][n];
                J_s.J[2][n] = M->J[ind(i, j, k)].J[2][n];

                Pivot_s[n] = M->Pivot[find(i, j, k, n)];

            }

            LL = EvaluateLandauLifshitz_GPU(Mk, Heff);
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

                LL = EvaluateLandauLifshitz_GPU(z6, Heff);

                F_0.X[0] = fma(h_d, Da * LL.X[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, Da * LL.X[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, Da * LL.X[2], F_0.X[2]);

                if (n == 0) //initial guess
                {
#pragma unroll DIM
                    for (int i = 0; i < DIM; i++)
                    {
                        z6[i] = Mn[i] + (347. / 382.) * h_d * F_0.X[i];
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
__global__ void RungeKuttaStage_7_ESDIRK547(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Da = 23. / 125.;
    double a71 = -188593204321. / 4778616380481.,
        a72 = a71,
        a73 = 2809310203510. / 10304234040467.,
        a74 = 1021729336898. / 2364210264653.,
        a75 = 870612361811. / 2470410392208.,
        a76 = -1307970675534. / 8059683598661.;

    int Index = (i * NUMY + j) * NUMZ + k;

    if (Index < (NUM * NUMY * NUMZ))
    {
        if (M->Mat[ind(i, j, k)] == 1)
        {

            double Mk[3], Mn[3], Heff[3];

            int Pivot_s[3], Flag;
            Jacobian J_s;
            Vector LL, dzi, F_0;
            double z7[3], k1[3], k2[3], k3[3], k4[3], k5[3], k6[3];


#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Mk[n] = M->M[mind(1, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];

                k1[n] = M->M[mind(3, i, j, k, n)];
                k2[n] = M->M[mind(4, i, j, k, n)];
                k3[n] = M->M[mind(5, i, j, k, n)];
                k4[n] = M->M[mind(6, i, j, k, n)];
                k5[n] = M->M[mind(7, i, j, k, n)];

                J_s.J[0][n] = M->J[ind(i, j, k)].J[0][n];
                J_s.J[1][n] = M->J[ind(i, j, k)].J[1][n];
                J_s.J[2][n] = M->J[ind(i, j, k)].J[2][n];

                Pivot_s[n] = M->Pivot[find(i, j, k, n)];

            }

            LL = EvaluateLandauLifshitz_GPU(Mk, Heff);
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

                LL = EvaluateLandauLifshitz_GPU(z7, Heff);

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
__global__ void RungeKuttaFinalSolution_ESDIRK547(MAG M, FIELD H, MEMDATA DATA)
{
    double* Heff;
    double* SH_M1;
    double* SH_M0;
    int* SH_Mat;

    extern __shared__ double DAT[];
    int SHAREDSIZE = blockDim.x * blockDim.y * blockDim.z;

    SH_M1 = &DAT[0];
    SH_M0 = &DAT[SHAREDSIZE * DIM];
    Heff = &SH_M0[SHAREDSIZE * DIM];
    SH_Mat = (int*)&Heff[SHAREDSIZE * DIM];

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    int I = threadIdx.z;
    int J = threadIdx.y;
    int K = threadIdx.x;

    int Index = (i * NUMY + j) * NUMZ + k;
    int tid = ((I * blockDim.y + J) * blockDim.x + K);

    Heff[tid * DIM] = 0.0;
    SH_M1[tid * DIM] = 0.0;
    SH_M0[tid * DIM] = 0.0;
    Heff[tid * DIM + 1] = 0.0;
    SH_M1[tid * DIM + 1] = 0.0;
    SH_M0[tid * DIM + 1] = 0.0;
    Heff[tid * DIM + 2] = 0.0;
    SH_M1[tid * DIM + 2] = 0.0;
    SH_M0[tid * DIM + 2] = 0.0;
    SH_Mat[tid] = 0.0;

    double Mn1[3], Mnk[3];
    Vector Data;

    if (Index < (NUM * NUMY * NUMZ))
    {
        if (M->Mat[ind(i, j, k)] != 0)
        {
            //RK stage 7

            double Da = 23. / 125.;


            double a71 = -188593204321. / 4778616380481.,
                a72 = a71,
                a73 = 2809310203510. / 10304234040467.,
                a74 = 1021729336898. / 2364210264653.,
                a75 = 870612361811. / 2470410392208.,
                a76 = -1307970675534. / 8059683598661.;

            double b1s = -582099335757. / 7214068459310.,
                b2s = b1s,
                b3s = 615023338567. / 3362626566945.,
                b4s = 3192122436311. / 6174152374399.,
                b5s = 6156034052041. / 14430468657929.,
                b6s = -1011318518279. / 9693750372484.,
                b7s = 1914490192573. / 13754262428401.;

            double Mn1[3], Mn[3], Heff[3];
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Mn1[n] = M->M[mind(1, i, j, k, n)];
                Mn[n] = M->M[mind(0, i, j, k, n)];
            }

            Vector dmdt = EvaluateLandauLifshitz_GPU(Mn1, Heff);


            //Error Estimate
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mnk[n] = fma((b1s)*h_d, M->M[mind(3, i, j, k, n)], Mn[n]);
                Mnk[n] = fma((b2s)*h_d, M->M[mind(4, i, j, k, n)], Mnk[n]);
                Mnk[n] = fma((b3s)*h_d, M->M[mind(5, i, j, k, n)], Mnk[n]);
                Mnk[n] = fma((b4s)*h_d, M->M[mind(6, i, j, k, n)], Mnk[n]);
                Mnk[n] = fma((b5s)*h_d, M->M[mind(7, i, j, k, n)], Mnk[n]);
                Mnk[n] = fma((b6s)*h_d, M->M[mind(8, i, j, k, n)], Mnk[n]);
                Mnk[n] = fma((b7s)*h_d, dmdt.X[n], Mnk[n]);
            }

            VectorNormalise(Mnk);

            Data.X[0] = 0.8 * sqrt(pow((Mnk[0] - Mn1[0]), 2.0) + pow((Mnk[1] - Mn1[1]), 2.0)
                + pow((Mnk[2] - Mn1[2]), 2.0));


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

        ////re-use shared memory for stepsize,MaxTorque,MinTorque;

        //SH_M0[tid * DIM] = Data.X[0]; //LTE Error
        //SH_M0[tid * DIM + 1] = 0.0;
        //SH_M0[tid * DIM + 2] = 0.0;

        //SH_M1[tid * DIM] = Data.X[1]; //Max Torque
        //SH_M1[tid * DIM + 1] = 0.0;
        //SH_M1[tid * DIM + 2] = 0.0;

        //Heff[tid * DIM] = Data.X[2]; //Min Torque
        //Heff[tid * DIM + 1] = 0.0;
        //Heff[tid * DIM + 2] = 0.0;

        //__syncthreads();

        ////Reductions

        //for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
        //{
        //    if (tid < s)
        //    {
        //        SH_M0[tid * DIM] = fmax(SH_M0[tid * DIM], SH_M0[(tid + s) * DIM]);
        //        SH_M1[tid * DIM] = fmax(SH_M1[tid * DIM], SH_M1[(tid + s) * DIM]);
        //        Heff[tid * DIM] = fmin(Heff[tid * DIM], Heff[(tid + s) * DIM]);
        //        __syncthreads();
        //    }
        //}

        //if (tid < 32)
        //{
        //    warpMaxReduce(SH_M0, tid * DIM);
        //    warpMaxReduce(SH_M1, tid * DIM);
        //    warpMinReduce(Heff, tid * DIM);
        //}

        //__syncthreads();

        //if (tid == 0)
        //{
        //    int bidx = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);
        //    DATA->StepReduction[bidx] = SH_M0[0];
        //    DATA->MaxTorqueReduction[bidx] = SH_M1[0];
        //    DATA->dE_Reduction[bidx] = Heff[0];
        //    __syncthreads();
        //}
    }
    return;
}
__host__ void RungeKuttaESDIRK547_StageEvaluations(MAG M_d, FIELD H_d, MEMDATA DATA, PLANS P, double h,
    int* ResetFlag, int AllocSize)
{

    double c2 = 46. / 125., c3 = 7121331996143. / 11335814405378.,
        c4 = 49. / 353., c5 = 3706679970760. / 5295570149437.,
        c6 = 347. / 382.;

    double tn = t_h;

    RungeKuttaStage_CopyPreviousStep << <NumberofBlocksIntegrator, NumberofThreadsIntegrator, AllocSize >> >
        (M_d, H_d, DATA, *ResetFlag);

    t_h = tn + c2 * h;
    UpdateDeviceTime(t_h);

    if (*ResetFlag == 1)
    {
        ComputeFields(DATA, M_d, H_d, P, 0);
    }


    RungeKuttaStage_2_ESDIRK547 << <NumberofBlocksIntegrator, NumberofThreadsIntegrator, AllocSize >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c3 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_3_ESDIRK547 << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c4 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_4_ESDIRK547_TorqueAndField << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    RungeKuttaStage_4_ESDIRK547 << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c5 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_5_ESDIRK547 << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c6 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_6_ESDIRK547 << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_7_ESDIRK547 << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);


    RungeKuttaFinalSolution_ESDIRK547 << <NumberofBlocksIntegrator, NumberofThreadsIntegrator, AllocSize >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    return;
}
#include "ODE_Integrator_ESDIRK659.cuh"
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

__global__ void RungeKuttaStage_2_ESDIRK659(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Da = 2. / 9.;
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
                //store field
                H->H_stage[find(i, j, k, n)] = Heff[n];
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
__global__ void RungeKuttaStage_3_ESDIRK659(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Da = 2. / 9.;
    double a31 = 1. / 9., a32 = -52295652026801. / 1014133226193379.;

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


#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Heff[n] = H->H_stage[find(i, j, k, n)];
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

                LL = EvaluateLandauLifshitz_GPU(z3, Heff);

                F_0.X[0] = fma(h_d, Da * LL.X[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, Da * LL.X[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, Da * LL.X[2], F_0.X[2]);

                if (n == 0) //initial guess
                {
#pragma unroll DIM
                    for (int i = 0; i < DIM; i++)
                    {
                        z3[i] = Mn[i] + (376327483029687. / 1335600577485745.) * h_d * F_0.X[i];
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
__global__ void RungeKuttaStage_4_ESDIRK659_TorqueAndField(MAG M, FIELD H, MEMDATA DATA)
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


            //Update k3 torque
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                M->M[mind(5, i, j, k, n)] = LL.X[n];
            }


        }
    }
    return;

}
__global__ void RungeKuttaStage_4_ESDIRK659(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Da = 2. / 9.;
    double a41 = 37633260247889. / 456511413219805.,
        a42 = -162541608159785. / 642690962402252.,
        a43 = 186915148640310. / 408032288622937.;

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

                LL = EvaluateLandauLifshitz_GPU(z4, Heff);

                F_0.X[0] = fma(h_d, Da * LL.X[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, Da * LL.X[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, Da * LL.X[2], F_0.X[2]);

                if (n == 0) //initial guess
                {
#pragma unroll DIM
                    for (int i = 0; i < DIM; i++)
                    {
                        z4[i] = Mn[i] + (433625707911282. / 850513180247701.) * h_d * F_0.X[i];
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
            }
        }
    }
    return;

}
__global__ void RungeKuttaStage_5_ESDIRK659(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Da = 2. / 9.;
    double a51 = -37161579357179. / 532208945751958.,
        a52 = -211140841282847. / 266150973773621.,
        a53 = 884359688045285. / 894827558443789.,
        a54 = 845261567597837. / 1489150009616527.;

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

                LL = EvaluateLandauLifshitz_GPU(z5, Heff);

                F_0.X[0] = fma(h_d, Da * LL.X[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, Da * LL.X[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, Da * LL.X[2], F_0.X[2]);

                if (n == 0) //initial guess
                {
#pragma unroll DIM
                    for (int i = 0; i < DIM; i++)
                    {
                        z5[i] = Mn[i] + (183. / 200.) * h_d * F_0.X[i];
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
__global__ void RungeKuttaStage_6_ESDIRK659(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Da = 2. / 9.;
    double a61 = 32386175866773. / 281337331200713.,
        a62 = 498042629717897. / 1553069719539220.,
        a63 = -73718535152787. / 262520491717733.,
        a64 = -147656452213061. / 931530156064788.,
        a65 = -16605385309793. / 2106054502776008.;

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

                H->H_stage[find(i, j, k, n)] = H->H_eff[find(i, j, k, n)];
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

            //swap for stage 2 field
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {

                Heff[n] = H->H_stage[find(i, j, k, n)];
            }

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
                        z6[i] = Mn[i] + (62409086037595. / 296036819031271.) * h_d * F_0.X[i];
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
__global__ void RungeKuttaStage_7_ESDIRK659(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Da = 2. / 9.;

    double a71 = -38317091100349. / 1495803980405525.,
        a72 = 233542892858682. / 880478953581929.,
        a73 = -281992829959331. / 709729395317651.,
        a74 = -52133614094227. / 895217507304839.,
        a75 = -9321507955616. / 673810579175161.,
        a76 = 79481371174259. / 817241804646218.;

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


                if (n == 0) //initial guess
                {
#pragma unroll DIM
                    for (int i = 0; i < DIM; i++)
                    {
                        z7[i] = Mn[i] + (81796628710131. / 911762868125288.) * h_d * F_0.X[i];
                    }
                }

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
__global__ void RungeKuttaStage_8_ESDIRK659(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Da = 2. / 9.;

    double a81 = -486324380411713. / 1453057025607868.,
        a82 = -1085539098090580. / 1176943702490991.,
        a83 = 370161554881539. / 461122320759884.,
        a84 = 804017943088158. / 886363045286999.,
        a85 = -15204170533868. / 934878849212545.,
        a86 = -248215443403879. / 815097869999138.,
        a87 = 339987959782520. / 552150039467091.;

    int Index = (i * NUMY + j) * NUMZ + k;

    if (Index < (NUM * NUMY * NUMZ))
    {
        if (M->Mat[ind(i, j, k)] == 1)
        {

            double Mk[3], Mn[3], Heff[3];

            int Pivot_s[3], Flag;
            Jacobian J_s;
            Vector LL, dzi, F_0;
            double z8[3], k1[3], k2[3], k3[3], k4[3], k5[3], k6[3], k7[3];


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
                k6[n] = M->M[mind(8, i, j, k, n)];


                J_s.J[0][n] = M->J[ind(i, j, k)].J[0][n];
                J_s.J[1][n] = M->J[ind(i, j, k)].J[1][n];
                J_s.J[2][n] = M->J[ind(i, j, k)].J[2][n];

                Pivot_s[n] = M->Pivot[find(i, j, k, n)];

            }

            LL = EvaluateLandauLifshitz_GPU(Mk, Heff);
            k7[0] = LL.X[0];
            k7[1] = LL.X[1];
            k7[2] = LL.X[2];

            z8[0] = Mk[0],
                z8[1] = Mk[1],
                z8[2] = Mk[2];
            dzi.X[0] = 0.0;
            dzi.X[1] = 0.0;
            dzi.X[2] = 0.0;

            //swap for stage 5 field
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {

                Heff[n] = H->H_stage[find(i, j, k, n)];
            }


            //k6 system  
            for (int n = 0; n < NEWTONITERATIONS; n++)
            {
                F_0.X[0] = Mn[0];
                F_0.X[1] = Mn[1];
                F_0.X[2] = Mn[2];

                F_0.X[0] = fma(h_d, a81 * k1[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a81 * k1[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a81 * k1[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a82 * k2[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a82 * k2[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a82 * k2[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a83 * k3[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a83 * k3[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a83 * k3[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a84 * k4[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a84 * k4[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a84 * k4[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a85 * k5[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a85 * k5[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a85 * k5[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a86 * k6[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a86 * k6[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a86 * k6[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a87 * k7[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a87 * k7[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a87 * k7[2], F_0.X[2]);

                LL = EvaluateLandauLifshitz_GPU(z8, Heff);

                F_0.X[0] = fma(h_d, Da * LL.X[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, Da * LL.X[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, Da * LL.X[2], F_0.X[2]);

                if (n == 0) //initial guess
                {
#pragma unroll DIM
                    for (int i = 0; i < DIM; i++)
                    {
                        z8[i] = Mn[i] + (97. / 100.) * h_d * F_0.X[i];
                    }
                }


                F_0.X[0] -= z8[0];
                F_0.X[1] -= z8[1];
                F_0.X[2] -= z8[2];

                Flag = Crout_LU_with_Pivoting_Solve(&J_s.J[0][0], &F_0.X[0], Pivot_s, &dzi.X[0], DIM);

                z8[0] += dzi.X[0], z8[1] += dzi.X[1], z8[2] += dzi.X[2];

                if ((fabs(dzi.X[0]) + fabs(dzi.X[1]) + fabs(dzi.X[2])) <= AbsTol)
                {
                    break;
                }
            }

            VectorNormalise(z8);
            //Update Mn1
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                M->M[mind(1, i, j, k, n)] = z8[n];
                M->M[mind(9, i, j, k, n)] = k7[n];
            }
        }
    }
    return;
}
__global__ void RungeKuttaStage_9_ESDIRK659(MAG M, FIELD H, MEMDATA DATA)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Da = 2. / 9.;

    double a91 = 0.,
        a92 = 0.,
        a93 = 0.,
        a94 = 281246836687281. / 672805784366875.,
        a95 = 250674029546725. / 464056298040646.,
        a96 = 88917245119922. / 798581755375683.,
        a97 = 127306093275639. / 658941305589808.,
        a98 = -319515475352107. / 658842144391777.;

    int Index = (i * NUMY + j) * NUMZ + k;

    if (Index < (NUM * NUMY * NUMZ))
    {
        if (M->Mat[ind(i, j, k)] == 1)
        {

            double Mk[3], Mn[3], Heff[3];

            int Pivot_s[3], Flag;
            Jacobian J_s;
            Vector LL, dzi, F_0;
            double z9[3], k1[3], k2[3], k3[3], k4[3], k5[3], k6[3], k7[3], k8[3];


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
                k6[n] = M->M[mind(8, i, j, k, n)];
                k7[n] = M->M[mind(9, i, j, k, n)];


                J_s.J[0][n] = M->J[ind(i, j, k)].J[0][n];
                J_s.J[1][n] = M->J[ind(i, j, k)].J[1][n];
                J_s.J[2][n] = M->J[ind(i, j, k)].J[2][n];

                Pivot_s[n] = M->Pivot[find(i, j, k, n)];

            }

            LL = EvaluateLandauLifshitz_GPU(Mk, Heff);
            k8[0] = LL.X[0];
            k8[1] = LL.X[1];
            k8[2] = LL.X[2];

            z9[0] = Mk[0],
                z9[1] = Mk[1],
                z9[2] = Mk[2];
            dzi.X[0] = 0.0;
            dzi.X[1] = 0.0;
            dzi.X[2] = 0.0;


            //k6 system  
            for (int n = 0; n < NEWTONITERATIONS; n++)
            {
                F_0.X[0] = Mn[0];
                F_0.X[1] = Mn[1];
                F_0.X[2] = Mn[2];


                F_0.X[0] = fma(h_d, a94 * k4[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a94 * k4[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a94 * k4[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a95 * k5[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a95 * k5[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a95 * k5[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a96 * k6[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a96 * k6[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a96 * k6[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a97 * k7[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a97 * k7[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a97 * k7[2], F_0.X[2]);

                F_0.X[0] = fma(h_d, a98 * k8[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, a98 * k8[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, a98 * k8[2], F_0.X[2]);

                LL = EvaluateLandauLifshitz_GPU(z9, Heff);

                F_0.X[0] = fma(h_d, Da * LL.X[0], F_0.X[0]);
                F_0.X[1] = fma(h_d, Da * LL.X[1], F_0.X[1]);
                F_0.X[2] = fma(h_d, Da * LL.X[2], F_0.X[2]);

                F_0.X[0] -= z9[0];
                F_0.X[1] -= z9[1];
                F_0.X[2] -= z9[2];

                Flag = Crout_LU_with_Pivoting_Solve(&J_s.J[0][0], &F_0.X[0], Pivot_s, &dzi.X[0], DIM);

                z9[0] += dzi.X[0], z9[1] += dzi.X[1], z9[2] += dzi.X[2];

                if ((fabs(dzi.X[0]) + fabs(dzi.X[1]) + fabs(dzi.X[2])) <= AbsTol)
                {
                    break;
                }
            }

            VectorNormalise(z9);
            //Update Mn1
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                M->M[mind(1, i, j, k, n)] = z9[n];
                M->M[mind(10, i, j, k, n)] = k8[n];
            }
        }
    }
    return;
}
__global__ void RungeKuttaFinalSolution_ESDIRK659(MAG M, FIELD H, MEMDATA DATA)
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
        if (M->Mat[ind(i, j, k)] == 1)
        {
            //RK Error Stage
            double Da = 2. / 9.;

            double a91 = 0.,
                a92 = 0.,
                a93 = 0.,
                a94 = 281246836687281. / 672805784366875.,
                a95 = 250674029546725. / 464056298040646.,
                a96 = 88917245119922. / 798581755375683.,
                a97 = 127306093275639. / 658941305589808.,
                a98 = -319515475352107. / 658842144391777.;

            double b1s = -204006714482445. / 253120897457864.,
                b2s = 0.0,
                b3s = -818062434310719. / 743038324242217.,
                b4s = 3176520686137389. / 1064235527052079.,
                b5s = -574817982095666. / 1374329821545869.,
                b6s = -507643245828272. / 1001056758847831.,
                b7s = 2013538191006793. / 972919262949000.,
                b8s = 352681731710820. / 726444701718347.,
                b9s = -12107714797721. / 746708658438760.;

            double Mn[3], Mn1[3], Heff[3], Mns[3];
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                Mn[n] = M->M[mind(0, i, j, k, n)];
                Heff[n] = H->H_eff[find(i, j, k, n)];
                Mn1[n] = M->M[mind(1, i, j, k, n)];
            }

            Vector dmdt = EvaluateLandauLifshitz_GPU(Mn1, Heff);

            for (int n = 0; n < DIM; n++)
            {
                Mns[n] = fma((-b1s) * h_d, M->M[mind(3, i, j, k, n)], 0.0);

                Mns[n] = fma((-b3s) * h_d, M->M[mind(5, i, j, k, n)], Mns[n]);
                Mns[n] = fma((a94 - b4s) * h_d, M->M[mind(6, i, j, k, n)], Mns[n]);
                Mns[n] = fma((a95 - b5s) * h_d, M->M[mind(7, i, j, k, n)], Mns[n]);
                Mns[n] = fma((a96 - b6s) * h_d, M->M[mind(8, i, j, k, n)], Mns[n]);
                Mns[n] = fma((a97 - b7s) * h_d, M->M[mind(9, i, j, k, n)], Mns[n]);
                Mns[n] = fma((a98 - b8s) * h_d, M->M[mind(10, i, j, k, n)], Mns[n]);
                Mns[n] = fma((Da - b9s) * h_d, dmdt.X[n], Mns[n]);
            }

            M->M[mind(0, i, j, k, 0)] = Mn1[0];
            M->M[mind(0, i, j, k, 1)] = Mn1[1];
            M->M[mind(0, i, j, k, 2)] = Mn1[2];

            Data.X[0] = 0.9 * sqrt(Mns[0] * Mns[0] + Mns[1] * Mns[1]
                + Mns[2] * Mns[2]);


            Mns[0] = fmax(dmdt.X[0], dmdt.X[1]);
            Mns[0] = fmax(Mns[0], dmdt.X[2]);

            Mns[1] = fmin(dmdt.X[0], dmdt.X[1]);
            Mns[1] = fmin(Mns[1], dmdt.X[2]);

            Data.X[1] = Mns[0]; //Max Torque
            Data.X[2] = Mns[1]; //Min Torque
        }

        if (M->Mat[ind(i, j, k)] == 1) //Only write to global memory if integration took place
        {
            // M->M[mind(0, i, j, k, 0)] = M->M[mind(1, i, j, k, 0)];
            // M->M[mind(0, i, j, k, 1)] = M->M[mind(1, i, j, k, 1)];
            // M->M[mind(0, i, j, k, 2)] = M->M[mind(1, i, j, k, 2)];
        }
        else
        {
        }

        //re-use shared memory for stepsize,MaxTorque,MinTorque;

        SH_M0[tid * DIM] = Data.X[0]; //LTE Error
        SH_M0[tid * DIM + 1] = 0.0;
        SH_M0[tid * DIM + 2] = 0.0;

        SH_M1[tid * DIM] = Data.X[1]; //Max Torque
        SH_M1[tid * DIM + 1] = 0.0;
        SH_M1[tid * DIM + 2] = 0.0;

        Heff[tid * DIM] = Data.X[2]; //Min Torque
        Heff[tid * DIM + 1] = 0.0;
        Heff[tid * DIM + 2] = 0.0;

        __syncthreads();

        //Reductions

        for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
        {
            if (tid < s)
            {
                SH_M0[tid * DIM] = fmax(SH_M0[tid * DIM], SH_M0[(tid + s) * DIM]);
                SH_M1[tid * DIM] = fmax(SH_M1[tid * DIM], SH_M1[(tid + s) * DIM]);
                Heff[tid * DIM] = fmin(Heff[tid * DIM], Heff[(tid + s) * DIM]);
                __syncthreads();
            }
        }

        if (tid < 32)
        {
            warpMaxReduce(SH_M0, tid * DIM);
            warpMaxReduce(SH_M1, tid * DIM);
            warpMinReduce(Heff, tid * DIM);
        }

        __syncthreads();

        if (tid == 0)
        {
            int bidx = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);
            DATA->StepReduction[bidx] = SH_M0[0];
            DATA->MaxTorqueReduction[bidx] = SH_M1[0];
            DATA->dE_Reduction[bidx] = Heff[0];
            __syncthreads();
        }
    }
    return;
}
__host__ void RungeKuttaESDIRK659_StageEvaluations(MAG M_d, FIELD H_d, MEMDATA DATA, PLANS P, double h,
    int* ResetFlag, int AllocSize)
{

    double c2 = 4. / 9.,
        c3 = 376327483029687. / 1335600577485745.,
        c4 = 433625707911282. / 850513180247701.,
        c5 = 183. / 200.,
        c6 = 62409086037595. / 296036819031271.,
        c7 = 81796628710131. / 911762868125288.,
        c8 = 97. / 100.,
        c9 = 1.0;

    double tn = t_h;

    RungeKuttaStage_CopyPreviousStep << <NumberofBlocksIntegrator, NumberofThreadsIntegrator, AllocSize >> >
        (M_d, H_d, DATA, *ResetFlag);

    t_h = tn + c2 * h;
    UpdateDeviceTime(t_h);

    if (*ResetFlag == 1)
    {
        ComputeFields(DATA, M_d, H_d, P, 0);
    }


    RungeKuttaStage_2_ESDIRK659 << <NumberofBlocksIntegrator, NumberofThreadsIntegrator, AllocSize >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c3 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_3_ESDIRK659 << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c4 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_4_ESDIRK659_TorqueAndField << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    RungeKuttaStage_4_ESDIRK659 << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c5 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_5_ESDIRK659 << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c6 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_6_ESDIRK659 << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    t_h = tn + c7 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_7_ESDIRK659 << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    t_h = tn + c8 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_8_ESDIRK659 << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);


    t_h = tn + c9 * h;
    UpdateDeviceTime(t_h);

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaStage_9_ESDIRK659 << <NumberofBlocksIntegrator, NumberofThreadsIntegrator >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    ComputeFields_RKStageEvaluation(DATA, M_d, H_d, P, *ResetFlag);

    RungeKuttaFinalSolution_ESDIRK659 << <NumberofBlocksIntegrator, NumberofThreadsIntegrator, AllocSize >> >
        (M_d, H_d, DATA);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    return;
}
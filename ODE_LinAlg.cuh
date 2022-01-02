#include <cuda_runtime.h>
#include "DataTypes.cuh"
#include "GlobalDefines.cuh"
#include "Device_Globals_Constants.cuh"
#include "Array_Indexing_Functions.cuh"
#include <device_launch_parameters.h>
#include "Host_Globals.cuh"
#include <helper_cuda.h>
#ifndef ODE_LINALG_CUH
#define ODE_LINALG_CUH

__device__ void inline LLJacobian_DIRKEvaluate_GPU(Jacobian* J, double* Mn1, double* H, double Lamb)
{
    double Dalpha = Lamb;
    double A = -Gamma / (1 + alpha * alpha);
    double B = alpha * A;

    J->J[0][0] = fma(-(B * h_d * Dalpha), (Mn1[1] * H[1] + Mn1[2] * H[2]), 1.0); //dmx/dmx

    J->J[0][1] = fma((-h_d * Dalpha), (A * H[2] + B * (Mn1[0] * H[1] - 2. * Mn1[1] * H[0])), 0.0); //dmx /dmy

    J->J[0][2] = fma((-h_d * Dalpha), (-A * H[1] + B * (Mn1[0] * H[2] - 2. * Mn1[2] * H[0])), 0.0); //dmx /dmz


    J->J[1][0] = fma((-h_d * Dalpha), (-A * H[0] + B * (Mn1[1] * H[0] - 2. * Mn1[0] * H[1])), 0.0); // dmy/dmx

    J->J[1][1] = fma(-(B * h_d * Dalpha), (Mn1[0] * H[0] + Mn1[2] * H[2]), 1.0); //dmy/dmy

    J->J[1][2] = fma((-h_d * Dalpha), (A * H[0] + B * (Mn1[1] * H[2] - 2. * Mn1[2] * H[1])), 0.0); //dmy /dmz


    J->J[2][0] = fma((-h_d * Dalpha), (A * H[1] + B * (Mn1[2] * H[0] - 2. * Mn1[0] * H[2])), 0.0); // dmz / dmx

    J->J[2][1] = fma((-h_d * Dalpha), (-A * H[0] + B * (Mn1[2] * H[1] - 2. * Mn1[1] * H[2])), 0.0); //dmz /dmy
    J->J[2][2] = fma(-(B * h_d * Dalpha), (Mn1[0] * H[0] + Mn1[1] * H[1]), 1.0); //dmz/dmz

    return;
}
__device__ void inline LLJacobian_ESDIRK_GPU(Jacobian* J, double* m, double* H, double Lamb)
{
    //J = I - hlamb * J_LL
    const double hl = -h_d * Lamb;
    double A = -Gamma / (1 + alpha * alpha);
    double B = alpha * A;

    J->J[0][0] = fma(hl, (B * (m[1] * H[1] + m[2] * H[2])),1.0);
    J->J[0][1] = fma(hl, (B * (m[0] * H[1] - 2 * m[1] * H[0]) + A * (H[2])),0.0);
    J->J[0][2] = fma(hl, (B * (m[0] * H[2] - 2 * m[2] * H[0]) + A * (H[1])),0.0);

    J->J[1][0] = fma(hl, (B * (m[1] * H[0] - 2 * m[0] * H[1]) + A * (-H[2])),0.0);
    J->J[1][1] = fma(hl, (B * (m[2] * H[2] + m[0] * H[0])),1.0);
    J->J[1][2] = fma(hl, (B * (m[1] * H[2] - 2 * m[2] * H[1]) + A * (H[0])),0.0);

    J->J[2][0] = fma(hl, (B * (m[2] * H[0] - 2 * m[0] * H[2]) + A * (H[1])),0.0);
    J->J[2][1] = fma(hl, (B * (m[2] * H[1] - 2 * m[1] * H[2]) + A * (-H[0])),0.0);
    J->J[2][2] = fma(hl, (B * (m[0] * H[0] + m[1] * H[1])),1.0);

    return;
}

__device__ void inline LLJacobian_DIRKEvaluate_STT_GPU(Jacobian* J, double* Mn1, double* H, double Lamb,double* Hstt)
{
    const double denominator = 1.0 / (1 + alpha * alpha);
    const double Dalpha = Lamb;
    const double A = -Gamma * denominator;
    const double B = alpha * A;

    const double Prefactor_1 = -(1.0 + SpinTransferTorqueParameters.Xi * alpha) * denominator; // m x m x Hstt
    const double Prefactor_2 = -(SpinTransferTorqueParameters.Xi - alpha) * denominator; // m x Hstt

    J->J[0][0] = fma(-(B * h_d * Dalpha), (Mn1[1] * H[1] + Mn1[2] * H[2]), 1.0); //dmx/dmx

    J->J[0][0] = fma(-(Prefactor_1 * h_d * Dalpha), (Mn1[1] * Hstt[1] + Mn1[2] * Hstt[2]), J->J[0][0]);

    J->J[0][1] = fma((-h_d * Dalpha), (A * H[2] + B * (Mn1[0] * H[1] - 2. * Mn1[1] * H[0])), 0.0); //dmx /dmy

    J->J[0][1] = fma((-h_d * Dalpha), (Prefactor_2 * Hstt[2] + Prefactor_1 * (Mn1[0] * Hstt[1] - 2. * Mn1[1] * Hstt[0])), J->J[0][1]);

    J->J[0][2] = fma((-h_d * Dalpha), (-A * H[1] + B * (Mn1[0] * H[2] - 2. * Mn1[2] * H[0])), 0.0); //dmx /dmz

    J->J[0][2] = fma((-h_d * Dalpha), (-Prefactor_2 * Hstt[1] + Prefactor_1 * (Mn1[0] * Hstt[2] - 2. * Mn1[2] * Hstt[0])), J->J[0][2]);

    J->J[1][0] = fma((-h_d * Dalpha), (-A * H[0] + B * (Mn1[1] * H[0] - 2. * Mn1[0] * H[1])), 0.0); // dmy/dmx

    J->J[1][0] = fma((-h_d * Dalpha), (-Prefactor_2 * Hstt[0] + Prefactor_1 * (Mn1[1] * Hstt[0] - 2. * Mn1[0] * Hstt[1])), J->J[1][0]);

    J->J[1][1] = fma(-(B * h_d * Dalpha), (Mn1[0] * H[0] + Mn1[2] * H[2]), 1.0); //dmy/dmy

    J->J[1][1] = fma(-(Prefactor_1 * h_d * Dalpha), (Mn1[0] * Hstt[0] + Mn1[2] * Hstt[2]), J->J[1][1]);

    J->J[1][2] = fma((-h_d * Dalpha), (A * H[0] + B * (Mn1[1] * H[2] - 2. * Mn1[2] * H[1])), 0.0); //dmy /dmz

    J->J[1][2] = fma((-h_d * Dalpha), (Prefactor_2 * Hstt[0] + Prefactor_1 * (Mn1[1] * Hstt[2] - 2. * Mn1[2] * Hstt[1])), J->J[1][2]);


    J->J[2][0] = fma((-h_d * Dalpha), (A * H[1] + B * (Mn1[2] * H[0] - 2. * Mn1[0] * H[2])), 0.0); // dmz / dmx


    J->J[2][0] = fma((-h_d * Dalpha), (Prefactor_2 * Hstt[1] + Prefactor_1 * (Mn1[2] * Hstt[0] - 2. * Mn1[0] * Hstt[2])), J->J[2][0]);

    J->J[2][1] = fma((-h_d * Dalpha), (-A * H[0] + B * (Mn1[2] * H[1] - 2. * Mn1[1] * H[2])), 0.0); //dmz /dmy

    J->J[2][1] = fma((-h_d * Dalpha), (-Prefactor_2 * Hstt[0] + Prefactor_1 * (Mn1[2] * Hstt[1] - 2. * Mn1[1] * Hstt[2])), J->J[2][1]);

    J->J[2][2] = fma(-(B * h_d * Dalpha), (Mn1[0] * H[0] + Mn1[1] * H[1]), 1.0); //dmz/dmz

    J->J[2][2] = fma(-(Prefactor_1 * h_d * Dalpha), (Mn1[0] * Hstt[0] + Mn1[1] * Hstt[1]), J->J[2][2]);

    return;
}

__device__ void inline LLJacobian_STT_GPU(Jacobian* J, double* m, double* H, double Lamb)
{
    const double P1 = (1.0/(1.0+alpha*alpha))*(1.0 + SpinTransferTorqueParameters.Xi * alpha); // m x m x Hstt
    const double P2 = (1.0 / (1.0 + alpha * alpha)) * (SpinTransferTorqueParameters.Xi - alpha); // m x Hstt

    const double hl = -h_d * Lamb;

    J->J[0][0] = fma(hl,(P1 * (m[1] * H[1] + m[2] * H[2])), J->J[0][0]);
    J->J[0][1] = fma(hl,(P1 * (m[0] * H[1] - 2 * m[1] * H[0]) + P2 * (H[2])), J->J[0][1]);
    J->J[0][2] = fma(hl,(P1 * (m[0] * H[2] - 2 * m[2] * H[0]) + P2 * (H[1])), J->J[0][2]);

    J->J[1][0] = fma(hl,(P1 * (m[1] * H[0] - 2 * m[0] * H[1]) + P2 * (-H[2])), J->J[1][0]);
    J->J[1][1] = fma(hl,(P1 * (m[2] * H[2] + m[0] * H[0])), J->J[1][1]);
    J->J[1][2] = fma(hl,(P1 * (m[1] * H[2] - 2 * m[2] * H[1]) + P2 * (H[0])), J->J[1][2]);

    J->J[2][0] = fma(hl,(P1 * (m[2] * H[0] - 2 * m[0] * H[2]) + P2 * (H[1])), J->J[2][0]);
    J->J[2][1] = fma(hl,(P1 * (m[2] * H[1] - 2 * m[1] * H[2]) + P2 * (-H[0])), J->J[2][1]);
    J->J[2][2] = fma(hl,(P1 * (m[0] * H[0] + m[1] * H[1])), J->J[2][2]);
}
__device__ int inline Crout_LU_Decomposition_with_Pivoting(double* A, int pivot[], int n)
{
    int row, i, j, k, p;
    double* p_k, * p_row, * p_col;
    double max;
    //         For each row and column, k = 0, ..., n-1,

    for (k = 0, p_k = A; k < n; p_k += n, k++) {
        //            find the pivot row
        pivot[k] = k;
        max = fabs(*(p_k + k));
        for (j = k + 1, p_row = p_k + n; j < n; j++, p_row += n) {
            if (max < fabs(*(p_row + k))) {
                max = fabs(*(p_row + k));
                pivot[k] = j;
                p_col = p_row;
            }
        }
        //     and if the pivot row differs from the current row, then
        //     interchange the two rows.
        if (pivot[k] != k)
            for (j = 0; j < n; j++) {
                max = *(p_k + j);
                *(p_k + j) = *(p_col + j);
                *(p_col + j) = max;
            }
        //                and if the matrix is singular, return error
        if (*(p_k + k) == 0.0) return -1;
        //      otherwise find the upper triangular matrix elements for row k.
        for (j = k + 1; j < n; j++) {
            *(p_k + j) /= *(p_k + k);
        }
        //            update remaining matrix
        for (i = k + 1, p_row = p_k + n; i < n; p_row += n, i++)
            for (j = k + 1; j < n; j++)
                *(p_row + j) -= *(p_row + k) * *(p_k + j);
    }
    return 0;
}
__device__ int inline Crout_LU_with_Pivoting_Solve(double* LU, double B[], int pivot[], double x[], int n)
{
    int i, k;
    double* p_k;
    double dum;
    //         Solve the linear equation Lx = B for x, where L is a lower
    //         triangular matrix.
    for (k = 0, p_k = LU; k < n; p_k += n, k++) {
        if (pivot[k] != k) { dum = B[k]; B[k] = B[pivot[k]]; B[pivot[k]] = dum; }
        x[k] = B[k];
        for (i = 0; i < k; i++) x[k] -= x[i] * *(p_k + i);
        x[k] /= *(p_k + k);
    }
    //         Solve the linear equation Ux = y, where y is the solution
    //         obtained above of Lx = B and U is an upper triangular matrix.
    //         The diagonal part of the upper triangular part of the matrix is
    //         assumed to be 1.0.
    for (k = n - 1, p_k = LU + n * (n - 1); k >= 0; k--, p_k -= n) {
        if (pivot[k] != k) { dum = B[k]; B[k] = B[pivot[k]]; B[pivot[k]] = dum; }
        for (i = k + 1; i < n; i++) x[k] -= x[i] * *(p_k + i);
        if (*(p_k + k) == 0.0) return -1;
    }
    return 0;
}
__host__ void SetNewtonTolerance();

__device__ void inline Crout_LU_Decomp(JAC J)
{
    //L11 = a11, L21 = a22, L31 = a33
    //U11 = 1, U22 = 1, U33 = 1
    J->J[0][1] = J->J[0][1] / J->J[0][0]; //U12
    J->J[0][2] = (J->J[0][2]) / J->J[0][0]; //U13
   
    J->J[1][1] = J->J[1][1] - (J->J[1][0]) * (J->J[0][1]); //L22
    J->J[1][2] = (J->J[1][2] - (J->J[1][0]) * (J->J[0][2])) / (J->J[1][1]); //U23

    J->J[2][1] = J->J[2][1] - J->J[2][0]* J->J[0][1]; //L32
    J->J[2][2] = J->J[2][2] - J->J[2][0]*J->J[0][2] - J->J[2][1]*J->J[1][2]; //L33
}
__device__ void inline Crout_LU_Solve(JAC J,double* x,double* b)
{
//Ly = b
    double y[3];

    y[0] = b[0] / (J->J[0][0]);
    y[1] = (b[1] - J->J[1][0] * y[0]) / (J->J[1][1]);
    y[2] = (b[2] - J->J[2][0] * y[0] - J->J[2][1] * y[1]) / (J->J[2][2]);

//Ux = y
    x[2] = y[2];
    x[1] = y[1] - J->J[1][2] * x[2];
    x[0] = y[0] - J->J[0][1] * x[1] - J->J[0][2] * x[2];

}

#endif // !ODE_LINALG_CUH

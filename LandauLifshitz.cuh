#include <cuda_runtime.h>
#include "DataTypes.cuh"
#include "Device_VectorMath.cuh"
#include "Device_Globals_Constants.cuh"
#include "GlobalDefines.cuh"

#ifndef LANDAULIFSHITZ_CUH
#define LANDAULIFSHITZ_CUH


__device__ inline Vector EvaluateLandauLifshitz_GPU(double* M, double* H)
{
    double A = -Gamma / (1.0 + alpha * alpha);
    double B = (alpha * A);
    Vector LL;

    Vector MxH = CrossProduct(M, H);
    Vector MxMxH = CrossProduct(M, &MxH.X[0]);

    LL.X[0] = A * MxH.X[0];
    LL.X[1] = A * MxH.X[1];
    LL.X[2] = A * MxH.X[2];

    LL.X[0] = fma(B, MxMxH.X[0], LL.X[0]);
    LL.X[1] = fma(B, MxMxH.X[1], LL.X[1]);
    LL.X[2] = fma(B, MxMxH.X[2], LL.X[2]);

    return LL;
}
__device__ inline Vector EvaluateLandauLifshitz_STT_GPU(double* M, double* H,double* H_stt,MaterialHandle Handle)
{
    const double denominator = 1.0 / (1.0 + alpha * alpha);
    const double A = -Gamma * denominator;
    const double B = (alpha * A);

    Vector LL;

    Vector MxH = CrossProduct(M, H);
    Vector MxMxH = CrossProduct(M, &MxH.X[0]);

    LL.X[0] = A * MxH.X[0];
    LL.X[1] = A * MxH.X[1];
    LL.X[2] = A * MxH.X[2];

    LL.X[0] = fma(B, MxMxH.X[0], LL.X[0]);
    LL.X[1] = fma(B, MxMxH.X[1], LL.X[1]);
    LL.X[2] = fma(B, MxMxH.X[2], LL.X[2]);

    if (Handle == SpinTransferTorqueParameters.handle)
    {
        const double Prefactor_1 = denominator*(1.0 + SpinTransferTorqueParameters.Xi * alpha);
        const double Prefactor_2 = denominator*(SpinTransferTorqueParameters.Xi - alpha);


        MxH = CrossProduct(M, H_stt);
        MxMxH = CrossProduct(M, &MxH.X[0]);

        LL.X[0] = fma(Prefactor_2, MxH.X[0], LL.X[0]);
        LL.X[1] = fma(Prefactor_2, MxH.X[1], LL.X[1]);
        LL.X[2] = fma(Prefactor_2, MxH.X[2], LL.X[2]);

        LL.X[0] = fma(Prefactor_1, MxMxH.X[0], LL.X[0]);
        LL.X[1] = fma(Prefactor_1, MxMxH.X[1], LL.X[1]);
        LL.X[2] = fma(Prefactor_1, MxMxH.X[2], LL.X[2]);
    }


    return LL;
}
__device__ Vector EvaluateLandauLifshitz_Material_GPU(double* M, double* H, double damping);

#endif // !LANDAULIFSHITZ_CUH

#include "LandauLifshitz.cuh"

__device__ Vector EvaluateLandauLifshitz_Material_GPU(double* M, double* H,double damping)
{
    double A = -Gamma / (1.0 + damping * damping);
    double B = (damping * A);
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
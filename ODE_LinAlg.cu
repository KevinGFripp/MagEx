#include "ODE_LinAlg.cuh"

__host__ void SetNewtonTolerance()
{
    AbsTol_h = (1e-3) * RelTol;
    checkCudaErrors(cudaMemcpyToSymbol(AbsTol, &AbsTol_h, sizeof(double)));
    cudaDeviceSynchronize();
}


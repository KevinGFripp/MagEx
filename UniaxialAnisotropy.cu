#include "UniaxialAnisotropy.cuh"
#include "Device_Globals_Constants.cuh"
#include <device_launch_parameters.h>
#include "Array_Indexing_Functions.cuh"

__global__ void UniaxialAnisotropy(MAG M, FIELD H)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;



    double Hkx = 0.0, Hky = 0.0, Hkz = 0.0;

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        int index = ind(i, j, k);
        if (M->Mat[index] == 0)
        {          
        }
        else
        {
            Hkx = K_UANIS * Uanisx * M->M[mind(0, i, j, k, 0)];
            Hky = K_UANIS * Uanisy * M->M[mind(0, i, j, k, 1)];
            Hkz = K_UANIS * Uanisz * M->M[mind(0, i, j, k, 2)];
            H->H_eff[find(i, j, k, 0)] += Hkx;
            H->H_eff[find(i, j, k, 1)] += Hky;
            H->H_eff[find(i, j, k, 2)] += Hkz;
        }
        H->H_anis[find(i, j, k, 0)] = Hkx;
        H->H_anis[find(i, j, k, 1)] = Hky;
        H->H_anis[find(i, j, k, 2)] = Hkz;
    }
}
__global__ void UniaxialAnisotropy_Mn1(MAG M, FIELD H)
{

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double Hkx=0.0, Hky=0.0, Hkz=0.0;

   

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        int index = ind(i, j, k);

        if (M->Mat[index] == 0)
        {          
        }
        else
        {
            Hkx = K_UANIS * Uanisx * M->M[mind(1, i, j, k, 0)];
            Hky = K_UANIS * Uanisy * M->M[mind(1, i, j, k, 1)];
            Hkz = K_UANIS * Uanisz * M->M[mind(1, i, j, k, 2)];
            H->H_eff[find(i, j, k, 0)] += Hkx;
            H->H_eff[find(i, j, k, 1)] += Hky;
            H->H_eff[find(i, j, k, 2)] += Hkz;
        }
        H->H_anis[find(i, j, k, 0)] = Hkx;
        H->H_anis[find(i, j, k, 1)] = Hky;
        H->H_anis[find(i, j, k, 2)] = Hkz;
    }
}
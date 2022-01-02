#include <cuda_runtime.h>
#include "Device_Globals_Constants.cuh"
#include "Host_Globals.cuh"
#include "GlobalDefines.cuh"
#ifndef ARRAY_INDEXING_FUNCTIONS_CUH
#define ARRAY_INDEXING_FUNCTIONS_CUH


__host__  int FFTind_h(int i, int j, int k);
__host__  int mind_h(int n, int i, int j, int k, int v);
__host__  int find_h(int i, int j, int k, int v);
__host__  int dmind_h(int i, int j, int k, int v);
__host__  int ind_h(int i, int j, int k);
__host__  int bind_h(int i, int j, int k, int v);
__host__ int SignD(int x);
__device__ int Sign(int x);


__device__ inline int FFTind(int i, int j, int k)
{
    return (k + PADNUMZ * (j + (PADNUMY) * i));
}
__device__ inline int FFTind_R2C(int i, int j, int k)
{
    //Outermost dimension of size N3/2 +1
    return (k + (PADNUMZ/2 + 1) * (j + (PADNUMY) * i));
}
__device__ inline int mind(int n, int i, int j, int k, int v)
{
    return ((((n * DIM + v) * NUM + i) * NUMY + j) * NUMZ + k);
}
__device__  inline int find(int i, int j, int k, int v)
{
    return (((v * NUM + i) * NUMY + j) * NUMZ + k);
}
__device__  inline int dmind(int i, int j, int k, int v)
{
    return (((v * NUM + i) * NUMY + j) * NUMZ + k);
}
__device__  inline int ind(int i, int j, int k)
{
    return (k + NUMZ * (j + (NUMY)*i));
}
__device__  inline int dind(int i, int j, int k)
{
    return (k + (NUMZ + 1) * (j + (NUMY + 1) * i));
}
__device__  inline int bind(int i, int j, int k, int v)
{
    return (((v * NUM + i) * NUMY + j) * NUMZ + k);
}

#endif // !ARRAY_INDEXING_FUNCTIONS_CUH

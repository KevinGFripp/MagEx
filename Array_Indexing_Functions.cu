#include "Array_Indexing_Functions.cuh"


__host__  int mind_h(int n, int i, int j, int k, int v)
{
    return ((((n * DIM + v) * NUM_h + i) * NUMY_h + j) * NUMZ_h + k);
}
__host__  int find_h(int i, int j, int k, int v)
{
    return (((v * NUM_h + i) * NUMY_h + j) * NUMZ_h + k);
}
__host__  int dmind_h(int i, int j, int k, int v)
{
    return (((v * NUM_h + i) * NUMY_h + j) * NUMZ_h + k);
}
__host__  int ind_h(int i, int j, int k)
{
    return (k + NUMZ_h * (j + (NUMY_h)*i));
}
__host__  int bind_h(int i, int j, int k, int v)
{
    return (((v * NUM_h + i) * NUMY_h + j) * NUMZ_h + k);
}
__host__ int SignD(int x)
{
    if (x == 0)
    {
        return -1;
    }
    return ((x > 0) - (x < 0));
}
__device__ int Sign(int x)
{
    if (x == 0)
    {
        return -1;
    }
    return ((x > 0) - (x < 0));
}
__host__ int FFTind_h(int i, int j, int k)
{
    return (k + 2 *NUMZ_h * (j + ((2 - PBC_y_h) * NUMY_h) * i));
}
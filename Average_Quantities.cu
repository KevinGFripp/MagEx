#include "Average_Quantities.cuh"
#include <device_launch_parameters.h>
#include "GlobalDefines.cuh"
#include "Device_Globals_Constants.cuh"
#include "Host_Globals.cuh"
#include "Array_Indexing_Functions.cuh"
#include "Reduction_Functions.cuh"
#include <helper_cuda.h>

__host__ double ExchangeEnergy(MAG M, FIELD H, MEMDATA DATA)
{
    double V_cell = CELL_h * CELLY_h * CELLZ_h * 1e-27;
    double Scale = (mu)*V_cell * MSAT_h;

    Vector Result = AverageExchange_Reduction();
    return Scale * (Result.X[0] + Result.X[1] + Result.X[2]);
}
__host__ double ZeemanEnergy(MAG M, FIELD H, MEMDATA DATA)
{
    double V_cell = CELL_h * CELLY_h * CELLZ_h *(1e-27);
    double Scale = -mu * MSAT_h * V_cell;


    Vector Result = AverageZeeman_Reduction();
    return Scale * (Result.X[0] + Result.X[1] + Result.X[2]);
}
__host__ double UniAnisotropyEnergy(MAG M, FIELD H, MEMDATA DATA)
{
    double V_cell = CELL_h * CELLY_h * CELLZ_h * 1e-27;
    double Scale = mu * V_cell * MSAT_h;
    double A = -Scale;

    Vector Result = AverageUniAnisotropy_Reduction();
    return A * (Result.X[0] + Result.X[1] + Result.X[2]);
}
__host__ double DemagEnergy(MAG M, FIELD H, MEMDATA DATA)
{
    double V_cell = CELL_h * CELLY_h * CELLZ_h * (1e-27);
    double Scale = 0.5*(mu) * V_cell * (MSAT_h);
   
    Vector Result = AverageDemag_Reduction();
    return Scale * (Result.X[0] + Result.X[1] + Result.X[2]);
}

__host__ Vector AverageDemag_Reduction()
{
    Vector Results;
    int maxThreads = 256;  // number of threads per block
    int whichKernel = 6;
    int maxBlocks = 64;

    int numBlocks = 0;
    int numThreads = 0;

    const int N = NUM_h * NUMY_h * NUMZ_h;
    getNumBlocksAndThreads(whichKernel,N, maxBlocks, maxThreads, numBlocks,
        numThreads);

    double* input_m = &(DEVICE_PTR_STRUCT.M)->M[mind_h(0, 0, 0, 0, 0)];
    double* input_h = &(DEVICE_PTR_STRUCT.H)->H_m[find_h(0, 0, 0, 0)];
    double* output_d = (DEVICE_PTR_STRUCT.DATA)->xReduction;

    Results.X[0] = -1.0*Reduce_Energy(N, numThreads, numBlocks, maxThreads, maxBlocks,
        input_m, input_h, output_d);

    input_m = &(DEVICE_PTR_STRUCT.M)->M[mind_h(0, 0, 0, 0, 1)];
    input_h = &(DEVICE_PTR_STRUCT.H)->H_m[find_h(0, 0, 0, 1)];
    output_d = (DEVICE_PTR_STRUCT.DATA)->yReduction;

    Results.X[1] = -1.0*Reduce_Energy(N, numThreads, numBlocks, maxThreads, maxBlocks,
        input_m, input_h, output_d);

    input_m = &(DEVICE_PTR_STRUCT.M)->M[mind_h(0, 0, 0, 0, 2)];
    input_h = &(DEVICE_PTR_STRUCT.H)->H_m[find_h(0, 0, 0, 2)];
    output_d = (DEVICE_PTR_STRUCT.DATA)->zReduction;

    Results.X[2] = -1.0*Reduce_Energy(N, numThreads, numBlocks, maxThreads, maxBlocks,
        input_m, input_h, output_d);

   
    return Results;
}
__host__ Vector AverageExchange_Reduction()
{
    Vector Results;
    int maxThreads = 256;  // number of threads per block
    int whichKernel = 6;
    int maxBlocks = 64;

    int numBlocks = 0;
    int numThreads = 0;

    const int N = NUM_h * NUMY_h * NUMZ_h;
    getNumBlocksAndThreads(whichKernel, N, maxBlocks, maxThreads, numBlocks,
        numThreads);

    double* input_m = &(DEVICE_PTR_STRUCT.M)->M[mind_h(0, 0, 0, 0, 0)];
    double* input_h = &(DEVICE_PTR_STRUCT.H)->H_ex[find_h(0, 0, 0, 0)];
    double* output_d = (DEVICE_PTR_STRUCT.DATA)->xReduction;

    Results.X[0] = Reduce_Energy(N, numThreads, numBlocks, maxThreads, maxBlocks,
        input_m, input_h, output_d);

    input_m = &(DEVICE_PTR_STRUCT.M)->M[mind_h(0, 0, 0, 0, 1)];
    input_h = &(DEVICE_PTR_STRUCT.H)->H_ex[find_h(0, 0, 0, 1)];
    output_d = (DEVICE_PTR_STRUCT.DATA)->yReduction;

    Results.X[1] = Reduce_Energy(N, numThreads, numBlocks, maxThreads, maxBlocks,
        input_m, input_h, output_d);

    input_m = &(DEVICE_PTR_STRUCT.M)->M[mind_h(0, 0, 0, 0, 2)];
    input_h = &(DEVICE_PTR_STRUCT.H)->H_ex[find_h(0, 0, 0, 2)];
    output_d = (DEVICE_PTR_STRUCT.DATA)->zReduction;

    Results.X[2] = Reduce_Energy(N, numThreads, numBlocks, maxThreads, maxBlocks,
        input_m, input_h, output_d);


    return Results;
}
__host__ Vector AverageZeeman_Reduction()
{
    Vector Results;
    int maxThreads = 256;  // number of threads per block
    int whichKernel = 6;
    int maxBlocks = 64;

    int numBlocks = 0;
    int numThreads = 0;

    const int N = NUM_h * NUMY_h * NUMZ_h;
    getNumBlocksAndThreads(whichKernel, N, maxBlocks, maxThreads, numBlocks,
        numThreads);

    double* input_m = &(DEVICE_PTR_STRUCT.M)->M[mind_h(0, 0, 0, 0, 0)];
    double* input_h = &(DEVICE_PTR_STRUCT.H)->H_ext[find_h(0, 0, 0, 0)];
    double* output_d = (DEVICE_PTR_STRUCT.DATA)->xReduction;

    Results.X[0] = Reduce_Energy(N, numThreads, numBlocks, maxThreads, maxBlocks,
        input_m, input_h, output_d);

    input_m = &(DEVICE_PTR_STRUCT.M)->M[mind_h(0, 0, 0, 0, 1)];
    input_h = &(DEVICE_PTR_STRUCT.H)->H_ext[find_h(0, 0, 0, 1)];
    output_d = (DEVICE_PTR_STRUCT.DATA)->yReduction;

    Results.X[1] = Reduce_Energy(N, numThreads, numBlocks, maxThreads, maxBlocks,
        input_m, input_h, output_d);

    input_m = &(DEVICE_PTR_STRUCT.M)->M[mind_h(0, 0, 0, 0, 2)];
    input_h = &(DEVICE_PTR_STRUCT.H)->H_ext[find_h(0, 0, 0, 2)];
    output_d = (DEVICE_PTR_STRUCT.DATA)->zReduction;

    Results.X[2] = Reduce_Energy(N, numThreads, numBlocks, maxThreads, maxBlocks,
        input_m, input_h, output_d);


    return Results;
}
__host__ Vector AverageUniAnisotropy_Reduction()
{
    Vector Results;
    int maxThreads = 256;  // number of threads per block
    int whichKernel = 6;
    int maxBlocks = 64;

    int numBlocks = 0;
    int numThreads = 0;

    const int N = NUM_h * NUMY_h * NUMZ_h;
    getNumBlocksAndThreads(whichKernel, N, maxBlocks, maxThreads, numBlocks,
        numThreads);

    double* input_m = &(DEVICE_PTR_STRUCT.M)->M[mind_h(0, 0, 0, 0, 0)];
    double* input_h = &(DEVICE_PTR_STRUCT.H)->H_anis[find_h(0, 0, 0, 0)];
    double* output_d = (DEVICE_PTR_STRUCT.DATA)->xReduction;

    Results.X[0] = Reduce_Energy(N, numThreads, numBlocks, maxThreads, maxBlocks,
        input_m, input_h, output_d);

    input_m = &(DEVICE_PTR_STRUCT.M)->M[mind_h(0, 0, 0, 0, 1)];
    input_h = &(DEVICE_PTR_STRUCT.H)->H_anis[find_h(0, 0, 0, 1)];
    output_d = (DEVICE_PTR_STRUCT.DATA)->yReduction;

    Results.X[1] = Reduce_Energy(N, numThreads, numBlocks, maxThreads, maxBlocks,
        input_m, input_h, output_d);

    input_m = &(DEVICE_PTR_STRUCT.M)->M[mind_h(0, 0, 0, 0, 2)];
    input_h = &(DEVICE_PTR_STRUCT.H)->H_anis[find_h(0, 0, 0, 2)];
    output_d = (DEVICE_PTR_STRUCT.DATA)->zReduction;

    Results.X[2] = Reduce_Energy(N, numThreads, numBlocks, maxThreads, maxBlocks,
        input_m, input_h, output_d);


    return Results;
}
__host__ Vector AverageMag_Reduction()
{
    Vector Results;
    int maxThreads = 256;  // number of threads per block
    int whichKernel = 6;
    int maxBlocks = 64;

    int numBlocks = 0;
    int numThreads = 0;

    const int N = NUM_h * NUMY_h * NUMZ_h;
    getNumBlocksAndThreads(whichKernel, N, maxBlocks, maxThreads, numBlocks,
        numThreads);

    double* input_d = &(DEVICE_PTR_STRUCT.M)->M[mind_h(0, 0, 0, 0, 0)];
    double* output_d = (DEVICE_PTR_STRUCT.DATA)->xReduction;

    Results.X[0] = Reduce_Sum(N, numThreads, numBlocks, maxThreads, maxBlocks, input_d, output_d);

    input_d = &(DEVICE_PTR_STRUCT.M)->M[mind_h(0, 0, 0, 0, 1)];
    output_d = (DEVICE_PTR_STRUCT.DATA)->yReduction;

    Results.X[1] = Reduce_Sum(N, numThreads, numBlocks, maxThreads, maxBlocks, input_d, output_d);

    input_d = &(DEVICE_PTR_STRUCT.M)->M[mind_h(0, 0, 0, 0, 2)];
    output_d = (DEVICE_PTR_STRUCT.DATA)->zReduction;

    Results.X[2] = Reduce_Sum(N, numThreads, numBlocks, maxThreads, maxBlocks, input_d, output_d);

    return Results;
}


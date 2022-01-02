#include "MemoryAllocation.cuh"
#include "Reduction_Functions.cuh"
#include "Data_Transfer_Functions.cuh"
#include "Device_FFT_Functions.cuh"
#include "Simulation_Parameter_Wrapper_Functions.cuh"
#include "ODE_LinAlg.cuh"
#include "ThreadsAndBlocks_Functions.cuh"
#include "ExchangeField.cuh"
#include "DemagnetisingTensor_Functions.cuh"
#include "DefineMaterials_Functions.cuh"

__host__ void EstimateDeviceMemoryRequirements()
{
    int Memsize_Reductions = 2 * 2048 * sizeof(double);

    int Memsize_N = 6 * 8 * sizeof(fftw_complex) * NUM_h * NUMY_h * NUMZ_h;

    //int Memsize_N =  6 * 8*sizeof(double) * NUM_h * NUMY_h * NUMZ_h;

    int Memsize_FFT = 24 * sizeof(fftw_complex) * NUM_h * NUMY_h * NUMZ_h;

    int Memsize_Mag = (DIM * DIM * NUM_h * NUMY_h * NUMZ_h * sizeof(double))
        + DIM * NUM_h * NUMY_h * NUMZ_h * sizeof(int)
        + NUM_h * NUMY_h * NUMZ_h * sizeof(int);

    int Memsize_Fields = 5 * DIM * NUM_h * NUMY_h * NUMZ_h * sizeof(double);

    double Total = (double)(Memsize_N + Memsize_FFT + Memsize_Mag + Memsize_Fields + Memsize_Reductions) / (1024.0 * 1024.0 * 1024.0);

   // printf("Device Memory required = %lf Gb \n", Total * 9.765);

    return;
}
__host__ void GlobalInitialise(MEMDATA* DATA_h, MAG* M_h, FIELD* H_h, PLANS* P_h,
    MEMDATA* DATA_temp, MAG* M_temp, FIELD* H_temp,
    MEMDATA* DATA_d, MAG* M_d, FIELD* H_d)
{
    MemInitialise_d(DATA_temp, M_temp, H_temp, DATA_d, M_d, H_d);
    MemInitialise_h(DATA_h, M_h, H_h, P_h);
    AllocateSharedMemorySize();

    ReductionArraysInit(*DATA_h);

    CopyMemDataToDevice(*DATA_h, *DATA_d);

    if (UseSinglePrecision == false)
    {
        FFTPlansInitialise(*P_h);
    }
    else 
    {
        FFTPlansInitialise_SinglePrecision_R2C(*P_h);
    }

    CopyFieldToDevice(*H_h, *H_d);
    IncludeBiasField(false);
    IncludeExternalField(false, NULL);
    IncludeUniaxialAnisotropy(false);


    DemagTensorFFT_Symmetries(*DATA_h, *DATA_d);

    free((*(DATA_h))->kxx);
    free((*(DATA_h))->kxy);
    free((*(DATA_h))->kxz);
    free((*(DATA_h))->kyz);
    free((*(DATA_h))->kyy);
    free((*(DATA_h))->kzz);

    if (UseSinglePrecision == true)
    {
        cudaFree((*DATA_temp)->xFFT);
        cudaFree((*DATA_temp)->yFFT);
        cudaFree((*DATA_temp)->zFFT);
    }

    SetNewtonTolerance();

    checkCudaErrors(cudaDeviceSynchronize());
}
__host__  void MemInitialise_d(MEMDATA* DATA_temp, MAG* M_temp, FIELD* H_temp,
    MEMDATA* DATA_d, MAG* M_d, FIELD* H_d)
{

    int GRIDSIZE = ((2 - PBC_x_h) * NUM_h * (2 - PBC_y_h) * NUMY_h * 2 * (NUMZ_h));

    int GRIDSIZE_M = (NUM_h * NUMY_h * NUMZ_h);

    PADNUM_h = (2 - PBC_x_h) * NUM_h;
    PADNUMY_h = (2 - PBC_y_h) * NUMY_h;
    PADNUMZ_h = 2 * (NUMZ_h);

    ////don't over-pad data
    //if (NUMZ_h <= 4)
    //{
    //    PADNUMZ_h = 2*NUMZ_h - 1;
    //}

    checkCudaErrors(cudaMemcpyToSymbol(PADNUMZ, &PADNUMZ_h, sizeof(int)));
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMalloc((void**)&((*M_temp)->M), sizeof(double) * 12 * DIM * GRIDSIZE_M));
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaMalloc((void**)&((*M_temp)->Mat), sizeof(int) * GRIDSIZE_M));

    checkCudaErrors(cudaMalloc((void**)&((*M_temp)->Bd), sizeof(int) * DIM * GRIDSIZE_M));

    checkCudaErrors(cudaMalloc((void**)&((*M_temp)->NUMCELLS), sizeof(int)));


    //Implicit data
    checkCudaErrors(cudaMalloc((void**)&((*M_temp)->J), sizeof(jacobian) * GRIDSIZE_M));
    checkCudaErrors(cudaMalloc((void**)&((*M_temp)->Pivot), DIM * sizeof(int) * GRIDSIZE_M));

    int GRIDSIZE_D = ((NUM_h + 1) * (NUMY_h + 1) * (NUMZ_h + 1));


    checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->Nxx), GRIDSIZE_D * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->Nxy), GRIDSIZE_D * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->Nxz), GRIDSIZE_D * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->Nyy), GRIDSIZE_D * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->Nyz), GRIDSIZE_D * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->Nzz), GRIDSIZE_D * sizeof(double)));

   
    checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->xFFT), GRIDSIZE * sizeof(fftw_complex)));
    checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->yFFT), GRIDSIZE * sizeof(fftw_complex)));
    checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->zFFT), GRIDSIZE * sizeof(fftw_complex)));
    
 
    if (UseSinglePrecision == true)
    {
        checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->xFFT_s), GRIDSIZE * 2 * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->yFFT_s), GRIDSIZE * 2 * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->zFFT_s), GRIDSIZE * 2 * sizeof(float)));

        checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->Outx), GRIDSIZE * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->Outy), GRIDSIZE * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->Outz), GRIDSIZE * sizeof(float)));
    }
    else {
        checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->Outx_d), GRIDSIZE * sizeof(double)));
        checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->Outy_d), GRIDSIZE * sizeof(double)));
        checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->Outz_d), GRIDSIZE * sizeof(double)));
    }



    checkCudaErrors(cudaMalloc((void**)&((*H_temp)->H_ex), sizeof(double) * NUM_h * NUMY_h * NUMZ_h * DIM));
    checkCudaErrors(cudaMalloc((void**)&((*H_temp)->H_m), sizeof(double) * NUM_h * NUMY_h * NUMZ_h * DIM));

    checkCudaErrors(cudaMalloc((void**)&((*H_temp)->H_anis), sizeof(double) * NUM_h * NUMY_h * NUMZ_h * DIM));
    checkCudaErrors(cudaMalloc((void**)&((*H_temp)->H_ext), sizeof(double) * NUM_h * NUMY_h * NUMZ_h * DIM));

    checkCudaErrors(cudaMalloc((void**)&((*H_temp)->H_eff), sizeof(double) * NUM_h * NUMY_h * NUMZ_h * DIM));

    checkCudaErrors(cudaMalloc((void**)&((*H_temp)->H_stage), sizeof(double) * NUM_h * NUMY_h * NUMZ_h * DIM));
    checkCudaErrors(cudaMalloc((void**)&((*H_temp)->H_stage_1), sizeof(double) * NUM_h * NUMY_h * NUMZ_h * DIM));

  
    checkCudaErrors(cudaMalloc((void**)&((*H_temp)->H_STT), sizeof(double) * NUM_h * NUMY_h * NUMZ_h * DIM));
    

    //Threads and Blocks
    Fields_GetThreadsAndBlocks();
    Integration_GetThreadsAndBlocks();

    //Reduction Arrays
    CalculateThreadsAndBlocksReductions_1D();

    int REDUCTIONSIZE = (NumberofBlocksReduction.x * NumberofBlocksReduction.y * NumberofBlocksReduction.z);
    checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->xReduction), REDUCTIONSIZE * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->yReduction), REDUCTIONSIZE * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->zReduction), REDUCTIONSIZE * sizeof(double)));

    printf("---------CUDA configuration-------------\n");
    printf("Field : Blocks : %d x %d x %d \n", NumberofBlocks.x, NumberofBlocks.y, NumberofBlocks.z);
    printf("FIeld : Threads : %d x %d x %d \n", NumberofThreads.x, NumberofThreads.y, NumberofThreads.z);

    printf("Integration : Blocks : %d x %d x %d \n", NumberofBlocksIntegrator.x,
        NumberofBlocksIntegrator.y, NumberofBlocksIntegrator.z);
    printf("Integration : Threads : %d x %d x %d \n", NumberofThreadsIntegrator.x,
        NumberofThreadsIntegrator.y, NumberofThreadsIntegrator.z);

    printf("FFTs : Blocks : %d x %d x %d \n", NumberofBlocksPadded.x, NumberofBlocksPadded.y,
        NumberofBlocksPadded.z);
    printf("FFTs : Threads : %d x %d x %d \n", NumberofThreadsPadded.x,
        NumberofThreadsPadded.y, NumberofThreadsPadded.z);
    printf("----------------------------------------\n\n");

  

    checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->dE_Reduction),
        (NumberofBlocksIntegrator.x * NumberofBlocksIntegrator.y * NumberofBlocksIntegrator.z * sizeof(double))));

    checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->MaxTorqueReduction),
        (NumberofBlocksIntegrator.x * NumberofBlocksIntegrator.y * NumberofBlocksIntegrator.z * sizeof(double))));

    checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->StepReduction),
        (NumberofBlocksIntegrator.x * NumberofBlocksIntegrator.y * NumberofBlocksIntegrator.z * sizeof(double))));

    checkCudaErrors(cudaMalloc((void**)&((*DATA_temp)->NewtonStepsReduction),
        (NumberofBlocksIntegrator.x * NumberofBlocksIntegrator.y * NumberofBlocksIntegrator.z * sizeof(int))));

    //Copy the temp host struct with cudamalloc'd device pointers to the device structs
    checkCudaErrors(cudaMemcpy((*DATA_d), (*DATA_temp), sizeof(dataptr), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((*M_d), (*M_temp), sizeof(Magnetisation), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((*H_d), (*H_temp), sizeof(field), cudaMemcpyHostToDevice));

    CopyDevicePointers(*DATA_temp, *M_temp, *H_temp);

    return;
}
__host__  void MemInitialise_h(MEMDATA* DATA, MAG* M, FIELD* H, PLANS* P)
{
    
    printf("Initialising host memory.\n");

    *M = (Magnetisation*)malloc(sizeof(Magnetisation));
    int GRIDSIZE = ((2 - PBC_x_h) * NUM_h * (2 - PBC_y_h) * NUMY_h * 2 * (NUMZ_h));

    int GRIDSIZE_M = (NUM_h * NUMY_h * NUMZ_h);

    int GRIDSIZE_D = ((NUM_h + 1) * (NUMY_h + 1) * (NUMZ_h + 1));


    (*M)->M = (double*)malloc(sizeof(double) * 12 * DIM * GRIDSIZE_M);

    (*M)->Mat = (int*)malloc(sizeof(int) * GRIDSIZE_M);

    (*M)->Bd = (int*)malloc(sizeof(int) * DIM * GRIDSIZE_M);

    (*M)->NUMCELLS = (int*)malloc(sizeof(int));

    cudaHostRegister((*M)->M, sizeof(double) * 12 * DIM * GRIDSIZE_M, 0);
    
    PointerCheck((*M)->M != NULL);
    PointerCheck((*M)->Mat != NULL);
    PointerCheck((*M)->Bd != NULL);
    PointerCheck((*M)->NUMCELLS != NULL);

    (*DATA)->kxx = (fftw_complex*)malloc(GRIDSIZE * sizeof(fftw_complex));
    (*DATA)->kxy = (fftw_complex*)malloc(GRIDSIZE * sizeof(fftw_complex));
    (*DATA)->kxz = (fftw_complex*)malloc(GRIDSIZE * sizeof(fftw_complex));
    (*DATA)->kyz = (fftw_complex*)malloc(GRIDSIZE * sizeof(fftw_complex));
    (*DATA)->kyy = (fftw_complex*)malloc(GRIDSIZE * sizeof(fftw_complex));
    (*DATA)->kzz = (fftw_complex*)malloc(GRIDSIZE * sizeof(fftw_complex));

    PointerCheck((*DATA)->kxx != NULL);
    PointerCheck((*DATA)->kxy != NULL);
    PointerCheck((*DATA)->kxz != NULL);
    PointerCheck((*DATA)->kyy != NULL);
    PointerCheck((*DATA)->kyz != NULL);
    PointerCheck((*DATA)->kzz != NULL);


    (*DATA)->xFFT = (fftw_complex*)malloc(GRIDSIZE * sizeof(fftw_complex));
    (*DATA)->yFFT = (fftw_complex*)malloc(GRIDSIZE * sizeof(fftw_complex));
    (*DATA)->zFFT = (fftw_complex*)malloc(GRIDSIZE * sizeof(fftw_complex));

    if (UseSinglePrecision == true)
    {
        (*DATA)->xFFT_s = (cufftComplex*)malloc(GRIDSIZE * sizeof(cufftComplex));
        (*DATA)->yFFT_s = (cufftComplex*)malloc(GRIDSIZE * sizeof(cufftComplex));
        (*DATA)->zFFT_s = (cufftComplex*)malloc(GRIDSIZE * sizeof(cufftComplex));

        (*DATA)->Outx = (float*)malloc(GRIDSIZE * sizeof(float));
        (*DATA)->Outy = (float*)malloc(GRIDSIZE * sizeof(float));
        (*DATA)->Outz = (float*)malloc(GRIDSIZE * sizeof(float));

        PointerCheck((*DATA)->xFFT_s != NULL);
        PointerCheck((*DATA)->yFFT_s != NULL);
        PointerCheck((*DATA)->zFFT_s != NULL);

        PointerCheck((*DATA)->Outx != NULL);
        PointerCheck((*DATA)->Outy != NULL);
        PointerCheck((*DATA)->Outz != NULL);
    }

    PointerCheck((*DATA)->xFFT != NULL);
    PointerCheck((*DATA)->yFFT != NULL);
    PointerCheck((*DATA)->zFFT != NULL);

    (*H)->H_ex = (double*)malloc(sizeof(double) * GRIDSIZE_M * DIM);
    (*H)->H_m = (double*)malloc(sizeof(double) * GRIDSIZE_M * DIM);
    (*H)->H_eff = (double*)malloc(sizeof(double) * GRIDSIZE_M * DIM);

    PointerCheck((*H)->H_ex != NULL),
    PointerCheck((*H)->H_m != NULL),
    PointerCheck((*H)->H_eff != NULL);
  

    cudaHostRegister((*H)->H_ex, sizeof(double) * GRIDSIZE_M * DIM, 0);
    cudaHostRegister((*H)->H_m, sizeof(double) * GRIDSIZE_M * DIM, 0);
    cudaHostRegister((*H)->H_eff, sizeof(double) * GRIDSIZE_M * DIM, 0);


    int REDUCTIONSIZE = (NumberofBlocksReduction.x * NumberofBlocksReduction.y * NumberofBlocksReduction.z);
    (*DATA)->xReduction = (double*)malloc(REDUCTIONSIZE * sizeof(double));
    (*DATA)->yReduction = (double*)malloc(REDUCTIONSIZE * sizeof(double));
    (*DATA)->zReduction = (double*)malloc(REDUCTIONSIZE * sizeof(double));

    REDUCTIONSIZE = NumberofBlocksIntegrator.x * NumberofBlocksIntegrator.y * NumberofBlocksIntegrator.z;
    (*DATA)->StepReduction = (double*)malloc(REDUCTIONSIZE * sizeof(double));
    (*DATA)->MaxTorqueReduction = (double*)malloc(REDUCTIONSIZE * sizeof(double));
    (*DATA)->dE_Reduction = (double*)malloc(REDUCTIONSIZE * sizeof(double));
    (*DATA)->NewtonStepsReduction = (int*)malloc(REDUCTIONSIZE * sizeof(int));

    PointerCheck((*DATA)->StepReduction != NULL);
    PointerCheck((*DATA)->MaxTorqueReduction != NULL);
    PointerCheck((*DATA)->NewtonStepsReduction != NULL);
    PointerCheck((*DATA)->dE_Reduction != NULL);

    MagnetisationInitialise(*M);

    DemagFieldInitialise(*H);

    IsPBCEnabled ? ComputeDemagTensorNewell3D_PBC(*M, *DATA) : ComputeDemagTensorNewell_GQ(*M, *DATA);
}
__host__ void AllocateSharedMemorySize()
{
    int ConvShSize = 7 * (NumberofThreadsPadded.x * NumberofThreadsPadded.y
        * NumberofThreadsPadded.z) * sizeof(fftw_complex);

    SHAREDMEMSIZE_h = (DIM * ((NumberofThreads.x) * (NumberofThreads.y) * (NumberofThreads.z)) * sizeof(double));

    int MSIZE = (NumberofThreadsIntegrator.x * NumberofThreadsIntegrator.y * NumberofThreadsIntegrator.z);

    int ReductionSize = sizeof(double) * (NumberofBlocksIntegrator.x *
        NumberofBlocksIntegrator.y * NumberofBlocksIntegrator.z);

    int IntegratorSharedMemSize = 3 * (MSIZE * DIM * sizeof(double)) + MSIZE * sizeof(int);

  /*  if (SHAREDMEMSIZE_h <= 1024)
    {
        printf("Field Shared memory size required =%d Bytes \n ", 3 * SHAREDMEMSIZE_h);
    }
    else
    {
        printf("Field Shared memory size required =%d Kb \n ", (int)floor((double)3 * SHAREDMEMSIZE_h / 1024.0));
    }

    if (IntegratorSharedMemSize <= 1024)
    {
        printf("Integration Shared memory size required =%d Bytes \n ", IntegratorSharedMemSize);
    }
    else
    {
        printf("Integration Shared memory size required =%d Kb \n ", (int)floor((double)IntegratorSharedMemSize / 1024.0));
    }

    if (ReductionSize <= 1024)
    {
        printf("Integration Reduction Shared memory size required =%d Bytes \n ", ReductionSize);
    }
    else
    {
        printf("Integration Reduction memory size required =%d Kb \n ", (int)floor((double)ReductionSize / 1024.0));
    }
    if (ConvShSize <= 1024)
    {
        printf("Convolution Shared memory size required =%d Bytes \n ", ConvShSize);
    }
    else
    {
        printf("Convolution Shared memory size required =%d Kb \n ", (int)floor((double)ConvShSize / 1024.0));
    }*/


}

__host__ void MemoryClear(MEMDATA* DATA, MAG* M, FIELD* H, PLANS* P,
    MEMDATA* DATA_d, MAG* M_d, FIELD* H_d)
{
    cudaHostUnregister((*M)->M);
    free((*M)->M);
    free((*M)->J);
    free((*M)->Bd);
    free((*M)->Mat);
    free((*M)->Pivot);
    free((*M)->NUMCELLS);

    cudaHostUnregister((*H)->H_ex);
    cudaHostUnregister((*H)->H_m);
    cudaHostUnregister((*H)->H_eff);
    free((*H)->H_eff);
    free((*H)->H_ex);
    free((*H)->H_m);

    free((*DATA)->xReduction);
    free((*DATA)->yReduction);
    free((*DATA)->zReduction);
    free((*DATA)->dE_Reduction);
    free((*DATA)->MaxTorqueReduction);
    free((*DATA)->NewtonStepsReduction);
    free((*DATA)->StepReduction);

    free((*DATA)->xFFT);
    free((*DATA)->yFFT);
    free((*DATA)->zFFT);

    if (UseSinglePrecision == true)
    {
        free((*DATA)->xFFT_s);
        free((*DATA)->yFFT_s);
        free((*DATA)->zFFT_s);

        free((*DATA)->Outx);
        free((*DATA)->Outy);
        free((*DATA)->Outz);
    }

    cufftDestroy((*P)->HxPlan);
    cufftDestroy((*P)->HyPlan);
    cufftDestroy((*P)->HzPlan);

    cufftDestroy((*P)->MxPlan);
    cufftDestroy((*P)->MyPlan);
    cufftDestroy((*P)->MzPlan);

    cudaFree((*M_d)->M);
    cudaFree((*M_d)->J);
    cudaFree((*M_d)->Bd);
    cudaFree((*M_d)->Mat);
    cudaFree((*M_d)->Pivot);
    cudaFree((*M_d)->NUMCELLS);

    cudaFree((*H_d)->H_eff);
    cudaFree((*H_d)->H_ex);
    cudaFree((*H_d)->H_m);
    cudaFree((*H_d)->H_stage);
    cudaFree((*H_d)->H_stage_1);

    cudaFree((*DATA_d)->xFFT);
    cudaFree((*DATA_d)->yFFT);
    cudaFree((*DATA_d)->zFFT);

    cudaFree((*DATA_d)->xFFT_s);
    cudaFree((*DATA_d)->yFFT_s);
    cudaFree((*DATA_d)->zFFT_s);

    cudaFree((*DATA_d)->Nxx);
    cudaFree((*DATA_d)->Nxy);
    cudaFree((*DATA_d)->Nxz);

    cudaFree((*DATA_d)->Nyy);
    cudaFree((*DATA_d)->Nyz);
    cudaFree((*DATA_d)->Nzz);

    cudaFree((*DATA_d)->Outx);
    cudaFree((*DATA_d)->Outy);
    cudaFree((*DATA_d)->Outz);

    cudaFree((*DATA_d)->xReduction);
    cudaFree((*DATA_d)->yReduction);
    cudaFree((*DATA_d)->zReduction);
    cudaFree((*DATA_d)->dE_Reduction);
    cudaFree((*DATA_d)->MaxTorqueReduction);
    cudaFree((*DATA_d)->NewtonStepsReduction);
    cudaFree((*DATA_d)->StepReduction);
    
}
#include "EffectiveField.cuh"
#include <device_launch_parameters.h>
#include "GlobalDefines.cuh"
#include "Array_Indexing_Functions.cuh"
#include "Device_Globals_Constants.cuh"
#include "ExternalFields.cuh"
#include "Host_Globals.cuh"
#include "ExchangeField.cuh"
#include "DemagnetisingField.cuh"
#include <helper_cuda.h>
#include "Device_FFT_Functions.cuh"
#include "UniaxialAnisotropy.cuh"
#include "ZhangLi_STT.cuh"

__global__ void ComputeEffectiveField(MEMDATA DATA, FIELD H,MAG M)
{
    double Hm[3], Hex[3], Heff[3], Hext[3];

    //local index
    int I = threadIdx.z;
    int J = threadIdx.y;
    int K = threadIdx.x;

    //global memory index
    int i = I + blockIdx.z * blockDim.z;
    int j = J + blockIdx.y * blockDim.y;
    int k = K + blockIdx.x * blockDim.x;

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        Hext[0] = 0.0;
        Hext[1] = 0.0;
        Hext[2] = 0.0;

        Hm[0] = DATA->Outx_d[FFTind(i, j, k)];
        Hm[1] = DATA->Outy_d[FFTind(i, j, k)];
        Hm[2] = DATA->Outz_d[FFTind(i, j, k)];

        Hex[0] = H->H_ex[find(i, j, k, 0)];
        Hex[1] = H->H_ex[find(i, j, k, 1)];
        Hex[2] = H->H_ex[find(i, j, k, 2)];

        Heff[0] = Hm[0] + Hex[0];
        Heff[1] = Hm[1] + Hex[1];
        Heff[2] = Hm[2] + Hex[2];


        if (ExternalField == 1)
        {
            Vector extfield = ExcitationFunc_CW(t_d, CELL * i, CELLY * j, CELLZ * k);
            Heff[0] += extfield.X[0];
            Heff[1] += extfield.X[1];
            Heff[2] += extfield.X[2];

            Hext[0] = extfield.X[0];
            Hext[1] = extfield.X[1];
            Hext[2] = extfield.X[2];
        }

        int index = M->Mat[ind(i, j, k)] - 1; //material index lookup
        if (index >= 0)
        {
            Heff[0] += ArrayOfMaterials[index].Bext.X[0];
            Heff[1] += ArrayOfMaterials[index].Bext.X[1];
            Heff[2] += ArrayOfMaterials[index].Bext.X[2];

            Hext[0] += ArrayOfMaterials[index].Bext.X[0];
            Hext[1] += ArrayOfMaterials[index].Bext.X[1];
            Hext[2] += ArrayOfMaterials[index].Bext.X[2];
        }

        //global bias field
        Hext[0] += AMPx;
        Hext[1] += AMPy;
        Hext[2] += AMPz;


        //write back to memory
        H->H_m[find(i, j, k, 0)] = Hm[0];
        H->H_m[find(i, j, k, 1)] = Hm[1];
        H->H_m[find(i, j, k, 2)] = Hm[2];

        H->H_eff[find(i, j, k, 0)] = Heff[0] + AMPx;
        H->H_eff[find(i, j, k, 1)] = Heff[1] + AMPy;
        H->H_eff[find(i, j, k, 2)] = Heff[2] + AMPz;

        H->H_ext[find(i, j, k, 0)] = Hext[0];
        H->H_ext[find(i, j, k, 1)] = Hext[1];
        H->H_ext[find(i, j, k, 2)] = Hext[2];
    }

    return;
}
__global__ void ComputeEffectiveField_SinglePrecision_R2C(MEMDATA DATA, FIELD H,MAG M)
{   
    double Hm[3], Hex[3], Heff[3],Hext[3];

    //local index
    int I = threadIdx.z;
    int J = threadIdx.y;
    int K = threadIdx.x;

    //global memory index
    int i = I + blockIdx.z * blockDim.z;
    int j = J + blockIdx.y * blockDim.y;
    int k = K + blockIdx.x * blockDim.x;
  
    int index = ind(i, j, k);

    if (i< NUM && j < NUMY && k < NUMZ)
    {
        Hext[0] = 0.0;
        Hext[1] = 0.0;
        Hext[2] = 0.0;

        Hm[0] = DATA->Outx[FFTind(i, j, k)];
        Hm[1] = DATA->Outy[FFTind(i, j, k)];
        Hm[2] = DATA->Outz[FFTind(i, j, k)];

        Hex[0] = H->H_ex[find(i, j, k, 0)];
        Hex[1] = H->H_ex[find(i, j, k, 1)];
        Hex[2] = H->H_ex[find(i, j, k, 2)];

        Heff[0] = Hm[0] + Hex[0];
        Heff[1] = Hm[1] + Hex[1];
        Heff[2] = Hm[2] + Hex[2];


        if (ExternalField == 1)
        {
            Vector extfield = ExcitationFunc_CW(t_d, CELL * i, CELLY * j, CELLZ * k);
            Heff[0] += extfield.X[0];
            Heff[1] += extfield.X[1];
            Heff[2] += extfield.X[2];

            Hext[0] = extfield.X[0];
            Hext[1] = extfield.X[1];
            Hext[2] = extfield.X[2];
        }

        int index = M->Mat[ind(i, j, k)] - 1; //material index lookup
        if (index >= 0)
        {
            Heff[0] += ArrayOfMaterials[index].Bext.X[0];
            Heff[1] += ArrayOfMaterials[index].Bext.X[1];
            Heff[2] += ArrayOfMaterials[index].Bext.X[2];

            Hext[0] += ArrayOfMaterials[index].Bext.X[0];
            Hext[1] += ArrayOfMaterials[index].Bext.X[1];
            Hext[2] += ArrayOfMaterials[index].Bext.X[2];
        }

        //global bias field
        Hext[0] +=AMPx;
        Hext[1] +=AMPy;
        Hext[2] +=AMPz;


        //write back to memory
        H->H_m[find(i, j, k, 0)] = Hm[0];
        H->H_m[find(i, j, k, 1)] = Hm[1];
        H->H_m[find(i, j, k, 2)] = Hm[2];

        H->H_eff[find(i, j, k, 0)] = Heff[0] + AMPx;
        H->H_eff[find(i, j, k, 1)] = Heff[1] + AMPy;
        H->H_eff[find(i, j, k, 2)] = Heff[2] + AMPz;

        H->H_ext[find(i, j, k, 0)] = Hext[0];
        H->H_ext[find(i, j, k, 1)] = Hext[1];
        H->H_ext[find(i, j, k, 2)] = Hext[2];
    }

    return;
}
__host__ void ComputeFields(MEMDATA DATA, MAG M, FIELD H, PLANS P, int Flag)
{
    // Internal Field in units of A/nm -> multiply by mu0 in output to get Tesla
    //
    if (Flag == 1)
    {
        return;
    }
    int ExchangeHalo = 0;
    int ExchangeSharedMemorySize = 0;
    int SHAREDSIZE = DIM * sizeof(double) * (NumberofThreadsPadded.x * NumberofThreadsPadded.y * NumberofThreadsPadded.z);


    if (NoExchange == false)
    {
        if (IsPBCEnabled == false)
        {
            ExchangeField_FullGridBoundaries << <NumberofBlocks, NumberofThreads >> > (M, H);
        }
        else {
            ExchangeField_FullGridBoundaries_PBC << <NumberofBlocks, NumberofThreads >> > (M, H);
        }
    }
    cudaDeviceSynchronize();

    if (NoDemag == false) {
        if (UseSinglePrecision == false) {
            MagnetisationFFTCompute3DInitialised << <NumberofBlocksPadded, NumberofThreadsPadded>> > (DATA, M);
        }
        else {
            MagnetisationFFTCompute3DInitialised_SinglePrecision_R2C << <NumberofBlocksPadded, NumberofThreadsPadded >> > (DATA, M);
        }
        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        if (UseSinglePrecision == false) {
            MagnetisationFFT(P);
        }
        else {
            MagnetisationFFT_SinglePrecision_R2C(P);
        }
        checkCudaErrors(cudaDeviceSynchronize());

        if (UseSinglePrecision == false) {
            ComputeDemagField3DConvolution_Symmetries << <NumberofBlocksPadded, NumberofThreadsPadded >> > (DATA);
        }
        else {
        ComputeDemagField3DConvolution_Symmetries_SinglePrecision_R2C << <NumberofBlocksPadded, NumberofThreadsPadded >> > (DATA);
        }
        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        if (UseSinglePrecision == false) {
            DemagFieldInverseFFT(P);
        }
        else
        {
            DemagFieldInverseFFT_SinglePrecision_R2C(P);
        }
        checkCudaErrors(cudaDeviceSynchronize());
    }
 
    if (UseSinglePrecision == false) {
        ComputeEffectiveField << <NumberofBlocks, NumberofThreads >> > (DATA, H,M);
    }
    else
    {
        ComputeEffectiveField_SinglePrecision_R2C << <NumberofBlocks, NumberofThreads>> > (DATA, H,M);
    }
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    if (UniAnisotropy_h == 1)
    {
        UniaxialAnisotropy << <NumberofBlocks, NumberofThreads >> > (M, H);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if (SpinTransferTorque_h == 1)
    {
        Compute_ZhangLi_STT << <NumberofBlocks, NumberofThreads >> > (DATA, H, M);
        checkCudaErrors(cudaDeviceSynchronize());
    }
}
__host__ void ComputeFields_RKStageEvaluation(MEMDATA DATA, MAG M, FIELD H, PLANS P, int Flag)
{
    // Internal Field in units of A/nm -> multiply by mu0 in output to get Tesla
    //
    if (Flag == 1)
    {
        return;
    }
    int ExchangeHalo = 0;
    int ExchangeSharedMemorySize = 0;
    int SHAREDSIZE = DIM * sizeof(double) * (NumberofThreadsPadded.x * NumberofThreadsPadded.y * NumberofThreadsPadded.z);


    if (NoExchange == false)
    {
        if (IsPBCEnabled == false)
        {
            ExchangeField_FullGridBoundaries_Mn1 << <NumberofBlocks, NumberofThreads >> > (M, H);
        }
        else {
            ExchangeField_FullGridBoundaries_PBC_Mn1 << <NumberofBlocks, NumberofThreads >> > (M, H);
        }
    }
    cudaDeviceSynchronize();

    if (NoDemag == false) {

        if (UseSinglePrecision == false) {
            MagnetisationFFTCompute3DInitialised_Mn1 << <NumberofBlocksPadded, NumberofThreadsPadded >> > (DATA, M);
        }
        else {
            MagnetisationFFTCompute3DInitialised_Mn1_SinglePrecision_R2C << <NumberofBlocksPadded, NumberofThreadsPadded >> > (DATA, M);
        }
        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        if (UseSinglePrecision == false) {
            MagnetisationFFT(P);
        }
        else
        {
            MagnetisationFFT_SinglePrecision_R2C(P);
        }
        checkCudaErrors(cudaDeviceSynchronize());

        if (UseSinglePrecision == false) {
            ComputeDemagField3DConvolution_Symmetries << <NumberofBlocksPadded, NumberofThreadsPadded >> > (DATA);
        }
        else
        {
         ComputeDemagField3DConvolution_Symmetries_SinglePrecision_R2C << <NumberofBlocksPadded, NumberofThreadsPadded >> > (DATA);
        }
        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        if (UseSinglePrecision == false) {
            DemagFieldInverseFFT(P);
        }
        else
        {
            DemagFieldInverseFFT_SinglePrecision_R2C(P);
        }
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if (UseSinglePrecision == false) {
        ComputeEffectiveField << <NumberofBlocks, NumberofThreads>> > (DATA, H,M);
    }
    else
    {
    ComputeEffectiveField_SinglePrecision_R2C << <NumberofBlocks, NumberofThreads>> > (DATA, H,M);
    }
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    if (UniAnisotropy_h == 1)
    {
        UniaxialAnisotropy_Mn1 << <NumberofBlocks, NumberofThreads >> > (M, H);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if (SpinTransferTorque_h == 1)
    {
        Compute_ZhangLi_STT_Mn1 << <NumberofBlocks, NumberofThreads >> > (DATA, H, M);
        checkCudaErrors(cudaDeviceSynchronize());
    }
}
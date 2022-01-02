#include "Device_FFT_Functions.cuh"
#include <cufft.h>
#include "Device_Globals_Constants.cuh"
#include "Host_Globals.cuh"
#include "GlobalDefines.cuh"
#include "Pointer_Functions.cuh"
#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include "DemagnetisingTensor_Functions.cuh"

__host__ void FFTPlansInitialise(PLANS P)
{
    int Nx = 2 * NUM_h;
    int Ny = 2 * NUMY_h;
    int Nz = 2 * NUMZ_h;

    if (IsPBCEnabled == false)
    {
        Nx = 2 * NUM_h;
        Ny = 2 * NUMY_h;
        Nz = 2 * NUMZ_h;
    }
    if (PBC_x_h != 0)
    {
        Nx = NUM_h;
        Ny = 2 * NUMY_h;
        Nz = 2 * NUMZ_h;
    }
    if (PBC_y_h != 0)
    {
        Nx = 2 * NUM_h;
        Ny = NUMY_h;
        Nz = 2 * NUMZ_h;
    }
    if (PBC_y_h != 0 && PBC_x_h != 0)
    {
        Nx = NUM_h;
        Ny = NUMY_h;
        Nz = 2 * NUMZ_h;
    }

    int FFT_SIZE = Nx * Ny * Nz;

    checkCudaErrors(cudaMemcpyToSymbol(FFT_NORM, &FFT_SIZE, sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(PADNUM, &Nx, sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(PADNUMY, &Ny, sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(PADNUMZ, &Nz, sizeof(int)));

    CHECK_CUFFT_ERRORS(
        cufftPlan3d(&(P->MxPlan), Nx, Ny, Nz, CUFFT_D2Z));
    CHECK_CUFFT_ERRORS(
        cufftPlan3d(&(P->MyPlan), Nx, Ny, Nz, CUFFT_D2Z));
    CHECK_CUFFT_ERRORS(
        cufftPlan3d(&(P->MzPlan), Nx, Ny, Nz, CUFFT_D2Z));

    CHECK_CUFFT_ERRORS(
        cufftPlan3d(&(P->HxPlan), Nx, Ny, Nz, CUFFT_Z2D));
    CHECK_CUFFT_ERRORS(
        cufftPlan3d(&(P->HyPlan), Nx, Ny, Nz, CUFFT_Z2D));
    CHECK_CUFFT_ERRORS(
        cufftPlan3d(&(P->HzPlan), Nx, Ny, Nz, CUFFT_Z2D));

    PointerPlansCheck(P->MxPlan != CUFFT_SUCCESS), PointerPlansCheck(P->MyPlan != CUFFT_SUCCESS),
        PointerPlansCheck(P->MzPlan != CUFFT_SUCCESS), PointerPlansCheck(P->HxPlan != CUFFT_SUCCESS),
        PointerPlansCheck(P->HyPlan != CUFFT_SUCCESS), PointerPlansCheck(P->HzPlan != CUFFT_SUCCESS);

}
__host__ void FFTPlansInitialise_SinglePrecision_R2C(PLANS P)
{
    int Nx = 2 * NUM_h;
    int Ny = 2 * NUMY_h;
    int Nz = 2 * NUMZ_h;

    if (IsPBCEnabled == false)
    {
        Nx = 2 * NUM_h;
        Ny = 2 * NUMY_h;
        Nz = 2 * NUMZ_h;
    }
    if (PBC_x_h != 0)
    {
        Nx = NUM_h;
        Ny = 2 * NUMY_h;
        Nz = 2 * NUMZ_h;
    }
    if (PBC_y_h != 0)
    {
        Nx = 2 * NUM_h;
        Ny = NUMY_h;
        Nz = 2 * NUMZ_h;
    }
    if (PBC_y_h != 0 && PBC_x_h != 0)
    {
        Nx = NUM_h;
        Ny = NUMY_h;
        Nz = 2 * NUMZ_h;
    }

    int FFT_SIZE = Nx * Ny * Nz;

    checkCudaErrors(cudaMemcpyToSymbol(FFT_NORM, &FFT_SIZE, sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(PADNUM, &Nx, sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(PADNUMY, &Ny, sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(PADNUMZ, &Nz, sizeof(int)));

    CHECK_CUFFT_ERRORS(
        cufftPlan3d(&(P->MxPlan), Nx, Ny, Nz, CUFFT_R2C));
    CHECK_CUFFT_ERRORS(
        cufftPlan3d(&(P->MyPlan), Nx, Ny, Nz, CUFFT_R2C));
    CHECK_CUFFT_ERRORS(
        cufftPlan3d(&(P->MzPlan), Nx, Ny, Nz, CUFFT_R2C));

    CHECK_CUFFT_ERRORS(
        cufftPlan3d(&(P->HxPlan), Nx, Ny, Nz, CUFFT_C2R));
    CHECK_CUFFT_ERRORS(
        cufftPlan3d(&(P->HyPlan), Nx, Ny, Nz, CUFFT_C2R));
    CHECK_CUFFT_ERRORS(
        cufftPlan3d(&(P->HzPlan), Nx, Ny, Nz, CUFFT_C2R));

    PointerPlansCheck(P->MxPlan != CUFFT_SUCCESS), PointerPlansCheck(P->MyPlan != CUFFT_SUCCESS),
    PointerPlansCheck(P->MzPlan != CUFFT_SUCCESS), PointerPlansCheck(P->HxPlan != CUFFT_SUCCESS),
    PointerPlansCheck(P->HyPlan != CUFFT_SUCCESS), PointerPlansCheck(P->HzPlan != CUFFT_SUCCESS);

}
__host__ void DemagFieldInverseFFT_SinglePrecision_R2C(PLANS P)
{
    CHECK_CUFFT_ERRORS(
        cufftExecC2R(P->HxPlan, ((DEVICE_PTR_STRUCT.DATA)->xFFT_s),
            ((DEVICE_PTR_STRUCT.DATA)->Outx)));

    CHECK_CUFFT_ERRORS(
        cufftExecC2R(P->HyPlan, ((DEVICE_PTR_STRUCT.DATA)->yFFT_s),
            ((DEVICE_PTR_STRUCT.DATA)->Outy)));

    CHECK_CUFFT_ERRORS(
        cufftExecC2R(P->HzPlan, ((DEVICE_PTR_STRUCT.DATA)->zFFT_s),
            ((DEVICE_PTR_STRUCT.DATA)->Outz)));
}
__host__ void DemagFieldInverseFFT(PLANS P)
{
    CHECK_CUFFT_ERRORS(
        cufftExecZ2D(P->HxPlan, (cufftDoubleComplex*)((DEVICE_PTR_STRUCT.DATA)->xFFT),
            (double*)((DEVICE_PTR_STRUCT.DATA)->Outx_d)));

    CHECK_CUFFT_ERRORS(
        cufftExecZ2D(P->HyPlan, (cufftDoubleComplex*)((DEVICE_PTR_STRUCT.DATA)->yFFT),
            (double*)((DEVICE_PTR_STRUCT.DATA)->Outy_d)));

    CHECK_CUFFT_ERRORS(
        cufftExecZ2D(P->HzPlan, (cufftDoubleComplex*)((DEVICE_PTR_STRUCT.DATA)->zFFT),
            (double*)((DEVICE_PTR_STRUCT.DATA)->Outz_d)));
}
__host__ void MagnetisationFFT(PLANS P)
{
    CHECK_CUFFT_ERRORS(
        cufftExecD2Z(P->MxPlan, ((DEVICE_PTR_STRUCT.DATA)->Outx_d),
            (cufftDoubleComplex*)((DEVICE_PTR_STRUCT.DATA)->xFFT)));

    CHECK_CUFFT_ERRORS(
        cufftExecD2Z(P->MyPlan, ((DEVICE_PTR_STRUCT.DATA)->Outy_d),
            (cufftDoubleComplex*)(DEVICE_PTR_STRUCT.DATA)->yFFT));

    CHECK_CUFFT_ERRORS(
        cufftExecD2Z(P->MzPlan, ((DEVICE_PTR_STRUCT.DATA)->Outz_d),
            (cufftDoubleComplex*)((DEVICE_PTR_STRUCT.DATA))->zFFT));
}
__host__ void MagnetisationFFT_SinglePrecision_R2C(PLANS P)
{
    
    CHECK_CUFFT_ERRORS(
        cufftExecR2C(P->MxPlan, ((DEVICE_PTR_STRUCT.DATA)->Outx),
            ((DEVICE_PTR_STRUCT.DATA)->xFFT_s)));

    CHECK_CUFFT_ERRORS(
        cufftExecR2C(P->MyPlan, ((DEVICE_PTR_STRUCT.DATA)->Outy),
            (DEVICE_PTR_STRUCT.DATA)->yFFT_s));

    CHECK_CUFFT_ERRORS(
        cufftExecR2C(P->MzPlan, ((DEVICE_PTR_STRUCT.DATA)->Outz),
            ((DEVICE_PTR_STRUCT.DATA))->zFFT_s));

}
__host__ void DemagTensorFFT_Symmetries(MEMDATA DATA_h, MEMDATA DATA_d)
{
    cufftHandle KxxPlan;
    int Nx = 2 * NUM_h;
    int Ny = 2 * NUMY_h;
    int Nz = 2 * NUMZ_h;
    if (IsPBCEnabled == false)
    {
        Nx = 2 * NUM_h;
        Ny = 2 * NUMY_h;
        Nz = 2 * NUMZ_h;
    }
    if (PBC_x_h != 0)
    {
        Nx = NUM_h;
        Ny = 2 * NUMY_h;
        Nz = 2 * NUMZ_h;
    }
    if (PBC_y_h != 0)
    {
        Nx = 2 * NUM_h;
        Ny = NUMY_h;
        Nz = 2 * NUMZ_h;
    }
    if (PBC_y_h != 0 && PBC_x_h != 0)
    {
        Nx = NUM_h;
        Ny = NUMY_h;
        Nz = 2 * NUMZ_h;
    }

    CHECK_CUFFT_ERRORS(
        cufftPlan3d(&KxxPlan, Nx, Ny, Nz, CUFFT_Z2Z));

    int SIZE = Nx * Ny * Nz * sizeof(fftw_complex);

    MEMDATA temp = (MEMDATA)malloc(sizeof(Dataptr));
    PointerCheck(temp != NULL);

    checkCudaErrors(cudaMemcpy(temp, DATA_d, sizeof(Dataptr), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(temp->xFFT, DATA_h->kxx, SIZE, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(temp->yFFT, DATA_h->kyy, SIZE, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(temp->zFFT, DATA_h->kzz, SIZE, cudaMemcpyHostToDevice));

    //Diagonals

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    CHECK_CUFFT_ERRORS(
        cufftExecZ2Z(KxxPlan, (cufftDoubleComplex*)((DEVICE_PTR_STRUCT.DATA)->xFFT),
            (cufftDoubleComplex*)((DEVICE_PTR_STRUCT.DATA)->xFFT), CUFFT_FORWARD));
    CHECK_CUFFT_ERRORS(
        cufftExecZ2Z(KxxPlan, (cufftDoubleComplex*)((DEVICE_PTR_STRUCT.DATA)->yFFT),
            (cufftDoubleComplex*)((DEVICE_PTR_STRUCT.DATA)->yFFT), CUFFT_FORWARD));
    CHECK_CUFFT_ERRORS(
        cufftExecZ2Z(KxxPlan, (cufftDoubleComplex*)((DEVICE_PTR_STRUCT.DATA)->zFFT),
            (cufftDoubleComplex*)((DEVICE_PTR_STRUCT.DATA)->zFFT), CUFFT_FORWARD));

    DemagTensorStoreFirstOctant_Diagonals << <NumberofBlocksPadded, NumberofThreadsPadded >> > (DATA_d);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //Off-Diagonals

    checkCudaErrors(cudaMemcpy(temp->xFFT, DATA_h->kxy, SIZE, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(temp->yFFT, DATA_h->kxz, SIZE, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(temp->zFFT, DATA_h->kyz, SIZE, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    CHECK_CUFFT_ERRORS(
        cufftExecZ2Z(KxxPlan, (cufftDoubleComplex*)((DEVICE_PTR_STRUCT.DATA)->xFFT),
            (cufftDoubleComplex*)((DEVICE_PTR_STRUCT.DATA)->xFFT), CUFFT_FORWARD));
    CHECK_CUFFT_ERRORS(
        cufftExecZ2Z(KxxPlan, (cufftDoubleComplex*)((DEVICE_PTR_STRUCT.DATA)->yFFT),
            (cufftDoubleComplex*)((DEVICE_PTR_STRUCT.DATA)->yFFT), CUFFT_FORWARD));
    CHECK_CUFFT_ERRORS(
        cufftExecZ2Z(KxxPlan, (cufftDoubleComplex*)((DEVICE_PTR_STRUCT.DATA)->zFFT),
            (cufftDoubleComplex*)((DEVICE_PTR_STRUCT.DATA)->zFFT), CUFFT_FORWARD));

    DemagTensorStoreFirstOctant_OffDiagonals << <NumberofBlocksPadded, NumberofThreadsPadded >> > (DATA_d);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    free(temp);
    cufftDestroy(KxxPlan);

}
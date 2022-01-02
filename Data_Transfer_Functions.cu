#include "Data_Transfer_Functions.cuh"

__host__ void CopyDevicePointers(MEMDATA DATA, MAG M, FIELD H)
{
    DEVICE_PTR_STRUCT.DATA = DATA;
    DEVICE_PTR_STRUCT.M = M;
    DEVICE_PTR_STRUCT.H = H;
}
__host__ void CopyMagToDevice(MAG M_h, MAG M_d)
{
    MAG temp = (MAG)malloc(sizeof(Magnetisation));
    checkCudaErrors(cudaMemcpy(temp, M_d, sizeof(Magnetisation), cudaMemcpyDeviceToHost));

    int SIZE = NUM_h * NUMY_h * NUMZ_h;

    checkCudaErrors(cudaMemcpy(temp->M, M_h->M, DIM * DIM * SIZE * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(temp->Mat, M_h->Mat, SIZE * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(temp->Bd, M_h->Bd, DIM * SIZE * sizeof(int), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(temp->M, M_h->M, DIM * DIM * SIZE * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(temp->Mat, M_h->Mat, SIZE * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(temp->NUMCELLS, M_h->NUMCELLS, sizeof(int), cudaMemcpyHostToDevice));
   
    free(temp);

    return;
}
__host__ void CopyFieldToDevice(FIELD F_h, FIELD F_d)
{
    FIELD temp = (FIELD)malloc(sizeof(Field));
    checkCudaErrors(cudaMemcpy(temp, F_d, sizeof(Field), cudaMemcpyDeviceToHost));
    int SIZE = NUM_h * NUMY_h * NUMZ_h;

    checkCudaErrors(cudaMemcpy(temp->H_m, F_h->H_m, DIM * SIZE * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(temp->H_ex, F_h->H_ex, DIM * SIZE * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(temp->H_eff, F_h->H_eff, DIM * SIZE * sizeof(double), cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(temp->H_ext, F_h->H_ext, DIM * SIZE * sizeof(double), cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(temp->H_anis, F_h->H_anis, DIM * SIZE * sizeof(double), cudaMemcpyHostToDevice));
    free(temp);
    return;
}
__host__ void CopyMemDataToDevice(MEMDATA DATA_h, MEMDATA DATA_d)
{

    int SIZE_M = NUM_h * NUMY_h * NUMZ_h * sizeof(double);


    int SIZE = ((2 - PBC_x_h) * NUM_h * (2 - PBC_y_h) * NUMY_h * 2 * (NUMZ_h)) * sizeof(fftw_complex);

    int SIZE_s = ((2 - PBC_x_h) * NUM_h * (2 - PBC_y_h) * NUMY_h * 2 * (NUMZ_h)) * sizeof(cufftComplex);

    MEMDATA temp = (MEMDATA)malloc(sizeof(Dataptr));
    PointerCheck(temp != NULL);

    checkCudaErrors(cudaMemcpy(temp, DATA_d, sizeof(Dataptr), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(temp->xFFT, DATA_h->xFFT, SIZE, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(temp->yFFT, DATA_h->yFFT, SIZE, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(temp->zFFT, DATA_h->zFFT, SIZE, cudaMemcpyHostToDevice));

    if (UseSinglePrecision == true)
    {
        checkCudaErrors(cudaMemcpy(temp->xFFT_s, DATA_h->xFFT_s, SIZE_s, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(temp->yFFT_s, DATA_h->yFFT_s, SIZE_s, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(temp->zFFT_s, DATA_h->zFFT_s, SIZE_s, cudaMemcpyHostToDevice));
    }

    int RSIZE = NumberofBlocksIntegrator.x * NumberofBlocksIntegrator.y * NumberofBlocksIntegrator.z;

    checkCudaErrors(cudaMemcpy((DEVICE_PTR_STRUCT.DATA)->MaxTorqueReduction, DATA_h->MaxTorqueReduction, RSIZE * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((DEVICE_PTR_STRUCT.DATA)->StepReduction, DATA_h->StepReduction, RSIZE * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((DEVICE_PTR_STRUCT.DATA)->dE_Reduction, DATA_h->dE_Reduction, RSIZE * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((DEVICE_PTR_STRUCT.DATA)->NewtonStepsReduction, DATA_h->NewtonStepsReduction, RSIZE * sizeof(int), cudaMemcpyHostToDevice));

    free(temp);
    return;
}
__host__ void CopyEffectiveFieldFromDevice(FIELD F_h, FIELD F_d)
{
    FIELD temp = (FIELD)malloc(sizeof(Field));
    checkCudaErrors(cudaMemcpy(temp, F_d, sizeof(Field), cudaMemcpyDeviceToHost));
    int SIZE = NUM_h * NUMY_h * NUMZ_h;

    checkCudaErrors(cudaMemcpy(F_h->H_eff, temp->H_eff,SIZE * DIM * sizeof(double), cudaMemcpyDeviceToHost));
    free(temp);
    return;
}
__host__ void CopyMemDataFromDevice(MEMDATA DATA_h, MEMDATA DATA_d)
{
    int SIZE = 8 * NUM_h * NUMY_h * NUMZ_h * sizeof(fftw_complex);
    int SIZE_s = 8 * NUM_h * NUMY_h * NUMZ_h * sizeof(cufftComplex);

    if (PBC_x_h != 0 || PBC_y_h != 0)
    {
        SIZE = 4 * NUM_h * NUMY_h * NUMZ_h * sizeof(fftw_complex);
    }
    if (PBC_x_h != 0 && PBC_y_h != 0)
    {
        SIZE = 2 * NUM_h * NUMY_h * NUMZ_h * sizeof(fftw_complex);
    }

    MEMDATA temp = (MEMDATA)malloc(sizeof(Dataptr));
    PointerCheck(temp != NULL);

    checkCudaErrors(cudaMemcpy(temp, DATA_d, sizeof(Dataptr), cudaMemcpyDeviceToHost));

  
        checkCudaErrors(cudaMemcpy(DATA_h->xFFT, temp->xFFT, SIZE, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(DATA_h->yFFT, temp->yFFT, SIZE, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(DATA_h->zFFT, temp->zFFT, SIZE, cudaMemcpyDeviceToHost));
    
    if (UseSinglePrecision == true)
    {
        checkCudaErrors(cudaMemcpy(DATA_h->xFFT_s, temp->xFFT_s, SIZE_s, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(DATA_h->yFFT_s, temp->yFFT_s, SIZE_s, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(DATA_h->zFFT_s, temp->zFFT_s, SIZE_s, cudaMemcpyDeviceToHost));
    }

    checkCudaErrors(cudaMemcpy(DATA_h->kxx, temp->kxx, SIZE, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(DATA_h->kxy, temp->kxy, SIZE, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(DATA_h->kxz, temp->kxz, SIZE, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(DATA_h->kyy, temp->kyy, SIZE, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(DATA_h->kyz, temp->kyz, SIZE, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(DATA_h->kzz, temp->kzz, SIZE, cudaMemcpyDeviceToHost));

    free(temp);
    return;
}
__host__ void CopyExchangeFieldFromDevice(FIELD F_h, FIELD F_d)
{
    FIELD temp = (FIELD)malloc(sizeof(Field));
    checkCudaErrors(cudaMemcpy(temp, F_d, sizeof(Field), cudaMemcpyDeviceToHost));
    int SIZE = NUM_h * NUMY_h * NUMZ_h;

    checkCudaErrors(cudaMemcpy(F_h->H_ex, temp->H_ex, DIM * SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    free(temp);
    return;
}
__host__ void CopyDemagFieldFromDevice(FIELD F_h, FIELD F_d)
{
    FIELD temp = (FIELD)malloc(sizeof(Field));
    checkCudaErrors(cudaMemcpy(temp, F_d, sizeof(Field), cudaMemcpyDeviceToHost));
    int SIZE = NUM_h * NUMY_h * NUMZ_h;

    checkCudaErrors(cudaMemcpy(F_h->H_m, temp->H_m, DIM * SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    free(temp);
    return;
}
__host__ void CopyDemagComponentsFromDevice(FIELD H_h, FIELD H_d)
{
    FIELD temp = (FIELD)malloc(sizeof(Field));
    checkCudaErrors(cudaMemcpy(temp, H_d, sizeof(Field), cudaMemcpyDeviceToHost));
    int SIZE = NUM_h * NUMY_h * NUMZ_h;

    int index_xcomp = find_h(0, 0, 0, 0);
    int index_ycomp = find_h(0, 0, 0, 1);
    int index_zcomp = find_h(0, 0, 0, 2);

    checkCudaErrors(cudaMemcpy(&(H_h->H_m[index_xcomp]), &(temp->H_m[index_xcomp]), SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&(H_h->H_m[index_ycomp]), &(temp->H_m[index_ycomp]), SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&(H_h->H_m[index_zcomp]), &(temp->H_m[index_zcomp]), SIZE * sizeof(double), cudaMemcpyDeviceToHost));

    free(temp);
}
__host__ void CopyMagFromDevice(MAG M_h, MAG M_d)
{
    MAG temp = (MAG)malloc(sizeof(Magnetisation));
    checkCudaErrors(cudaMemcpy(temp, M_d, sizeof(Magnetisation), cudaMemcpyDeviceToHost));
    int SIZE = NUM_h * NUMY_h * NUMZ_h;

    checkCudaErrors(cudaMemcpy(M_h->M, temp->M, DIM * DIM * SIZE * sizeof(double), cudaMemcpyDeviceToHost));


    free(temp);
}
__host__ void CopyMagComponentsFromDevice(MAG M_h, MAG M_d)
{
    MAG temp = (MAG)malloc(sizeof(Magnetisation));
    checkCudaErrors(cudaMemcpy(temp, M_d, sizeof(Magnetisation), cudaMemcpyDeviceToHost));
    int SIZE = NUM_h * NUMY_h * NUMZ_h;

    int index_xcomp = mind_h(0, 0, 0, 0, 0);
    int index_ycomp = mind_h(0, 0, 0, 0, 1);
    int index_zcomp = mind_h(0, 0, 0, 0, 2);

    checkCudaErrors(cudaMemcpy(&(M_h->M[index_xcomp]), &(temp->M[index_xcomp]), SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&(M_h->M[index_ycomp]), &(temp->M[index_ycomp]), SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&(M_h->M[index_zcomp]), &(temp->M[index_zcomp]), SIZE * sizeof(double), cudaMemcpyDeviceToHost));

    free(temp);
}
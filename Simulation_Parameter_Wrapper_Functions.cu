#include "Simulation_Parameter_Wrapper_Functions.cuh"
__host__ void SetMaxError(double E)
{
    RelTol = E;
}
__host__ void PBC(int x, int y)
{
    if (x > 0)
    {
        PBC_x_h = 1;
        PBC_x_images_h = x;
        IsPBCEnabled = true;
    }
    if (y > 0)
    {
        PBC_y_h = 1;
        PBC_y_images_h = y;
        IsPBCEnabled = true;
    }

    checkCudaErrors(cudaMemcpyToSymbol(PBC_x, &PBC_x_h, sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(PBC_y, &PBC_y_h, sizeof(int)));
}
__host__ void GridSize(int Nx, int Ny, int Nz)
{
    NUM_h = Nx;
    NUMY_h = Ny;
    NUMZ_h = Nz;
    checkCudaErrors(cudaMemcpyToSymbol(NUM, &NUM_h, sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(NUMY, &NUMY_h, sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(NUMZ, &NUMZ_h, sizeof(int)));

    cudaDeviceSynchronize();
}
__host__ void CellSize(double X, double Y, double Z)
{
    CELL_h = X, CELLY_h = Y, CELLZ_h = Z;
    checkCudaErrors(cudaMemcpyToSymbol(CELL, &CELL_h, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(CELLY, &CELLY_h, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(CELLZ, &CELLZ_h, sizeof(double)));
}
__host__ void SetMaterialParameters(double Ms, double A, double damping)
{
    A_ex_h = A * (1e18), MSAT_h = Ms * (1e-3), alpha_h = damping;

    checkCudaErrors(cudaMemcpyToSymbol(MSAT, &MSAT_h, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(A_ex, &A_ex_h, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(alpha, &alpha_h, sizeof(double)));

}
__host__ Region SetRegion(int x1, int x2, int y1, int y2, int z1, int z2)
{
    Region R;
    R.x[0] = x1;
    R.x[1] = x2;
    R.y[0] = y1;
    R.y[1] = y2;
    R.z[0] = z1;
    R.z[1] = z2;
    return R;
}
Region DefineRegion(int x1, int x2, int y1, int y2, int z1, int z2)
{
    Region result;
    result.x[0] = x1;
    result.x[1] = x2;
    result.y[0] = y1;
    result.y[1] = y2;
    result.z[0] = z1;
    result.z[1] = z2;

    return result;
}
__host__ void ApplyBiasField(double X, double Y, double Z)
{
    IncludeBiasField(true);
    AMPx_h = X / mu, AMPy_h = Y / mu, AMPz_h = Z / mu;
    checkCudaErrors(cudaMemcpyToSymbol(AMPx, &AMPx_h, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(AMPy, &AMPy_h, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(AMPz, &AMPz_h, sizeof(double)));
}
__host__ void ApplySpinTransferTorque_InMaterial(MaterialHandle handle,double P, Vector J, double Xi)
{
    SpinTransferTorque_h = 1;
    STTParameters params;
    params.handle = handle;
    params.J[0] = J.X[0];
    params.J[1] = J.X[1];
    params.J[2] = J.X[2];
    params.P = P;
    params.Xi = Xi;

    checkCudaErrors(cudaMemcpyToSymbol(SpinTransferTorque, &SpinTransferTorque_h, sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(SpinTransferTorqueParameters, &params, sizeof(STTParameters)));

}
__host__ void IncludeExternalField(bool A, ExtFieldFunctionPtr f)
{
    if (A == true)
    {
        ExtFieldFunctionPtr f_h, f_d;
        ExternalField_h = 1;
        // checkCudaErrors(cudaMemcpyToSymbol(&(DEVICE_PTR_STRUCT.DATA)->ExtFieldAdd,ExcitiationFunc_CW, sizeof(ExtFieldFunctionPtr)));
        checkCudaErrors(cudaMemcpyToSymbol(ExternalField, &ExternalField_h, sizeof(int)));
        //  checkCudaErrors(cudaMemcpyFromSymbol(&f_h,f, sizeof(ExtFieldFunctionPtr)));

        //  checkCudaErrors(cudaMemcpy(&(DEVICE_PTR_STRUCT.DATA)->ExtFieldAdd, f_h,
         //     sizeof(ExtFieldFunctionPtr), cudaMemcpyHostToDevice));
    }
    else
    {
        ExternalField_h = 0;
        checkCudaErrors(cudaMemcpyToSymbol(ExternalField, &ExternalField_h, sizeof(int)));
    }

}
__host__ void IncludeBiasField(bool A)
{
    if (A == true)
    {
        BiasField_h = 1;
        checkCudaErrors(cudaMemcpyToSymbol(BiasField, &BiasField_h, sizeof(int)));
    }
    else
    {
        BiasField_h = 0;
        checkCudaErrors(cudaMemcpyToSymbol(BiasField, &BiasField_h, sizeof(int)));
    }

}
__host__ void IncludeUniaxialAnisotropy(bool A)
{
    if (A == true)
    {
        UniAnisotropy_h = 1;
        checkCudaErrors(cudaMemcpyToSymbol(UniAnisotropy, &UniAnisotropy_h, sizeof(int)));
    }
    else {
        UniAnisotropy_h = 0;
        checkCudaErrors(cudaMemcpyToSymbol(UniAnisotropy, &UniAnisotropy_h, sizeof(int)));
    }

}
__host__ void SetMethod(int n)
{
    METHOD_h = n;
}
__host__ void SetTime(double t)
{
    TIME = t;
    return;
}
__host__ void SetStepSize(double step)
{
    h_h = step;
    checkCudaErrors(cudaMemcpyToSymbol(h_d, &h_h, sizeof(double)));
    cudaDeviceSynchronize();
}
__host__ void SetSamplingPeriod(double T)
{
    Sampling_Period = T;
}
void SetUniaxialAnisotropy(double Ku, int x, int y, int z)
{
    if (Ku != 0 && (x != 0 || y != 0 || z != 0))
    {
        UniAnisotropy_h = 1,
        Uanisx_h = x,
        Uanisy_h = y,
        Uanisz_h = z;
        K_UANIS_h =2. * Ku / (mu *MSAT_h);
        checkCudaErrors(cudaMemcpyToSymbol(UniAnisotropy, &UniAnisotropy_h, sizeof(int)));
        checkCudaErrors(cudaMemcpyToSymbol(Uanisx, &Uanisx_h, sizeof(int)));
        checkCudaErrors(cudaMemcpyToSymbol(Uanisy, &Uanisy_h, sizeof(int)));
        checkCudaErrors(cudaMemcpyToSymbol(Uanisz, &Uanisz_h, sizeof(int)));
        checkCudaErrors(cudaMemcpyToSymbol(K_UANIS, &K_UANIS_h, sizeof(double)));

    }
}
void StrictTimeSampling(bool x)
{
    StrictSampling = x;
}
void SinglePrecision(bool x)
{
    UseSinglePrecision = x;
}
Vector MakeVector(double x,double y,double z)
{
    Vector result;
    result.X[0] = x;
    result.X[1] = y;
    result.X[2] = z;
    return result;
}



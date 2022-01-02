#include <cuda_runtime.h>
#include "DataTypes.cuh"
#include "Host_Globals.cuh"
#include "Device_Globals_Constants.cuh"
#include "GlobalDefines.cuh"
#include <helper_cuda.h>

#ifndef SIMULATION_PARAMETER_WRAPPER_FUNCTIONS_CUH
#define SIMULATION_PARAMETER_WRAPPER_FUNCTIONS_CUH

__host__ void SetMaxError(double E);
__host__ void PBC(int x, int y);
__host__ void GridSize(int Nx, int Ny, int Nz);
__host__ void CellSize(double X, double Y, double Z);
__host__ void SetMethod(int n);
__host__ void SetTime(double t);
__host__ void SetSamplingPeriod(double T);
__host__ void SetStepSize(double step);
__host__ void SetMaterialParameters(double Ms, double A, double damping);
__host__ Region SetRegion(int x1, int x2, int y1, int y2, int z1, int z2);
Region DefineRegion(int x1, int x2, int y1, int y2, int z1, int z2);
__host__ void IncludeExternalField(bool A, ExtFieldFunctionPtr f);
__host__ void IncludeUniaxialAnisotropy(bool A);
__host__ void IncludeBiasField(bool A);
__host__ void ApplyBiasField(double X, double Y, double Z);
__host__ void ApplySpinTransferTorque_InMaterial(MaterialHandle handle,
	                                             double P, Vector J, double Xi);
void SetUniaxialAnisotropy(double Ku, int x, int y, int z);
void StrictTimeSampling(bool x);
void SinglePrecision(bool x);
Vector MakeVector(double x, double y, double z);
#endif
#include "Host_Globals.cuh"
#include "DataTypes.cuh"
#include "GlobalDefines.cuh"
#include "Device_Globals_Constants.cuh"
#include <helper_cuda.h>

#ifndef DEFINEMATERIALS_FUNCTIONS_CUH
#define DEFINEMATERIALS_FUNCTIONS_CUH

void InitialiseMaterialsArray();
void WriteMaterialsArrayToDevice();
MaterialHandle DefineMaterial(double Ms, double Aex, double damping, double Ku, Vector Bext,int ExtType);
MaterialHandle DefineMaterial(double Ms, double Aex, double damping, double Ku, Vector Bext);
MaterialHandle DefineMaterial(double Ms, double Aex, double damping, Vector Bext);
MaterialHandle DefineMaterial(double Ms, double Aex, double damping, double Ku);
MaterialHandle DefineMaterial(double Ms, double Aex, double damping);
void ApplyMaterialParameters(MaterialHandle handle, double Ms, double Aex, double damping, double Ku, Vector Bext,int ExtType);

#endif // !DEFINEMATERIALS_FUNCTIONS_CUH

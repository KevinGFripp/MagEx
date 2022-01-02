#include "DataTypes.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

#ifndef PRINT_AND_LOG_FUNCTIONS_CUH
#define PRINT_AND_LOG_FUNCTIONS_CUH

//Convert to Big-Endian
__host__ void ReverseByteOrder_float(char* data);
//OOMMF OVF 2.0
__host__ void WriteOVF_Mag_Binary_4(OutputFormat Out, MAG M, int c);
__host__ void WriteOVF_Mag_Binary_8(OutputFormat Out, MAG M, int c);

//Mag Binary File Format (mbf)
__host__ void Write_mbf(OutputFormat Out, MAG M, int c);
__host__ void Write_mbf_single(OutputFormat Out, MAG M, int c);

//Field Binary File Format (Bmf)
__host__ void Write_bmf(OutputFormat Out, FIELD H, int c);
__host__ void Write_emf(OutputFormat Out, FIELD H, int c);

__host__ void ScheduleSampling(FILE* log, int* Count, int* ControlCount,
    double Period, OutputFormat Out, MEMDATA DATA_d, MAG M_d, MAG M,
    FIELD H_d, FIELD H, Vector Step, double stepsize);

//
//Printing and Log Functions
//
__host__ FILE* CreateLog();
__host__ void UpdateLog(FILE* log, double MaxTorque, double step, double ControlCount, double MaxErr,
    double ExEnergy, double DmEnergy, double TotalEnergy, double Mx, double My,
    double Mz, double time);
__host__ OutputFormat NewOutput();
__host__ void printDemagTensorSlice(MEMDATA DATA, int k);
__host__ void printFFT(MEMDATA DATA, int k);
__host__ void printField(FIELD H, int k);
__host__  void printExchangeField(FIELD H, int k);
__host__ void printmagnetisationframe(MAG M, int c, int k);
__host__ void printDemagTensorFFT(MEMDATA DATA, int k);
__host__ void printmagnetisation_Outputs(MAG M, int c, OutputFormat Out);
__host__  void printExchangeField_Outputs(FIELD H, int c, OutputFormat Out);
__host__  void printDemagField_Outputs(FIELD H, int c, OutputFormat Out);
__host__  void printEffectiveField_Outputs(FIELD H, int c, OutputFormat Out);
__host__ void printmagnetisationFull_Outputs(MAG M, int c, OutputFormat Out);

#endif // !PRINT_AND_LOG_FUNCTIONS_CUH

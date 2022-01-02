#include <cuda_runtime.h>
#include "DataTypes.cuh"

#ifndef REDUCTION_FUNCTIONS_CUH
#define REDUCTION_FUNCTIONS_CUH
__host__ void ReductionArraysInit(MEMDATA DATA);

__device__ void warpReduce(volatile double* sdata, int tid);
__global__ void SumReduction(double* idata, double* odata, int N);
__global__ void SumReductionStrided(double* idata, double* odata, int N, int Stride);
__global__ void SumReductionSmallData(double* idata, double* odata, int N);
__host__ void testreduction();
__host__ void TestMaxReduction();
__device__ void warpMinReduce(volatile double* sdata, int tid);
__device__ void warpMaxReduce(volatile double* sdata, int tid);
__global__ void MaxReduction(double* idata, double* odata, int N);
__global__ void MinReduction(double* idata, double* odata, int N);
__global__ void SumEnergyReduction(double* idata_M, double* idata_H, double* odata, int N);
__global__ void MaxReductionSmallN(double* idata, double* odata, int N);
__global__ void MinReductionSmallN(double* idata, double* odata, int N);
__global__ void SumReductionSmallN(double* idata, double* odata, int N);
__global__ void SumEnergyReductionSmallN(double* idata_M, double* idata_H, double* odata, int N);

__global__ void SumZeemanEnergyReduction_x(double* idata_M, double* idata_H, double* odata, int N);
__global__ void SumZeemanEnergyReduction_y(double* idata_M, double* idata_H, double* odata, int N);
__global__ void SumZeemanEnergyReduction_z(double* idata_M, double* idata_H, double* odata, int N);
__global__ void SumZeemanEnergyReductionSmallN_x(double* idata_M, double* idata_H, double* odata, int N);
__global__ void SumZeemanEnergyReductionSmallN_y(double* idata_M, double* idata_H, double* odata, int N);
__global__ void SumZeemanEnergyReductionSmallN_z(double* idata_M, double* idata_H, double* odata, int N);

unsigned int nextPow2(unsigned int x);
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks,
    int maxThreads, int& blocks, int& threads);
double Reduce_Sum(int  n, int  numThreads, int  numBlocks, int  maxThreads, int  maxBlocks,
    double* d_idata, double* d_odata);
double Reduce_Max(int  n, int  numThreads, int  numBlocks, int  maxThreads, int  maxBlocks,
    double* d_idata, double* d_odata);
double Reduce_Min(int  n, int  numThreads, int  numBlocks, int  maxThreads, int  maxBlocks,
    double* d_idata, double* d_odata);
double Reduce_Energy(int  n, int  numThreads, int  numBlocks, int  maxThreads, int  maxBlocks,
    double* M_idata, double* H_idata, double* E_odata);


double StepError_Reduction();
double MaxTorque_Reduction();
double MinTorque_Reduction();

#endif // !REDUCTION_FUNCTIONS_CUH

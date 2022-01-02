#include "ExchangeField.cuh"
#include "Device_Globals_Constants.cuh"
#include "Host_Globals.cuh"
#include "GlobalDefines.cuh"
#include "Array_Indexing_Functions.cuh"
#include <device_launch_parameters.h>
#include <helper_cuda.h>
__host__ void ExchangeStencilParameters()
{
    double dr_x_h = CELL_h * CELL_h,
        dr_y_h = CELLY_h * CELLY_h,
        dr_z_h = CELLZ_h * CELLZ_h;

    double Cx_h = (2.0 * A_ex_h) / (mu * MSAT_h * dr_x_h),
        Cy_h = (2.0 * A_ex_h) / (mu * MSAT_h * dr_y_h),
        Cz_h = (2.0 * A_ex_h) / (mu * MSAT_h * dr_z_h);

    checkCudaErrors(cudaMemcpyToSymbol(dr_x, &dr_x_h, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(dr_y, &dr_y_h, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(dr_z, &dr_z_h, sizeof(double)));

    checkCudaErrors(cudaMemcpyToSymbol(Cx, &Cx_h, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(Cy, &Cy_h, sizeof(double)));
    checkCudaErrors(cudaMemcpyToSymbol(Cz, &Cz_h, sizeof(double)));

    return;
}
__global__ void ExchangeField_FullGridBoundaries(MAG M, FIELD H)
{
    double Hex_x = 0.0, Hex_y = 0.0, Hex_z = 0.0;

    int dx1 = 0, dx2 = 0, dy1 = 0, dy2 = 0, dz1 = 0, dz2 = 0;

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;


    if ((i < NUM) && (j < NUMY) && (k < NUMZ))
    {
        if (M->Mat[ind(i, j, k)] == 0)
        {
            H->H_ex[find(i, j, k, 0)] = Hex_x;
            H->H_ex[find(i, j, k, 1)] = Hex_y;
            H->H_ex[find(i, j, k, 2)] = Hex_z;
        }
        else
        {
            dx1 = (M->Bd[bind(i, j, k, 0)] + M->Bd[bind(i, j, k, 0)] * M->Bd[bind(i, j, k, 0)]) / 2;
            dy1 = (M->Bd[bind(i, j, k, 1)] + M->Bd[bind(i, j, k, 1)] * M->Bd[bind(i, j, k, 1)]) / 2;
            dx2 = (M->Bd[bind(i, j, k, 0)] - M->Bd[bind(i, j, k, 0)] * M->Bd[bind(i, j, k, 0)]) / 2;
            dy2 = (M->Bd[bind(i, j, k, 1)] - M->Bd[bind(i, j, k, 1)] * M->Bd[bind(i, j, k, 1)]) / 2;
            dz1 = (M->Bd[bind(i, j, k, 2)] + M->Bd[bind(i, j, k, 2)] * M->Bd[bind(i, j, k, 2)]) / 2;
            dz2 = (M->Bd[bind(i, j, k, 2)] - M->Bd[bind(i, j, k, 2)] * M->Bd[bind(i, j, k, 2)]) / 2;


            Hex_x = Cx * (M->M[mind(0, i + 1 - dx1, j, k, 0)] + M->M[mind(0, i - 1 - dx2, j, k, 0)] - 2 * M->M[mind(0, i, j, k, 0)]);
            Hex_x += Cy * (M->M[mind(0, i, j + 1 - dy1, k, 0)] + M->M[mind(0, i, j - 1 - dy2, k, 0)] - 2 * M->M[mind(0, i, j, k, 0)]);

            Hex_y = Cx * (M->M[mind(0, i + 1 - dx1, j, k, 1)] + M->M[mind(0, i - 1 - dx2, j, k, 1)] - 2 * M->M[mind(0, i, j, k, 1)]);
            Hex_y += Cy * (M->M[mind(0, i, j + 1 - dy1, k, 1)] + M->M[mind(0, i, j - 1 - dy2, k, 1)] - 2 * M->M[mind(0, i, j, k, 1)]);

            Hex_z = Cx * (M->M[mind(0, i + 1 - dx1, j, k, 2)] + M->M[mind(0, i - 1 - dx2, j, k, 2)] - 2 * M->M[mind(0, i, j, k, 2)]);
            Hex_z += Cy * (M->M[mind(0, i, j + 1 - dy1, k, 2)] + M->M[mind(0, i, j - 1 - dy2, k, 2)] - 2 * M->M[mind(0, i, j, k, 2)]);

            if (NUMZ == 1)
            {
            }
            else
            {
                Hex_x += Cz * (M->M[mind(0, i, j, k + 1 - dz1, 0)] + M->M[mind(0, i, j, k - 1 - dz2, 0)] - 2 * M->M[mind(0, i, j, k, 0)]);

                Hex_y += Cz * (M->M[mind(0, i, j, k + 1 - dz1, 1)] + M->M[mind(0, i, j, k - 1 - dz2, 1)] - 2 * M->M[mind(0, i, j, k, 1)]);

                Hex_z += Cz * (M->M[mind(0, i, j, k + 1 - dz1, 2)] + M->M[mind(0, i, j, k - 1 - dz2, 2)] - 2 * M->M[mind(0, i, j, k, 2)]);
            }
            H->H_ex[find(i, j, k, 0)] = Hex_x;
            H->H_ex[find(i, j, k, 1)] = Hex_y;
            H->H_ex[find(i, j, k, 2)] = Hex_z;
        }
    }

    return;
}
__global__ void ExchangeField_FullGridBoundaries_PBC(MAG M, FIELD H)
{
    double H_array[3];
    H_array[0] = 0;
    H_array[1] = 0;
    H_array[2] = 0;

    int dx1 = 0, dx2 = 0, dy1 = 0, dy2 = 0, dz1 = 0, dz2 = 0;

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;


    if ((i < NUM) && (j <NUMY) && (k < NUMZ))
    {
        if (M->Mat[ind(i, j, k)] == 0)
        {
            H->H_ex[find(i, j, k, 0)] = 0.0;
            H->H_ex[find(i, j, k, 1)] = 0.0;
            H->H_ex[find(i, j, k, 2)] = 0.0;
        }
        else
        {
            dx1 = (M->Bd[bind(i, j, k, 0)] + M->Bd[bind(i, j, k, 0)] * M->Bd[bind(i, j, k, 0)]) / 2;
            dy1 = (M->Bd[bind(i, j, k, 1)] + M->Bd[bind(i, j, k, 1)] * M->Bd[bind(i, j, k, 1)]) / 2;
            dx2 = (M->Bd[bind(i, j, k, 0)] - M->Bd[bind(i, j, k, 0)] * M->Bd[bind(i, j, k, 0)]) / 2;
            dy2 = (M->Bd[bind(i, j, k, 1)] - M->Bd[bind(i, j, k, 1)] * M->Bd[bind(i, j, k, 1)]) / 2;
            dz1 = (M->Bd[bind(i, j, k, 2)] + M->Bd[bind(i, j, k, 2)] * M->Bd[bind(i, j, k, 2)]) / 2;
            dz2 = (M->Bd[bind(i, j, k, 2)] - M->Bd[bind(i, j, k, 2)] * M->Bd[bind(i, j, k, 2)]) / 2;

#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                H_array[n] = Cx * (M->M[mind(0, i + 1 - dx1, j, k, n)]
                    + M->M[mind(0, i - 1 - dx2, j, k, n)]
                    - 2 * M->M[mind(0, i, j, k, n)]);

                H_array[n] += Cy * (M->M[mind(0, i, PBCWrapRightY(j + 1 - dy1), k, n)]
                    + M->M[mind(0, i, PBCWrapLeftY(j - 1 - dy2), k, n)]
                    - 2 * M->M[mind(0, i, j, k, n)]);

                if (NUMZ == 1) {}
                else
                {
                    H_array[n] += Cz * (M->M[mind(0, i, j, k + 1 - dz1, n)]
                        + M->M[mind(0, i, j, k - 1 - dz2, n)]
                        - 2 * M->M[mind(0, i, j, k, n)]);
                }
            }

            H->H_ex[find(i, j, k, 0)] = H_array[0];
            H->H_ex[find(i, j, k, 1)] = H_array[1];
            H->H_ex[find(i, j, k, 2)] = H_array[2];
        }
    }

    return;
}
__global__ void ExchangeField_FullGridBoundaries_Mn1(MAG M, FIELD H)
{
    double Hex_x = 0.0, Hex_y = 0.0, Hex_z = 0.0;

    int dx1 = 0, dx2 = 0, dy1 = 0, dy2 = 0, dz1 = 0, dz2 = 0;

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

  
    if ((i < NUM) && (j < NUMY) && (k < NUMZ))
    {

        if (M->Mat[ind(i, j, k)] == 0)
        {
            H->H_ex[find(i, j, k, 0)] = Hex_x;
            H->H_ex[find(i, j, k, 1)] = Hex_y;
            H->H_ex[find(i, j, k, 2)] = Hex_z;
        }
        else
        {
            dx1 = (M->Bd[bind(i, j, k, 0)] + M->Bd[bind(i, j, k, 0)] * M->Bd[bind(i, j, k, 0)]) / 2;
            dy1 = (M->Bd[bind(i, j, k, 1)] + M->Bd[bind(i, j, k, 1)] * M->Bd[bind(i, j, k, 1)]) / 2;
            dx2 = (M->Bd[bind(i, j, k, 0)] - M->Bd[bind(i, j, k, 0)] * M->Bd[bind(i, j, k, 0)]) / 2;
            dy2 = (M->Bd[bind(i, j, k, 1)] - M->Bd[bind(i, j, k, 1)] * M->Bd[bind(i, j, k, 1)]) / 2;
            dz1 = (M->Bd[bind(i, j, k, 2)] + M->Bd[bind(i, j, k, 2)] * M->Bd[bind(i, j, k, 2)]) / 2;
            dz2 = (M->Bd[bind(i, j, k, 2)] - M->Bd[bind(i, j, k, 2)] * M->Bd[bind(i, j, k, 2)]) / 2;


            Hex_x = Cx * (M->M[mind(1, i + 1 - dx1, j, k, 0)] + M->M[mind(1, i - 1 - dx2, j, k, 0)] - 2 * M->M[mind(1, i, j, k, 0)]);
            Hex_x += Cy * (M->M[mind(1, i, j + 1 - dy1, k, 0)] + M->M[mind(1, i, j - 1 - dy2, k, 0)] - 2 * M->M[mind(1, i, j, k, 0)]);

            Hex_y = Cx * (M->M[mind(1, i + 1 - dx1, j, k, 1)] + M->M[mind(1, i - 1 - dx2, j, k, 1)] - 2 * M->M[mind(1, i, j, k, 1)]);
            Hex_y += Cy * (M->M[mind(1, i, j + 1 - dy1, k, 1)] + M->M[mind(1, i, j - 1 - dy2, k, 1)] - 2 * M->M[mind(1, i, j, k, 1)]);

            Hex_z = Cx * (M->M[mind(1, i + 1 - dx1, j, k, 2)] + M->M[mind(1, i - 1 - dx2, j, k, 2)] - 2 * M->M[mind(1, i, j, k, 2)]);
            Hex_z += Cy * (M->M[mind(1, i, j + 1 - dy1, k, 2)] + M->M[mind(1, i, j - 1 - dy2, k, 2)] - 2 * M->M[mind(1, i, j, k, 2)]);

            if (NUMZ == 1)
            {
            }
            else
            {
                Hex_x += Cz * (M->M[mind(1, i, j, k + 1 - dz1, 0)] + M->M[mind(1, i, j, k - 1 - dz2, 0)] - 2 * M->M[mind(1, i, j, k, 0)]);

                Hex_y += Cz * (M->M[mind(1, i, j, k + 1 - dz1, 1)] + M->M[mind(1, i, j, k - 1 - dz2, 1)] - 2 * M->M[mind(1, i, j, k, 1)]);

                Hex_z += Cz * (M->M[mind(1, i, j, k + 1 - dz1, 2)] + M->M[mind(1, i, j, k - 1 - dz2, 2)] - 2 * M->M[mind(1, i, j, k, 2)]);
            }
            H->H_ex[find(i, j, k, 0)] = Hex_x;
            H->H_ex[find(i, j, k, 1)] = Hex_y;
            H->H_ex[find(i, j, k, 2)] = Hex_z;
        }
    }

    return;
}
__global__ void ExchangeField_FullGridBoundaries_PBC_Mn1(MAG M, FIELD H)
{
    double H_array[3];
    H_array[0] = 0;
    H_array[1] = 0;
    H_array[2] = 0;

    int dx1 = 0, dx2 = 0, dy1 = 0, dy2 = 0, dz1 = 0, dz2 = 0;

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    if ((i < NUM) && (j < NUMY) && (k < NUMZ))
    {

        if (M->Mat[ind(i, j, k)] == 0)
        {
            H->H_ex[find(i, j, k, 0)] = 0.0;
            H->H_ex[find(i, j, k, 1)] = 0.0;
            H->H_ex[find(i, j, k, 2)] = 0.0;
        }
        else
        {
            dx1 = (M->Bd[bind(i, j, k, 0)] + M->Bd[bind(i, j, k, 0)] * M->Bd[bind(i, j, k, 0)]) / 2;
            dy1 = (M->Bd[bind(i, j, k, 1)] + M->Bd[bind(i, j, k, 1)] * M->Bd[bind(i, j, k, 1)]) / 2;
            dx2 = (M->Bd[bind(i, j, k, 0)] - M->Bd[bind(i, j, k, 0)] * M->Bd[bind(i, j, k, 0)]) / 2;
            dy2 = (M->Bd[bind(i, j, k, 1)] - M->Bd[bind(i, j, k, 1)] * M->Bd[bind(i, j, k, 1)]) / 2;
            dz1 = (M->Bd[bind(i, j, k, 2)] + M->Bd[bind(i, j, k, 2)] * M->Bd[bind(i, j, k, 2)]) / 2;
            dz2 = (M->Bd[bind(i, j, k, 2)] - M->Bd[bind(i, j, k, 2)] * M->Bd[bind(i, j, k, 2)]) / 2;

#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                H_array[n] = Cx * (M->M[mind(1, PBCWrapRightX(i + 1 - dx1), j, k, n)]
                    + M->M[mind(1, PBCWrapLeftX(i - 1 - dx2), j, k, n)]
                    - 2 * M->M[mind(1, i, j, k, n)]);

                H_array[n] += Cy * (M->M[mind(1, i, PBCWrapRightY(j + 1 - dy1), k, n)]
                    + M->M[mind(1, i, PBCWrapLeftY(j - 1 - dy2), k, n)]
                    - 2 * M->M[mind(1, i, j, k, n)]);

                if (NUMZ == 1) {}
                else
                {
                    H_array[n] += Cz * (M->M[mind(1, i, j, k + 1 - dz1, n)]
                        + M->M[mind(1, i, j, k - 1 - dz2, n)]
                        - 2 * M->M[mind(1, i, j, k, n)]);
                }
            }

            H->H_ex[find(i, j, k, 0)] = H_array[0];
            H->H_ex[find(i, j, k, 1)] = H_array[1];
            H->H_ex[find(i, j, k, 2)] = H_array[2];
        }
    }

    return;
}

__device__ int ExWrapX(int x)
{
    if (x >= NUM)
    {
        return NUM - 1;
    }
    if (x < 0)
    {
        return 0;
    }
    return x;
}
__device__ int ExWrapY(int y)
{
    if (y >= NUMY)
    {
        return NUMY - 1;
    }
    if (y < 0)
    {
        return 0;
    }
    return y;
}
__device__ int ExWrapZ(int z)
{
    if (z >= NUMZ)
    {
        return NUMZ - 1;
    }
    if (z < 0)
    {
        return 0;
    }
    return z;
}
__device__ int PBCWrapRightX(int index)
{
    if (PBC_x == 0)
    {
        return index;
    }

    if (index == (NUM - 1))
    {
        return 0;
    }
    return index;
}
__device__ int PBCWrapRightY(int index)
{
    if (PBC_y == 0)
    {
        return index;
    }
    if (index == (NUMY - 1))
    {
        return 0;
    }

    return index;
}
__device__ int PBCWrapLeftX(int index)
{
    if (PBC_x == 0)
    {
        return index;
    }

    if (index == 0)
    {
        return NUM - 1;
    }
    return index;
}
__device__ int PBCWrapLeftY(int index)
{
    if (PBC_y == 0)
    {
        return index;
    }
    if (index == 0)
    {
        return NUMY - 1;
    }

    return index;
}
#include "ZhangLi_STT.cuh"
#include "Array_Indexing_Functions.cuh"
#include "Device_Globals_Constants.cuh"
__global__ void Compute_ZhangLi_STT(MEMDATA DATA, FIELD H, MAG M)
{
    //local index
    int I = threadIdx.z;
    int J = threadIdx.y;
    int K = threadIdx.x;

    //global memory index
    int i = I + blockIdx.z * blockDim.z;
    int j = J + blockIdx.y * blockDim.y;
    int k = K + blockIdx.x * blockDim.x;

    double H_stt[3];
    H_stt[0] = 0.0,
    H_stt[1] = 0.0,
    H_stt[2] = 0.0;

    const double xi = SpinTransferTorqueParameters.Xi;
    const double P = SpinTransferTorqueParameters.P;
    const double jx = P * SpinTransferTorqueParameters.J[0];
    const double jy = P * SpinTransferTorqueParameters.J[1];
    const double jz = P * SpinTransferTorqueParameters.J[2];

    const double Ux = (-1.0/(1+xi*xi) ) * (P *B_m * jx )/(e_c * MSAT*1000.0);
    const double Uy = (-1.0 / (1 + xi * xi)) * (P * B_m * jy) / (e_c * MSAT * 1000.0);
    const double Uz = (-1.0 / (1 + xi * xi)) * (P * B_m * jz) / (e_c * MSAT * 1000.0);

    int index = M->Mat[ind(i, j, k)] - 1; //material index

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (SpinTransferTorqueParameters.handle == index)
        {
            //first derivative
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                H_stt[n] += (Ux /2.0/CELL) * (M->M[mind(0, hclampX(i + 1), j, k,n)]
                    - M->M[mind(0, lclampX(i - 1), j, k,n)]);

                H_stt[n] += (Uy /2.0/CELLY) * (M->M[mind(0, i, hclampY(j + 1), k,n)]
                    - M->M[mind(0, i, lclampY(j - 1), k,n)]);

                H_stt[n] += (Uz /2.0/CELLZ) * (M->M[mind(0, i, j, hclampZ(k + 1),n)]
                    - M->M[mind(0, i, j, lclampZ(k - 1),n)]);
            }
        }

        H->H_STT[find(i, j, k, 0)] = H_stt[0];
        H->H_STT[find(i, j, k, 1)] = H_stt[1];
        H->H_STT[find(i, j, k, 2)] = H_stt[2];
    }
}
__global__ void Compute_ZhangLi_STT_Mn1(MEMDATA DATA, FIELD H, MAG M)
{
    //local index
    int I = threadIdx.z;
    int J = threadIdx.y;
    int K = threadIdx.x;

    //global memory index
    int i = I + blockIdx.z * blockDim.z;
    int j = J + blockIdx.y * blockDim.y;
    int k = K + blockIdx.x * blockDim.x;

    double H_stt[3];
    H_stt[0] = 0.0,
    H_stt[1] = 0.0,
    H_stt[2] = 0.0;

    const double xi = SpinTransferTorqueParameters.Xi;
    const double P = SpinTransferTorqueParameters.P;
    const double jx = P * SpinTransferTorqueParameters.J[0];
    const double jy = P * SpinTransferTorqueParameters.J[1];
    const double jz = P * SpinTransferTorqueParameters.J[2];

    const double Ux = (-1.0 / (1 + xi * xi)) * (P * B_m * jx) / (e_c * MSAT * 1000.0); // m/s
    const double Uy = (-1.0 / (1 + xi * xi)) * (P * B_m * jy) / (e_c * MSAT * 1000.0); // m/s
    const double Uz = (-1.0 / (1 + xi * xi)) * (P * B_m * jz) / (e_c * MSAT * 1000.0); // m/s

    int index = M->Mat[ind(i, j, k)]; //material handle

    if (i < NUM && j < NUMY && k < NUMZ)
    {
        if (SpinTransferTorqueParameters.handle == index)
        {
            //first derivative
#pragma unroll DIM
            for (int n = 0; n < DIM; n++)
            {
                H_stt[n] += (Ux/2.0/CELL) * (M->M[mind(1, hclampX(i + 1), j, k, n)]
                    - M->M[mind(1, lclampX(i - 1), j, k, n)]);

                H_stt[n] += (Uy/2.0/CELLY) * (M->M[mind(1, i, hclampY(j + 1), k, n)]
                    - M->M[mind(1, i, lclampY(j - 1), k, n)]);

                H_stt[n] += (Uz /2.0/CELLZ) * (M->M[mind(1, i, j, hclampZ(k + 1), n)]
                    - M->M[mind(1, i, j, lclampZ(k - 1), n)]);
            }
        }

        H->H_STT[find(i, j, k, 0)] = H_stt[0];
        H->H_STT[find(i, j, k, 1)] = H_stt[1];
        H->H_STT[find(i, j, k, 2)] = H_stt[2];
    }
}


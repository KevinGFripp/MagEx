#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#include "DataTypes.cuh"
#include "Device_Globals_Constants.cuh"
#include "Host_Globals.cuh"
#include "Array_Indexing_Functions.cuh"
#include "GlobalDefines.cuh"
#include <helper_cuda.h>
#include "Pointer_Functions.cuh"

#ifndef DEMAGNETISINGTENSOR_FUNCTIONS_CUH
#define DEMAGNETISINGTENSOR_FUNCTIONS_CUH
__host__  void DemagFieldInitialise(FIELD H);
__host__ void InitialiseDemagTensor3D(MEMDATA DATA);
__host__ void ComputeDemagTensorNewell_GQ(MAG M, MEMDATA DATA);
__host__ void ComputeDemagTensorNewell3D_PBC(MAG M, MEMDATA DATA);
__host__ double NewellF_GQ(double X, double Y, double Z, double Dx, double Dy, double Dz,
    double y, double z, double yprime, double zprime);
__host__ double NewellG_GQ(double X, double Y, double Z, double Dx, double Dy, double Dz,
    double y, double z, double zprime, double xprime);
//__host__ double Nxx_GQ_5(double X, double Y, double Z, double Dx, double Dy, double Dz);
__host__ double inline Nxx_GQ_5(double X, double Y, double Z, double Dx, double Dy, double Dz)
{
    const int N = 5;
    double w[5];
    double Xi[5];

    double Coefficient = pow(0.5, 4);
    double Offset = 0.5;

    w[0] = 0.5688888888888889;
    w[1] = 0.4786286704993665;
    w[2] = 0.4786286704993665;
    w[3] = 0.2369268850561891;
    w[4] = 0.2369268850561891;

    Xi[0] = 0;
    Xi[1] = -0.5384693101056831;
    Xi[2] = 0.5384693101056831;
    Xi[3] = -0.9061798459386640;
    Xi[4] = 0.9061798459386640;

    double y;
    double z;
    double yp;
    double zp;

    double I1 = 0;
    double I2 = 0;
    double I3 = 0;

    for (int dyp = 0; dyp < N; dyp++)
    {

        for (int dzp = 0; dzp < N; dzp++)
        {

            for (int dy = 0; dy < N; dy++)
            {

                for (int dz = 0; dz < N; dz++)
                {
                    y = 0.5 * Xi[dy] + Offset;
                    yp = 0.5 * Xi[dyp] + Offset;
                    z = 0.5 * Xi[dz] + Offset;
                    zp = 0.5 * Xi[dzp] + Offset;

                    I1 += Coefficient * w[dz] * w[dy] * w[dzp] * w[dyp] * NewellF_GQ(X, Y, Z, Dx, Dy, Dz, y, z, yp, zp);
                    I2 += Coefficient * w[dz] * w[dy] * w[dzp] * w[dyp] * NewellF_GQ(X + Dx, Y, Z, Dx, Dy, Dz, y, z, yp, zp);
                    I3 += Coefficient * w[dz] * w[dy] * w[dzp] * w[dyp] * NewellF_GQ(X - Dx, Y, Z, Dx, Dy, Dz, y, z, yp, zp);

                }

            }

        }

    }

    return (2 * I1 - I2 - I3);
}
//__host__ double Nxx_GQ_7(double X, double Y, double Z, double Dx, double Dy, double Dz);
 __host__ double inline Nxx_GQ_7(double X, double Y, double Z, double Dx, double Dy, double Dz)
{
    const int N = 7;
    double w[7];
    double Xi[7];

    double Coefficient = pow(0.5, 4);
    double Offset = 0.5;

    w[0] = 0.4179591836734694, Xi[0] = 0.0000000000000000;
    w[1] = 0.3818300505051189, Xi[1] = 0.4058451513773972;
    w[2] = 0.3818300505051189, Xi[2] = -0.4058451513773972;
    w[3] = 0.2797053914892766, Xi[3] = -0.7415311855993945;
    w[4] = 0.2797053914892766, Xi[4] = 0.7415311855993945;
    w[5] = 0.1294849661688697, Xi[5] = -0.9491079123427585;
    w[6] = 0.1294849661688697, Xi[6] = 0.9491079123427585;

    double y;
    double z;
    double yp;
    double zp;

    double I1 = 0;
    double I2 = 0;
    double I3 = 0;


    for (int dyp = 0; dyp < N; dyp++)
    {

        for (int dzp = 0; dzp < N; dzp++)
        {

            for (int dy = 0; dy < N; dy++)
            {

                for (int dz = 0; dz < N; dz++)
                {
                    y = 0.5 * Xi[dy] + Offset;
                    yp = 0.5 * Xi[dyp] + Offset;
                    z = 0.5 * Xi[dz] + Offset;
                    zp = 0.5 * Xi[dzp] + Offset;

                    I1 += Coefficient * w[dz] * w[dy] * w[dzp] * w[dyp] * NewellF_GQ(X, Y, Z, Dx, Dy, Dz, y, z, yp, zp);
                    I2 += Coefficient * w[dz] * w[dy] * w[dzp] * w[dyp] * NewellF_GQ(X + Dx, Y, Z, Dx, Dy, Dz, y, z, yp, zp);
                    I3 += Coefficient * w[dz] * w[dy] * w[dzp] * w[dyp] * NewellF_GQ(X - Dx, Y, Z, Dx, Dy, Dz, y, z, yp, zp);

                }

            }

        }

    }

    return (2 * I1 - I2 - I3);
}
//__host__ double Nxy_GQ_5(double X, double Y, double Z, double Dx, double Dy, double Dz);
__host__ double inline Nxy_GQ_5(double X, double Y, double Z, double Dx, double Dy, double Dz)
{
    const int N = 5;
    double w[5];
    double Xi[5];

    double Coefficient = pow(0.5, 4);
    double Offset = 0.5;

    w[0] = 0.5688888888888889;
    w[1] = 0.4786286704993665;
    w[2] = 0.4786286704993665;
    w[3] = 0.2369268850561891;
    w[4] = 0.2369268850561891;

    Xi[0] = 0;
    Xi[1] = -0.5384693101056831;
    Xi[2] = 0.5384693101056831;
    Xi[3] = -0.9061798459386640;
    Xi[4] = 0.9061798459386640;


    double y;
    double z;
    double zp;
    double xp;

    double I1 = 0;
    double I2 = 0;
    double I3 = 0;
    double I4 = 0;

    for (int dxp = 0; dxp < N; dxp++)
    {
        for (int dzp = 0; dzp < N; dzp++)
        {
            for (int dz = 0; dz < N; dz++)
            {
                for (int dy = 0; dy < N; dy++)
                {
                    y = 0.5 * Xi[dy] + Offset;
                    xp = 0.5 * Xi[dxp] + Offset;
                    z = 0.5 * Xi[dz] + Offset;
                    zp = 0.5 * Xi[dzp] + Offset;

                    I1 += Coefficient * w[dy] * w[dz] * w[dzp] * w[dxp] * NewellG_GQ(X, Y, Z, Dx, Dy, Dz, y, z, zp, xp);
                    I2 += Coefficient * w[dy] * w[dz] * w[dzp] * w[dxp] * NewellG_GQ(X - Dx, Y, Z, Dx, Dy, Dz, y, z, zp, xp);
                    I3 += Coefficient * w[dy] * w[dz] * w[dzp] * w[dxp] * NewellG_GQ(X, Y + Dy, Z, Dx, Dy, Dz, y, z, zp, xp);
                    I4 += Coefficient * w[dy] * w[dz] * w[dzp] * w[dxp] * NewellG_GQ(X - Dx, Y + Dy, Z, Dx, Dy, Dz, y, z, zp, xp);


                }

            }

        }

    }

    return (I1 - I2 - I3 + I4);
}
//__host__ double Nxy_GQ_7(double X, double Y, double Z, double Dx, double Dy, double Dz);
__host__ double inline Nxy_GQ_7(double X, double Y, double Z, double Dx, double Dy, double Dz)
{
    const int N = 7;
    double w[7];
    double Xi[7];

    double Coefficient = pow(0.5, 4);
    double Offset = 0.5;

    w[0] = 0.4179591836734694, Xi[0] = 0.0000000000000000;
    w[1] = 0.3818300505051189, Xi[1] = 0.4058451513773972;
    w[2] = 0.3818300505051189, Xi[2] = -0.4058451513773972;
    w[3] = 0.2797053914892766, Xi[3] = -0.7415311855993945;
    w[4] = 0.2797053914892766, Xi[4] = 0.7415311855993945;
    w[5] = 0.1294849661688697, Xi[5] = -0.9491079123427585;
    w[6] = 0.1294849661688697, Xi[6] = 0.9491079123427585;

    double y;
    double z;
    double zp;
    double xp;

    double I1 = 0;
    double I2 = 0;
    double I3 = 0;
    double I4 = 0;


    for (int dxp = 0; dxp < N; dxp++)
    {

        for (int dzp = 0; dzp < N; dzp++)
        {

            for (int dz = 0; dz < N; dz++)
            {

                for (int dy = 0; dy < N; dy++)
                {
                    y = 0.5 * Xi[dy] + Offset;
                    xp = 0.5 * Xi[dxp] + Offset;
                    z = 0.5 * Xi[dz] + Offset;
                    zp = 0.5 * Xi[dzp] + Offset;

                    I1 += Coefficient * w[dy] * w[dz] * w[dzp] * w[dxp] * NewellG_GQ(X, Y, Z, Dx, Dy, Dz, y, z, zp, xp);
                    I2 += Coefficient * w[dy] * w[dz] * w[dzp] * w[dxp] * NewellG_GQ(X - Dx, Y, Z, Dx, Dy, Dz, y, z, zp, xp);
                    I3 += Coefficient * w[dy] * w[dz] * w[dzp] * w[dxp] * NewellG_GQ(X, Y + Dy, Z, Dx, Dy, Dz, y, z, zp, xp);
                    I4 += Coefficient * w[dy] * w[dz] * w[dzp] * w[dxp] * NewellG_GQ(X - Dx, Y + Dy, Z, Dx, Dy, Dz, y, z, zp, xp);
                }

            }

        }

    }

    return (I1 - I2 - I3 + I4);
}
__host__ double NxxInt(double X, double Y, double Z, double Dx, double Dy, double Dz);
__host__ double NxyInt(double X, double Y, double Z, double Dx, double Dy, double Dz);
__host__ double NewellF(double x, double y, double z);
__host__ double NewellG(double x, double y, double z);
__host__ double SelfDemagNx(double x, double y, double z);
__host__ double AccurateSum(int n, double* arr);
__host__ double DemagAsymptoticDiag(double x, double y, double z, double hx, double hy, double hz);
__host__ double DemagAsymptoticOffDiag(double x, double y, double z, double hx, double hy, double hz);
__host__ double PointDipole_Nxx(double x, double y, double z);
__host__ double PointDipole_Nxy(double x, double y, double z);
__host__ int DemagTensorPBCWrapNx(int I);
__host__ int DemagTensorPBCWrapNy(int J);
__host__ int DemagTensorWrapNx(int I);
__host__ int DemagTensorWrapNy(int J);
__host__ int DemagTensorWrapNz(int K);
__global__ void DemagTensorStoreFirstOctant_OffDiagonals(MEMDATA DATA);
__global__ void DemagTensorStoreFirstOctant_Diagonals(MEMDATA DATA);

#endif
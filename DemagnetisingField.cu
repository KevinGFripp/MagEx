#include "DemagnetisingField.cuh"
#include "Device_Globals_Constants.cuh"
#include "Array_Indexing_Functions.cuh"
#include <device_launch_parameters.h>
#include <cufft.h>

__global__ void MagnetisationFFTCompute3DInitialised(MEMDATA DATA, MAG M)
{
    double Scale = -MSAT / (FFT_NORM);

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double  mxdat;
    double  mydat;
    double  mzdat;

    if ((i < NUM) && (j < NUMY) && (k < NUMZ))
    {
        mxdat = M->M[mind(0, i, j, k, 0)];
        mydat = M->M[mind(0, i, j, k, 1)];
        mzdat = M->M[mind(0, i, j, k, 2)];

        DATA->Outx_d[FFTind(i, j, k)] = Scale * mxdat;
        DATA->Outy_d[FFTind(i, j, k)] = Scale * mydat;
        DATA->Outz_d[FFTind(i, j, k)] = Scale * mzdat;

    }
    else if ((i < PADNUM) && (j < PADNUMY) && (k < PADNUMZ))
    {
        DATA->Outx_d[FFTind(i, j, k)] = 0.0;
        DATA->Outy_d[FFTind(i, j, k)] = 0.0;
        DATA->Outz_d[FFTind(i, j, k)] = 0.0;

    }
    return;
}
__global__ void MagnetisationFFTCompute3DInitialised_SinglePrecision_R2C(MEMDATA DATA, MAG M)
{
    double Scale = -MSAT / (FFT_NORM);
   
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double  mxdat;
    double  mydat;
    double  mzdat;

    if ((i < NUM) && (j < NUMY) && (k < NUMZ))
    {
        mxdat = M->M[mind(0, i, j, k, 0)];
        mydat = M->M[mind(0, i, j, k, 1)];
        mzdat = M->M[mind(0, i, j, k, 2)];

        DATA->Outx[FFTind(i, j, k)] = Scale * mxdat;
        DATA->Outy[FFTind(i, j, k)] = Scale * mydat;
        DATA->Outz[FFTind(i, j, k)] = Scale * mzdat;
       
    }
    else if ((i < PADNUM) && (j < PADNUMY) && (k < PADNUMZ))
    {
        DATA->Outx[FFTind(i, j, k)] = 0.0;
        DATA->Outy[FFTind(i, j, k)] = 0.0;
        DATA->Outz[FFTind(i, j, k)] = 0.0;

    }
    return;
}
__global__ void MagnetisationFFTCompute3DInitialised_Mn1(MEMDATA DATA, MAG M)
{
    double Scale = -MSAT / (FFT_NORM);

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double  mxdat;
    double  mydat;
    double  mzdat;

    if ((i < NUM) && (j < NUMY) && (k < NUMZ))
    {
        mxdat = M->M[mind(1, i, j, k, 0)];
        mydat = M->M[mind(1, i, j, k, 1)];
        mzdat = M->M[mind(1, i, j, k, 2)];

        DATA->Outx_d[FFTind(i, j, k)] = Scale * mxdat;
        DATA->Outy_d[FFTind(i, j, k)] = Scale * mydat;
        DATA->Outz_d[FFTind(i, j, k)] = Scale * mzdat;

    }
    else if ((i < PADNUM) && (j < PADNUMY) && (k < PADNUMZ))
    {
        DATA->Outx_d[FFTind(i, j, k)] = 0.0;
        DATA->Outy_d[FFTind(i, j, k)] = 0.0;
        DATA->Outz_d[FFTind(i, j, k)] = 0.0;

    }
    return;
}
__global__ void MagnetisationFFTCompute3DInitialised_Mn1_SinglePrecision_R2C(MEMDATA DATA, MAG M)
{

   // int FFT_N = (2 - PBC_x) * (2 - PBC_y) * 2 * NUM * NUMY * NUMZ;

    double Scale = -MSAT / (FFT_NORM);

  
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    double  mxdat;
    double  mydat;
    double  mzdat;


    if ((i < NUM) && (j < NUMY) && (k < NUMZ))
    {
        mxdat = M->M[mind(1, i, j, k, 0)];
        mydat = M->M[mind(1, i, j, k, 1)];
        mzdat = M->M[mind(1, i, j, k, 2)];

        DATA->Outx[FFTind(i, j, k)] = Scale * mxdat;
        DATA->Outy[FFTind(i, j, k)] = Scale * mydat;
        DATA->Outz[FFTind(i, j, k)] = Scale * mzdat;     
    }
    else if ((i < PADNUM) && (j < PADNUMY) && (k < PADNUMZ))
    {
        DATA->Outx[FFTind(i, j, k)] = 0.0;
        DATA->Outy[FFTind(i, j, k)] = 0.0;
        DATA->Outz[FFTind(i, j, k)] = 0.0;

    }
   /* else if ((i < (2-PBC_x)*NUM) && (j < (2-PBC_y)*NUMY) && (k < 2*NUMZ))
    {
        DATA->Outx[FFTind(i, j, k)] = 0.0;
        DATA->Outy[FFTind(i, j, k)] = 0.0;
        DATA->Outz[FFTind(i, j, k)] = 0.0;    
    }*/
    return;
}
__global__ void ComputeDemagField3DConvolution_Symmetries(MEMDATA DATA)
{
    /* Discrete convolution Of N and M*/
     //HxFFT = NxxFFT*MxFFT +NxyFFT*MyFFT +NxzFFT*MzFFT
    // HyFFT = NxyFFT*MxFFT +NyyFFT*MyFFT +NyzFFT*MzFFT
    // HzFFT = NxzFFT*MxFFT +NyzFFT*MyFFT +NzzFFT*MzFFT
    // NFFT and MFFT are complex, therefore => NxxFFT*MxFFT = (NxxFFT.re*MxFFT.re -NxxFFT.im*MxFFT.im)
    // +i(NxxFFT.im*MxFFT.re + NxxFFT.re*MxFFT.im)*/

     //Reconstruct missing demag elements from even-odd symmetries of tensor
     //Diagonal terms even in i,j,k
     //off-diagonal terms odd in i,j and even in k
     //map i>Nx with i % (NUM-1) 
     //compute G = sign((NUM-1)-i)*sign((NUMY-1)-j), G is symmetry factor

     //FFT of purely real demag tensor is real
     //-1 factor applied to odd symmetry of real part of FFT


    int tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;


    if (i < PADNUM && j < (PADNUMY) && k < (PADNUMZ / 2 + 1))
    {
        int index = FFTind_R2C(i, j, k);
        double Nxx = 0., Nxy = 0., Nxz = 0., Nyz = 0., Nyy = 0., Nzz = 0.;
        cufftDoubleComplex Mx, My, Mz;
        double Hconv_r;
        double Hconv_i;


        Mx.x = DATA->xFFT[index][0], Mx.y = DATA->xFFT[index][1];
        My.x = DATA->yFFT[index][0], My.y = DATA->yFFT[index][1];
        Mz.x = DATA->zFFT[index][0], Mz.y = DATA->zFFT[index][1];


        int dx = i, dy = j, dz = k;

        if (i > (NUM))
        {
            dx = 2 * NUM - i;
        }

        if (j > (NUMY))
        {
            dy = 2 * NUMY - j;
        }

        if (k > (NUMZ))
        {
            dz = 2 * NUMZ - k;
        }

        int N_index = dind(dx, dy, dz);

        Nxx = DATA->Nxx[N_index];
        Nyy = DATA->Nyy[N_index];
        Nzz = DATA->Nzz[N_index];

        int G;
        G = Sign((NUM - i)) * Sign(NUMY - j);
        Nxy = G * DATA->Nxy[N_index];

        G = Sign((NUM - i)) * Sign(NUMZ - k);
        Nxz = G * DATA->Nxz[N_index];

        G = Sign((NUMY - j)) * Sign(NUMZ - k);
        Nyz = G * DATA->Nyz[N_index];


        Hconv_r = Nxx * Mx.x + Nxy * My.x + Nxz * Mz.x;
        Hconv_i = Nxx * Mx.y + Nxy * My.y + Nxz * Mz.y;

        DATA->xFFT[index][0] = Hconv_r;
        DATA->xFFT[index][1] = Hconv_i;


        Hconv_r = Nxy * Mx.x + Nyy * My.x + Nyz * Mz.x;
        Hconv_i = Nxy * Mx.y + Nyy * My.y + Nyz * Mz.y;

        DATA->yFFT[index][0] =Hconv_r;
        DATA->yFFT[index][1] =Hconv_i;

        Hconv_r = Nxz * Mx.x + Nyz * My.x + Nzz * Mz.x;
        Hconv_i = Nxz * Mx.y + Nyz * My.y + Nzz * Mz.y;

        DATA->zFFT[index][0] =Hconv_r;
        DATA->zFFT[index][1] =Hconv_i;
    }
}
__global__ void ComputeDemagField3DConvolution_Symmetries_SinglePrecision_R2C(MEMDATA DATA)
{
    /* Discrete convolution Of N and M*/
    //HxFFT = NxxFFT*MxFFT +NxyFFT*MyFFT +NxzFFT*MzFFT
   // HyFFT = NxyFFT*MxFFT +NyyFFT*MyFFT +NyzFFT*MzFFT
   // HzFFT = NxzFFT*MxFFT +NyzFFT*MyFFT +NzzFFT*MzFFT
   // NFFT and MFFT are complex, therefore => NxxFFT*MxFFT = (NxxFFT.re*MxFFT.re -NxxFFT.im*MxFFT.im)
   // +i(NxxFFT.im*MxFFT.re + NxxFFT.re*MxFFT.im)*/

    //Reconstruct missing demag elements from even-odd symmetries of tensor
    //Diagonal terms even in i,j,k
    //off-diagonal terms odd in i,j and even in k
    //map i>Nx with i % (NUM-1) 
    //compute G = sign((NUM-1)-i)*sign((NUMY-1)-j), G is symmetry factor

    //FFT of purely real demag tensor is real
    //-1 factor applied to odd symmetry of real part of FFT


    int tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

   // if (i < ((2-PBC_x)*NUM) && j < ((2-PBC_y)*NUMY) && k < (NUMZ + 1))
    if (i < PADNUM && j < (PADNUMY) && k < (PADNUMZ/2 + 1))
    {
        int index = FFTind_R2C(i, j, k);
            double Nxx = 0., Nxy = 0., Nxz = 0., Nyz = 0., Nyy = 0., Nzz = 0.;
            cufftComplex Mx, My, Mz;
            double Hconv_r;
            double Hconv_i;


            Mx.x = DATA->xFFT_s[index].x, Mx.y = DATA->xFFT_s[index].y;
            My.x = DATA->yFFT_s[index].x, My.y = DATA->yFFT_s[index].y;
            Mz.x = DATA->zFFT_s[index].x, Mz.y = DATA->zFFT_s[index].y;


        int dx = i, dy = j, dz = k;

        if (i > (NUM))
        {
            dx = 2 * NUM - i;
        }

        if (j > (NUMY))
        {
            dy = 2 * NUMY - j;
        }

        if (k > (NUMZ))
        {
            dz = 2 * NUMZ - k;
        }

        int N_index = dind(dx, dy, dz);

        Nxx = DATA->Nxx[N_index];
        Nyy = DATA->Nyy[N_index];
        Nzz = DATA->Nzz[N_index];

        int G;
        G = Sign((NUM - i)) * Sign(NUMY - j);
        Nxy = G * DATA->Nxy[N_index];

        G = Sign((NUM - i)) * Sign(NUMZ - k);
        Nxz = G * DATA->Nxz[N_index];

        G = Sign((NUMY - j)) * Sign(NUMZ - k);
        Nyz = G * DATA->Nyz[N_index];


        Hconv_r = Nxx * Mx.x + Nxy * My.x + Nxz * Mz.x;
        Hconv_i = Nxx * Mx.y + Nxy * My.y + Nxz * Mz.y;

        DATA->xFFT_s[index].x = (float)Hconv_r;
        DATA->xFFT_s[index].y = (float)Hconv_i;


        Hconv_r = Nxy * (double)Mx.x + Nyy * (double)My.x + Nyz * (double)Mz.x;
        Hconv_i = Nxy * (double)Mx.y + Nyy * (double)My.y + Nyz * (double)Mz.y;

        DATA->yFFT_s[index].x = (float)Hconv_r;
        DATA->yFFT_s[index].y = (float)Hconv_i;

        Hconv_r = Nxz * (double)Mx.x + Nyz * (double)My.x + Nzz * (double)Mz.x;
        Hconv_i = Nxz * (double)Mx.y + Nyz * (double)My.y + Nzz * (double)Mz.y;

        DATA->zFFT_s[index].x = (float)Hconv_r;
        DATA->zFFT_s[index].y = (float)Hconv_i;
    }

}
#include <cuda_runtime.h>
#include "DataTypes.cuh"
#include "GlobalDefines.cuh"
#ifndef _DEVICE_GLOBALS_CONSTANTS_CUH_
#define _DEVICE_GLOBALS_CONSTANTS_CUH_

extern __constant__ int NUM;
extern __constant__ int NUMY;
extern __constant__ int NUMZ;

extern __constant__ int FFT_NORM;

//FFT Pad size
extern __constant__ int PADNUM;
extern __constant__ int PADNUMY;
extern __constant__ int PADNUMZ;

extern __constant__ double CELL;
extern __constant__ double CELLY;
extern __constant__ double CELLZ;

extern __constant__ int PBC_x;
extern __constant__ int PBC_y;

extern __constant__ double MSAT;
extern __constant__ double alpha;
extern __constant__ double A_ex;

extern __constant__ double K_UANIS;
extern __constant__ int UniAnisotropy;
extern __constant__ int Uanisx;
extern __constant__ int Uanisy;
extern __constant__ int Uanisz;

extern __constant__ int ExternalField;
extern __constant__ int BiasField;
extern __constant__ double AMPx;
extern __constant__ double AMPy;
extern __constant__ double AMPz;

extern __constant__ int SpinTransferTorque;

extern __constant__ int METHOD;
extern __constant__ double AbsTol;

//Exchange Stencil
extern __constant__ double dr_x;
extern __constant__ double dr_y;
extern __constant__ double dr_z;

extern __constant__ double Cx;
extern __constant__ double Cy;
extern __constant__ double Cz;


extern __constant__ double TIME_d;
extern __constant__ double h_d;
extern __constant__ double t_d;
extern __constant__ int ResetFlag_d;

extern __device__ int SHAREDMEMDIM_x;
extern __device__ int SHAREDMEMDIM_y;
extern __device__ int SHAREDMEMDIM_z;

//Viewer Globals
extern __constant__ int Viewer_component;
extern __constant__ int Viewer_zslice;
extern __constant__ int Viewer_contrast;

//Materials

extern __constant__ MaterialProperties ArrayOfMaterials[MAXMATERIALNUM];

//Spin Transfer Torque
extern __constant__ STTParameters SpinTransferTorqueParameters;

#endif // !_DEVICE_GLOBALS_CONSTANTS_CUH_

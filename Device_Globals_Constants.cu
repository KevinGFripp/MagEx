#include "Device_Globals_Constants.cuh"

__constant__ int NUM;
__constant__ int NUMY;
__constant__ int NUMZ;

__constant__ int FFT_NORM;

//FFT Pad size
__constant__ int PADNUM;
__constant__ int PADNUMY;
__constant__ int PADNUMZ;

__constant__ double CELL;
__constant__ double CELLY;
__constant__ double CELLZ;

__constant__ int PBC_x;
__constant__ int PBC_y;

__constant__ double MSAT;
__constant__ double alpha;
__constant__ double A_ex;

__constant__ double K_UANIS;
__constant__ int UniAnisotropy;
__constant__ int Uanisx;
__constant__ int Uanisy;
__constant__ int Uanisz;

__constant__ int ExternalField;
__constant__ int BiasField;
__constant__ double AMPx;
__constant__ double AMPy;
__constant__ double AMPz;

__constant__ int SpinTransferTorque;

__constant__ int METHOD;
__constant__ double AbsTol;

//Exchange Stencil
__constant__ double dr_x;
__constant__ double dr_y;
__constant__ double dr_z;

__constant__ double Cx;
__constant__ double Cy;
__constant__ double Cz;


__constant__ double TIME_d;
__constant__ double h_d;
__constant__ double t_d;
__constant__ int ResetFlag_d;

__device__ int SHAREDMEMDIM_x;
__device__ int SHAREDMEMDIM_y;
__device__ int SHAREDMEMDIM_z;

//Viewer Globals
__constant__ int Viewer_component = 0;
__constant__ int Viewer_zslice = 0;
__constant__ int Viewer_contrast = 1;

//Materials
__constant__ MaterialProperties ArrayOfMaterials[MAXMATERIALNUM];

//Spin Transfer Torque
__constant__ STTParameters SpinTransferTorqueParameters;

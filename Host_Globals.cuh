#include <cuda_runtime.h>
#include "DataTypes.cuh"
#include "GlobalDefines.cuh"
#ifndef HOST_GLOBALS_H
#define HOST_GLOBALS_H

extern int SHAREDMEMSIZE_h;
extern dim3 NumberofBlocks;
extern dim3 NumberofThreads;

extern dim3 NumberofBlocksReduction; //Reduction kernels, power of 2
extern dim3 NumberofThreadsReduction;

extern dim3 NumberofBlocksIntegrator; //power of 2 to facilitate reductions
extern dim3 NumberofThreadsIntegrator;

extern dim3 NumberofBlocksPadded; // for padded data
extern dim3 NumberofThreadsPadded;

extern double TIME;
extern double t_h;

extern int NUM_h;
extern int NUMY_h;
extern int NUMZ_h;

//FFT pad size
extern int PADNUM_h;
extern int PADNUMY_h;
extern int PADNUMZ_h;

extern double CELL_h;
extern double CELLY_h;
extern double CELLZ_h;

extern bool IsPBCEnabled;
extern int PBC_x_h;
extern int PBC_y_h;
extern int PBC_x_images_h;
extern int PBC_y_images_h;

extern bool NoDemag;
extern bool NoExchange;
extern bool FixedTimeStep;
extern double MSAT_h;
extern double alpha_h;
extern double A_ex_h;
//anisotropy
extern double K_UANIS_h;
extern int UniAnisotropy_h;
extern int Uanisx_h;
extern int Uanisy_h;
extern int Uanisz_h;
//Sampling
extern bool StrictSampling;
extern double Sampling_Period;
extern double GUI_Sampling_Period;
extern int LoggingStage;
//

//external field
extern int ExternalField_h;
extern int BiasField_h;
extern double AMPx_h;
extern double AMPy_h;
extern double AMPz_h;
//

//STT
extern int SpinTransferTorque_h;

//Integrator parameters
extern int METHOD_h;
extern double h_h;
extern double RelTol;
extern double AbsTol_h;
//

//Viewer Host Side Globals
extern bool OpenGlInitialised;
extern bool BufferCreated;
extern bool BufferCreated_xz;
extern bool RenderWindowCreated;
extern int Viewer_zslice_host;
extern int Viewer_contrast_host;
extern bool ViewXZ;
extern bool ViewMag;
extern bool ViewDemag;
extern bool ViewExch;
extern bool ViewTorque;

extern DeviceStructOfPointers DEVICE_PTR_STRUCT;

//Materials
extern int NumberOfMaterials;

extern MaterialProperties ArrayOfMaterials_h[MAXMATERIALNUM];

//FFT Precision
extern bool UseSinglePrecision;
#endif // !HOST_GLOBALS_H

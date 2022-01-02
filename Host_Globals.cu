#include "Host_Globals.cuh"

dim3 NumberofBlocks;
dim3 NumberofThreads;

dim3 NumberofBlocksReduction; //power of 2
dim3 NumberofThreadsReduction;

dim3 NumberofBlocksIntegrator; //power of 2
dim3 NumberofThreadsIntegrator;

dim3 NumberofBlocksPadded; // for FFT padded data
dim3 NumberofThreadsPadded;

int SHAREDMEMSIZE_h = 0;
double TIME = 1.0;
double t_h = 0.0;

int NUM_h = 1;
int NUMY_h = 1;
int NUMZ_h = 1;

//FFT pad size
int PADNUM_h;
int PADNUMY_h;
int PADNUMZ_h;

double CELL_h = 4.0;
double CELLY_h = 4.0;
double CELLZ_h = 4.0;

bool IsPBCEnabled = false;
int PBC_x_h = 0;
int PBC_y_h = 0;
int PBC_x_images_h = 1;
int PBC_y_images_h = 1;

bool NoDemag = false;
bool NoExchange = false;
bool FixedTimeStep = false;
double MSAT_h = 0.0;
double alpha_h = 0.0;
double A_ex_h = 0.0;

//anisotropy
double K_UANIS_h = 0.0;
int UniAnisotropy_h = 0;
int Uanisx_h = 0;
int Uanisy_h = 0;
int Uanisz_h = 0;
//


//Sampling
bool StrictSampling = false;
double Sampling_Period = 0;
double GUI_Sampling_Period = 0.05;
int LoggingStage = 0;
//

//external field
int ExternalField_h = 0;
int BiasField_h = 0;
double AMPx_h = 0.0;
double AMPy_h = 0.0;
double AMPz_h = 0.0;
//

//STT
int SpinTransferTorque_h =0;

//Integrator parameters
int METHOD_h = 2;
double h_h = 1e-5;
double RelTol = 1e-5;
double AbsTol_h = 1e-10;
//

//Viewer Host Side Globals
bool OpenGlInitialised = false;
bool BufferCreated = false;
bool BufferCreated_xz = false;
bool RenderWindowCreated = false;
int Viewer_zslice_host = 0;
int Viewer_contrast_host = 1;
bool ViewXZ =false;
bool ViewMag = true;
bool ViewDemag = false;
bool ViewExch = false;
bool ViewTorque = false;

DeviceStructOfPointers DEVICE_PTR_STRUCT;

//FFT Precision
bool UseSinglePrecision = false;

//Materials
int NumberOfMaterials =1;

MaterialProperties ArrayOfMaterials_h[MAXMATERIALNUM];
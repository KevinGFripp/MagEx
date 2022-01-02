#include <cufft.h>

#ifndef _DATATYPES_CUH_
#define _DATATYPES_CUH_

typedef double fftw_complex[2];

typedef struct jacobian {
    double J[3][3];
}Jacobian; /*Stores evaluated Jacobian elements*/
typedef struct jacobian* JAC;

typedef struct field {
    double* H_m;
    double* H_ex;
    double* H_ext;
    double* H_anis;
    double* H_eff;
    double* H_stage;
    double* H_stage_1;
    double* H_STT;
}Field;
typedef struct field* FIELD;

typedef struct magnetisation {
    double* M;
    int* Mat;
    int* Bd;
    JAC J;
    int* Pivot;
    int* NUMCELLS;
}Magnetisation;
typedef struct magnetisation* MAG;

typedef struct fftplans {
    cufftHandle MxPlan;
    cufftHandle MyPlan;
    cufftHandle MzPlan;
    cufftHandle HxPlan;
    cufftHandle HyPlan;
    cufftHandle HzPlan;
}FFTPlans;
typedef struct fftplans* PLANS;

typedef struct vector {
    double X[3];
}Vector; /*Vector for LU solver output*/

typedef Vector(*ExtFieldFunctionPtr)(double t, double x, double y, double z);


typedef struct jacobian_rk2 {
    double J[6][6];
}Jacobian_rk2; /*Stores evaluated Jacobian elements*/
typedef struct jacobian_rk2* JAC_RK2;

typedef struct dataptr {
    fftw_complex* kxx;
    fftw_complex* kxy;
    fftw_complex* kyy;
    fftw_complex* kyz;
    fftw_complex* kzz;
    fftw_complex* kxz;
    double* Nxx;
    double* Nxy;
    double* Nyy;
    double* Nyz;
    double* Nzz;
    double* Nxz;
    //Double Precision
    fftw_complex* xFFT;
    fftw_complex* yFFT;
    fftw_complex* zFFT;
    double* Outx_d;
    double* Outy_d;
    double* Outz_d;
    //Single Precision
    cufftComplex* xFFT_s;
    cufftComplex* yFFT_s;
    cufftComplex* zFFT_s;
    float* Outx;
    float* Outy;
    float* Outz;
    double* xReduction;
    double* yReduction;
    double* zReduction;
    double* MaxTorqueReduction;
    double* StepReduction;
    double* dE_Reduction;
    int* NewtonStepsReduction;
}Dataptr;

typedef struct dataptr* MEMDATA;

typedef struct devicestructofpointers
{
    MEMDATA DATA;
    MAG M;
    FIELD H;
    PLANS P;
}DeviceStructOfPointers;

typedef struct regions {
    int x[2]; //lower[0], upper[1]
    int y[2];
    int z[2];
}Region;

typedef struct outputs {
    int xrange[2];
    int yrange[2];
    int zrange[2];
    bool m_unit;
    bool B_demag;
    bool B_full;
    bool B_exch;
}OutputFormat;

typedef struct material{
    double Ms;
    double Aex;
    double damping;
    double Ku;
    Vector Bext;
    char ExcitationType;
}MaterialProperties;

typedef int MaterialHandle;

typedef struct excitationparameters{
    MaterialHandle Handle;
    int ExcitationType;
    Vector Direction;
    Vector SpatialOffset;
    double Amplitude;
    double Frequency;
    double Wavenumber;
}ExcitationParameters;

//zhang-li spin-transfer torque
typedef struct sttparams{
    MaterialHandle handle;
    double J[3]; //current density
    double P; //polarisation rate 
    double Xi; // dimensionless non-adiabaticity 
}STTParameters;

#endif // !_DATATYPES_CUH_

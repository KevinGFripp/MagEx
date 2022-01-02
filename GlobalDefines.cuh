#ifndef _GLOBAL_DEFINES_
#define _GLOBAL_DEFINES_


//Reduction macros
#define MIN(x, y) ((x < y) ? x : y)


#define MAX_BLOCK_DIM_SIZE 65535

//OpenMP
#define THREADS 32

#define STEP 1.0e-5
#define DIM 3
#define RK54DP 1
#define TRAPEZOIDAL_BE 3 //Backward Euler initial step
#define TRAPEZOIDAL_FE 2 // Forward Euler initial step
#define ESDIRK54 4
#define HEUN 5 // Predictor-Corrector Trapezoidal
#define RK54BS 6
#define ESDIRK65 7
#define NEWTONITERATIONS 10
//

//Physical Constants
#define M_PI 3.14159265358979323846264338327
#define mu 1.2566370614359172953850573533118
#define Gamma 0.221079138584402 // 2pi 2.8GHz /kOe Gyromagnetic ratio * mu0
#define Gamma0 1.7595e11        // Gyromagnetic ratio of electron ,rad/Ts
#define e_c 1.602176634e-19 // C Elementary charge
#define m_e 9.1093837015e-31 // kg Electron rest mass
#define hbar (6.62607004e-34)/(2.0*M_PI) // Plank's constant / 2pi
#define B_m 9.274009994e-24 // J/T Bohr magneton

#define STT_Prefactor 2.067034679478783e-25 // mu0 * B_m / 1000* (2.0 * e_c * Gamma0 * mu0);

#define PERIOD 1e-12
#define eps 1e-16
#define Oersted 0.1 //mT
#define M_ADD 1
#define M_SUB -1

//Materials
#define MAXMATERIALNUM 8

//Excitation types
#define NoExcitation 0
#define UniformTemporalSinc 1
#define SpatialTemporalSinc 2
#define GaussianWavePacket 3

#define CHECK_CUFFT_ERRORS(call) { \
    cufftResult_t err; \
    if ((err = (call)) != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %d:%s at %s:%d\n", err, _cudaGetErrorEnum(err), \
                __FILE__, __LINE__); \
        exit(1); \
    } \
}

#endif // !_GLOBAL_DEFINES_


#include <helper_gl.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_gl_interop.h>
// CUDA runtime utilities and system includes
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_atomic_functions.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sm_60_atomic_functions.h>
#include <ctime>
#include <helper_cuda.h>

//Simulator Headers
#include "DataTypes.cuh"
#include "GlobalDefines.cuh"
#include "Device_Globals_Constants.cuh"
#include "Host_Globals.cuh"
#include "Array_Indexing_Functions.cuh"
#include "Simulation_Parameter_Wrapper_Functions.cuh"
#include "DemagnetisingTensor_Functions.cuh"
#include "Magnetisation_Functions.cuh"
#include "Data_Transfer_Functions.cuh"
#include "Pointer_Functions.cuh"
#include "Print_and_Log_Functions.cuh"
#include "ExchangeField.cuh"
#include "Reduction_Functions.cuh"
#include "Average_Quantities.cuh"
#include "ODE_LinAlg.cuh"
#include "Device_VectorMath.cuh"
#include "LandauLifshitz.cuh"
#include "Device_FFT_Functions.cuh"
#include "MemoryAllocation.cuh"
#include "ThreadsAndBlocks_Functions.cuh"
#include "Device_State_Functions.cuh"
#include "DemagnetisingField.cuh"
#include "UniaxialAnisotropy.cuh"
#include "EffectiveField.cuh"
#include "ExternalFields.cuh"
#include "ODE_Integrator.cuh"
#include "ODE_Integrator_RK54DP.cuh"
#include "ODE_Integrator_RK54BS.cuh"
#include "ODE_Integrator_ESDIRK54.cuh"
#include "ODE_Integrator_ESDIRK659.cuh"
#include "Host_OpenGL_Globals.cuh"
#include "Host_OpenGL_RenderWindow.cuh"
#include "Host_Engine.cuh"
#include "DefineMaterials_Functions.cuh"
#include "Reduction_Kernels.cuh"

__host__ void LandauDomainTest(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void ClockTimerTest();
__host__ void DiskArray(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void DiskTest(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void DiskUniformSinc_Stiffness(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void DemagTensorTest(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void StandardProblem4a(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void ThinFilmBV_PBC(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void MobiusMode_ThinFilmBV_PBC(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void ThinFilmBV_PBC_MaterialTest(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void Waveguide_BV_StandardProblem(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void ThinFilmDE_PBC(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void ThinFilmFV_PBC(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void ThinFilmAnisotropy_PBC(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void StandardProblem3(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void DipoleFieldTest(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void PermalloyStripe_Switching(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void VortexCoreGyration(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void PermalloyTransducer_PBC(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void PermalloySphere(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void StandardProblem5_MaterialTest(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void PermalloyTransducer_PBC_Demag(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d);
__host__ void Reduction_Sum_Test(int size);
__host__ void Reduction_ProdSumEnergy_Test(int size);
__host__ void Reduction_Max_Test(int size);
__host__ void Reduction_Min_Test(int size);

int main(int argc, char** argv)
{

    int nthreads, id;
    ARG_c = &argc;
    ARG_v = argv;

    printf("---------CPU Multi-Threading------------\n");
#pragma omp parallel
    {
        id = omp_get_thread_num();
        if (id == 0)
        {
            nthreads = omp_get_num_threads();

            printf("OpenMP : Running %d threads\n", nthreads);
        }
    }
    printf("----------------------------------------\n\n");
    printf("\n---------CUDA Device Info-------------\n");
    int devID = 0;
    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("Running on %s\n",
        deviceProps.name);
    printf("GPU Clock :%f GHz Memory clock : %d MHz\n",
        (float)(deviceProps.clockRate)/1000000.0,deviceProps.memoryClockRate/1000);
    printf("%s has %d Stream Multi-Processors\n",
        deviceProps.name, deviceProps.multiProcessorCount);
    printf("----------------------------------------\n\n");
    
    PLANS plans_h;
    MAG Mag_h, Mag_temp, Mag_d;
    FIELD H_h, H_temp, H_d;
    MEMDATA DATA_h, DATA_temp, DATA_d;

    Mag_h = (MAG)malloc(sizeof(Magnetisation));
    H_h = (FIELD)malloc(sizeof(Field));
    DATA_h = (MEMDATA)malloc(sizeof(Dataptr));
    plans_h = (PLANS)malloc(sizeof(fftplans));

    cudaHostRegister(Mag_h, sizeof(Magnetisation), 0);
    cudaHostRegister(H_h, sizeof(Field), 0);
    cudaHostRegister(DATA_h, sizeof(Dataptr), 0);

    PointerCheck(Mag_h != NULL), PointerCheck(H_h != NULL);
    PointerCheck(DATA_h != NULL), PointerCheck(plans_h != NULL);

    Mag_temp = (MAG)malloc(sizeof(Magnetisation));
    H_temp = (FIELD)malloc(sizeof(Field));
    DATA_temp = (MEMDATA)malloc(sizeof(Dataptr));


    PointerCheck(Mag_temp != NULL), PointerCheck(H_temp != NULL), PointerCheck(DATA_temp != NULL);

    cudaMalloc((void**)&(Mag_d), sizeof(Magnetisation));
    cudaMalloc((void**)&(H_d), sizeof(Field));
    cudaMalloc((void**)&(DATA_d), sizeof(Dataptr));

    InitialiseMaterialsArray();

   // Reduction_ProdSumEnergy_Test(1000000);
   // Reduction_Sum_Test(100000);
   // Reduction_Max_Test(1024);
   // Reduction_Min_Test(10000);
   // 
 // ClockTimerTest();

 // DiskArray(DATA_h, Mag_h, H_h, plans_h,
 //           DATA_temp, Mag_temp, H_temp,
 //           DATA_d, Mag_d, H_d);

    DiskTest(DATA_h, Mag_h, H_h, plans_h,
             DATA_temp, Mag_temp, H_temp,
             DATA_d, Mag_d, H_d);

   //DiskUniformSinc_Stiffness(DATA_h, Mag_h, H_h, plans_h,
   //                          DATA_temp, Mag_temp, H_temp,
   //                          DATA_d, Mag_d, H_d);

  //LandauDomainTest(DATA_h, Mag_h, H_h, plans_h,
  //                 DATA_temp, Mag_temp, H_temp,
  //                 DATA_d, Mag_d, H_d);

 // VortexCoreGyration(DATA_h, Mag_h, H_h, plans_h,
 //                    DATA_temp, Mag_temp, H_temp,
 //                    DATA_d, Mag_d, H_d);

 // PermalloyTransducer_PBC(DATA_h, Mag_h, H_h, plans_h,
 //                         DATA_temp, Mag_temp, H_temp,
 //                         DATA_d, Mag_d, H_d);

 // PermalloyTransducer_PBC_Demag(DATA_h, Mag_h, H_h, plans_h,
 //                               DATA_temp, Mag_temp, H_temp,
 //                               DATA_d, Mag_d, H_d);

  // StandardProblem5_MaterialTest(DATA_h, Mag_h, H_h, plans_h,
  //                               DATA_temp, Mag_temp, H_temp,
  //                               DATA_d, Mag_d, H_d);

  //StandardProblem4a(DATA_h, Mag_h, H_h, plans_h,
  //                   DATA_temp, Mag_temp, H_temp,
  //                   DATA_d, Mag_d, H_d);

    // DemagTensorTest(DATA_h, Mag_h, H_h, plans_h,
    //                 DATA_temp, Mag_temp, H_temp,
    //                 DATA_d, Mag_d, H_d);

      //DipoleFieldTest(DATA_h, Mag_h, H_h, plans_h,
      //                DATA_temp, Mag_temp, H_temp,
      //                DATA_d, Mag_d, H_d);

    // MagnetiseTest(DATA_h, Mag_h, H_h, plans_h,
    //               DATA_temp, Mag_temp, H_temp,
    //               DATA_d, Mag_d, H_d);

  // ThinFilmBV_PBC(DATA_h, Mag_h, H_h, plans_h,
  //                DATA_temp, Mag_temp, H_temp,
  //                DATA_d, Mag_d, H_d);

   // MobiusMode_ThinFilmBV_PBC(DATA_h, Mag_h, H_h, plans_h,
   //                           DATA_temp, Mag_temp, H_temp,
   //                           DATA_d, Mag_d, H_d);

 // Waveguide_BV_StandardProblem(DATA_h, Mag_h, H_h, plans_h,
 //                              DATA_temp, Mag_temp, H_temp,
 //                              DATA_d, Mag_d, H_d);
  
  //ThinFilmBV_PBC_MaterialTest(DATA_h, Mag_h, H_h, plans_h,
  //                            DATA_temp, Mag_temp, H_temp,
  //                            DATA_d, Mag_d, H_d);
   // 
   // ThinFilmDE_PBC(DATA_h, Mag_h, H_h, plans_h,
   //                DATA_temp, Mag_temp, H_temp,
   //                DATA_d, Mag_d, H_d);

   // ThinFilmFV_PBC(DATA_h, Mag_h, H_h, plans_h,
   //                DATA_temp, Mag_temp, H_temp,
  //                 DATA_d, Mag_d, H_d);

  // ThinFilmAnisotropy_PBC(DATA_h, Mag_h, H_h, plans_h,
  //                        DATA_temp, Mag_temp, H_temp,
  //                        DATA_d, Mag_d, H_d);

  // StandardProblem3(DATA_h, Mag_h, H_h, plans_h,
  //                  DATA_temp, Mag_temp, H_temp,
  //                  DATA_d, Mag_d, H_d);

   // PermalloyStripe_Switching(DATA_h, Mag_h, H_h, plans_h,
   //                           DATA_temp, Mag_temp, H_temp,
   //                           DATA_d, Mag_d, H_d);

   // PermalloySphere(DATA_h, Mag_h, H_h, plans_h,
   //                 DATA_temp, Mag_temp, H_temp,
   //                 DATA_d, Mag_d, H_d);
 
    return 0;
}
__host__ void ClockTimerTest()
{
    std::clock_t start;
    double duration;

    int count = 0;
    int countmax = 2;
    double period = 2.0;
    start = std::clock();
    printf("Running a timer for %f seconds \n", countmax * period);
    while (count < countmax)
    {
        duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
        if (duration >= period)
        {
            count++;
            printf("%f seconds elapsed \n", count * period);
            start = std::clock();
        }
    }
}
__host__ void ThinFilmBV_PBC(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{
    SinglePrecision(true);

    PBC(0,512);

    int Nx = 4096, Ny = 4, Nz = 16;
    CellSize(5.0, 5.0, 5.0);
    GridSize(Nx, Ny, Nz);
    SetMaterialParameters(8.0e5, 1.3e-11, 1.0);
    SetTime(5.0);
    SetSamplingPeriod(1e-1);
    SetMethod(ESDIRK54);

    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp,
                     &M_temp, &H_temp, &DATA_d, &M_d, &H_d);

    OutputFormat Out = NewOutput();

    BlockGeometry(M_h, Out);

    ApplyBiasField(100.0, 0.0, 0.0);
    UniformState(M_h, 100, 1, 1);


    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    SetMaterialParameters(8.0e5, 1.3e-11, 5e-3), SetTime(2.0), SetSamplingPeriod(1.0e-2);

    IncludeExternalField(true, &ExcitationFunc_CW);

    Out.m_unit = true;

   // StrictTimeSampling(true);
    SetMethod(RK54DP);

    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    return;
}
__host__ void MobiusMode_ThinFilmBV_PBC(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{
    SinglePrecision(true);

    PBC(0, 512);

    int Nx = 4096, Ny = 2, Nz = 16;
    CellSize(5.0, 5.0, 5.0);
    GridSize(Nx, Ny, Nz);
    SetMaterialParameters(8.0e5, 1.3e-11, 1.0);
    SetTime(5.0);
    SetSamplingPeriod(1e-1);
    SetMethod(ESDIRK54);

    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp,
        &M_temp, &H_temp, &DATA_d, &M_d, &H_d);

    OutputFormat Out = NewOutput();

    BlockGeometry(M_h, Out);

    IncludeExternalField(true, &ExcitationFunc_CW);
    UniformState(M_h, 100, 1, 1);


    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    SetMaterialParameters(8.0e5, 1.3e-11, 1e-4), SetTime(3.0), SetSamplingPeriod(1.0/(4.0*10.5));  

   // Out.m_unit = true;

   // StrictTimeSampling(true);

    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    return;
}
__host__ void ThinFilmBV_PBC_MaterialTest(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{
    SinglePrecision(true);

    PBC(0, 256);

    int Nx = 2048, Ny = 4, Nz = 4;
    CellSize(5.0, 5.0, 5.0);
    GridSize(Nx, Ny, Nz);
    SetMaterialParameters(8.0e5, 1.3e-11, 1.0);
    SetTime(3.0);
    SetSamplingPeriod(1e-1);
    SetMethod(ESDIRK54);

    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp,
                     &M_temp, &H_temp, &DATA_d, &M_d, &H_d);

    OutputFormat Out = NewOutput();

    MaterialHandle Permalloy = DefineMaterial(8e5, 1.3e-11, 1.0, 0.0, MakeVector(100.0, 0.0, 0.0),NoExcitation);

    BlockMaterial(M_h, Out, Permalloy);
 
    UniformState_InMaterial(M_h, 100, 1, 1,Permalloy);

    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    SetMaterialParameters(8.0e5, 1.3e-11, 5e-3), SetTime(10.0), SetSamplingPeriod(2.0e-2);

    IncludeExternalField(true, &ExcitationFunc_CW);

    Out.m_unit = true;

    StrictTimeSampling(true);

    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    return;
}
__host__ void StandardProblem5_MaterialTest(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{
    SinglePrecision(true);

    int Nx = 50, Ny = 50 , Nz = 5;
    CellSize(100.0/Nx,100.0/Ny,10.0/Nz);
    GridSize(Nx, Ny, Nz);
    SetMaterialParameters(8.0e5, 1.3e-11, 1.0);
    SetTime(3.0);
    SetSamplingPeriod(1e-1);
    SetMethod(ESDIRK54);

    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp,
                     &M_temp, &H_temp, &DATA_d, &M_d, &H_d);

    OutputFormat Out = NewOutput();

    MaterialHandle Permalloy = DefineMaterial(8e5, 1.3e-11, 1.0);

    BlockMaterial(M_h, Out, Permalloy);

    VortexState_InMaterial(M_h, 1, 1, 0, Permalloy);

    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    SetMaterialParameters(8.0e5, 1.3e-11,0.1),
    SetTime(5.0),
    SetSamplingPeriod(5.0e-3);

    SetMethod(RK54BS);
    ApplySpinTransferTorque_InMaterial(Permalloy, 1.0, MakeVector(1e12, 0.0, 0.0), 0.05);

    
    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    return;
}
__host__ void Waveguide_BV_StandardProblem(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{
    SinglePrecision(true);

    PBC(0,0);

    int Nx = 2048, Ny = 25, Nz = 1;
    CellSize(2.0, 2.0, 1.0);
    GridSize(Nx, Ny, Nz);
    SetMaterialParameters(8.6e5, 1.3e-11, 1.0);
    SetTime(2.0);
    SetSamplingPeriod(1e-1);
    SetMethod(ESDIRK54);

    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp,
                     &M_temp, &H_temp, &DATA_d, &M_d, &H_d);

    OutputFormat Out = NewOutput();

    MaterialHandle Permalloy = DefineMaterial(8.6e5, 1.3e-11, 1.0, 0.0, MakeVector(100.0, 0.0, 0.0), NoExcitation);

    BlockMaterial(M_h, Out, Permalloy);

    UniformState_InMaterial(M_h, 100, 1, 1, Permalloy);

    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    SetMaterialParameters(8.6e5, 1.3e-11,1e-3), SetTime(2.0), SetSamplingPeriod(1e-3);

    IncludeExternalField(true, &ExcitationFunc_CW);

   // Out.m_unit = true;

   // StrictTimeSampling(true);

    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    return;
}
__host__ void ThinFilmDE_PBC(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{
    SinglePrecision(true);

    PBC(0, 256);

    int Nx = 2048, Ny = 4, Nz = 4;
    CellSize(5.0, 5.0, 5.0);
    GridSize(Nx, Ny, Nz);
    SetMaterialParameters(8.0e5, 1.3e-11, 1.0);
    SetTime(3.0);
    SetSamplingPeriod(1e-1);
    SetMethod(ESDIRK54);

    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp,
                     &M_temp, &H_temp, &DATA_d, &M_d, &H_d);

    OutputFormat Out = NewOutput();

    BlockGeometry(M_h, Out);

    ApplyBiasField(0.0, 100.0, 0.0);
    UniformState(M_h, 1, 100, 1);


    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    SetMaterialParameters(8.0e5, 1.3e-11, 5e-3), SetTime(10.0), SetSamplingPeriod(2.0e-2);

    IncludeExternalField(true, &ExcitationFunc_CW);

    Out.m_unit = true;

    StrictTimeSampling(true);

    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    return;
}
__host__ void ThinFilmFV_PBC(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{
    SinglePrecision(true);

    PBC(0, 256);

    int Nx = 2048, Ny = 4, Nz = 4;
    CellSize(5.0, 5.0, 5.0);
    GridSize(Nx, Ny, Nz);
    SetMaterialParameters(8.0e5, 1.3e-11, 1.0);
    SetTime(3.0);
    SetSamplingPeriod(1e-1);
    SetMethod(ESDIRK54);

    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp,
                     &M_temp, &H_temp, &DATA_d, &M_d, &H_d);

    OutputFormat Out = NewOutput();

    BlockGeometry(M_h, Out);

    ApplyBiasField(0.0, 0.0, (mu*MSAT_h + 100.0));
    UniformState(M_h, 1, 1, 100);


    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    SetMaterialParameters(8.0e5, 1.3e-11, 5e-3), SetTime(10.0), SetSamplingPeriod(2.0e-2);

    IncludeExternalField(true, &ExcitationFunc_CW);

    Out.m_unit = true;

    StrictTimeSampling(true);

    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    return;
}
__host__ void ThinFilmAnisotropy_PBC(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{
    SinglePrecision(true);

    PBC(0, 256);

    int Nx = 2048, Ny = 4, Nz = 4;
    CellSize(5.0, 5.0, 5.0);
    GridSize(Nx, Ny, Nz);
    SetMaterialParameters(8.0e5, 1.3e-11, 1.0);
    SetTime(3.0);
    SetSamplingPeriod(1e-1);
    SetMethod(ESDIRK54);

    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp,
                     &M_temp, &H_temp, &DATA_d, &M_d, &H_d);

    OutputFormat Out = NewOutput();

    BlockGeometry(M_h, Out);

    
    SetUniaxialAnisotropy(5e4,1, 0, 0);//~100mT bias in +x
    UniformState(M_h, 100, 1, 1);


    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    SetMaterialParameters(8.0e5, 1.3e-11, 5e-3), SetTime(10.0), SetSamplingPeriod(2.0e-2);

    IncludeExternalField(true, &ExcitationFunc_CW);

  //  Out.m_unit = true;

  //  StrictTimeSampling(true);

    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    return;
}
__host__ void PermalloyTransducer_PBC(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{
    PBC(0,512);
    SinglePrecision(true);
    int Nx = 2048, Ny =2, Nz =16;
    CellSize(5.0, 5.0, 5.0);
    GridSize(Nx, Ny, Nz);
    SetMaterialParameters(8.0e5, 1.3e-11, 1.0);
    SetTime(5.0);
    SetSamplingPeriod(1e-1);
    SetMethod(ESDIRK54);
   
    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp,
                     &M_temp, &H_temp, &DATA_d, &M_d, &H_d);

    OutputFormat Out = NewOutput();

    Region Transducer = DefineRegion(Nx / 2 - 4, Nx / 2 + 5, 0, Ny-1, 3, 6);
    Region ThinFilm = DefineRegion(0, Nx - 1, 0, Ny-1, 10,13);
   
    MaterialHandle FilmMat = DefineMaterial(8e5,1.3e-11,1.0,MakeVector(100.0,0.0,0.0));
    MaterialHandle TransducerMat = DefineMaterial(8e5, 1.3e-11, 1.0,MakeVector(0.0, 0.0, 0.0));

    Cuboid_InRegion(M_h, Transducer, TransducerMat);
    Cuboid_InRegion(M_h, ThinFilm, FilmMat);

    UniformState_InMaterial(M_h, 100, 1, 1, FilmMat);
    UniformState_InMaterial(M_h, 1, 100, 1, TransducerMat);

    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    SetMaterialParameters(8.0e5, 1.3e-11, 5e-3), SetTime(10.0);

    ApplySpinTransferTorque_InMaterial(TransducerMat, 1.0, MakeVector(0.0, 0.0, 2e13), 0.005);

  //  Out.B_demag = true;
  //  Out.m_unit = true;
   // StrictTimeSampling(true);

    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    return;
}
__host__ void PermalloyTransducer_PBC_Demag(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{
    PBC(0, 512);
    SinglePrecision(true);
    int Nx = 2048, Ny = 2, Nz = 48;
    CellSize(5.0, 5.0, 5.0);
    GridSize(Nx, Ny, Nz);
    SetMaterialParameters(8.0e5, 1.3e-11, 1.0);
    SetTime(5.0);
    SetSamplingPeriod(1e-1);
    SetMethod(ESDIRK54);
    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp,
                     &M_temp, &H_temp, &DATA_d, &M_d, &H_d);

    OutputFormat Out = NewOutput();

    Region Transducer = DefineRegion(Nx / 2 - 5, Nx / 2 + 4, 0, Ny - 1, 17, 20);
    Region ThinFilm = DefineRegion(0, Nx - 1, 0, Ny - 1, 24, 27);

    MaterialHandle FilmMat = DefineMaterial(8e5, 1.3e-11, 1.0, MakeVector(0.0, 100.0, 0.0));
    MaterialHandle TransducerMat = DefineMaterial(8e5, 1.3e-11, 1.0, MakeVector(0.0, 0.0, 0.0));

    Cuboid_InRegion(M_h, Transducer, TransducerMat);
    Cuboid_InRegion(M_h, ThinFilm, FilmMat);

    UniformState_InMaterial(M_h, 100, 1, 1, FilmMat);
    UniformState_InMaterial(M_h, 1, -100, 1, TransducerMat);

    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    SetMaterialParameters(8.0e5, 1.3e-11, 5e-3), SetTime(10.0), SetSamplingPeriod(1. / (4. * (16.1)));

    IncludeExternalField(true, &ExcitationFunc_CW);

    //  Out.B_demag = true;   
    //  StrictTimeSampling(true);

    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    return;
}
__host__ void DemagTensorTest(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{

    PBC(128, 0);

    int Nx = 512, Ny = 512, Nz = 1;
    double dxyz = 1.0;

    GridSize(Nx, Ny, Nz);
    CellSize(dxyz, dxyz, dxyz);
    SetMaterialParameters(1 / (mu * (1e-6)), 0, 1.0);
    SetTime(1.0);
    SetSamplingPeriod(1e-1);
    SetMethod(RK54DP);

    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp, &M_temp, &H_temp, &DATA_d, &M_d, &H_d);

    // Region SingleCell = SetRegion(0, 0, 0, 0, 0, 0);

    // MagnetiseFilm3D_InRegion(M_h, SingleCell), UniformState_InRegion(M_h, 1, 0, 0, SingleCell);

    OutputFormat Out = NewOutput();
    BlockGeometry(M_h, Out);

    VortexState(M_h, 1, 1, 0);

    CopyMagToDevice(M_h, M_d);
    ComputeFields(DATA_d, M_d, H_d, P_h, 0);
    //ComputeFields_RKStageEvaluation(DATA_d, M_d, H_d, P_h, 0);

    Out.m_unit = true;
    Out.B_demag = true;
    Out.B_exch = true;
    CopyMagFromDevice(M_h, M_d);
    CopyDemagFieldFromDevice(H_h, H_d);
    CopyExchangeFieldFromDevice(H_h, H_d);
    printDemagField_Outputs(H_h, 0, Out);
    printExchangeField_Outputs(H_h, 0, Out);
    printmagnetisation_Outputs(M_h, 0, Out);

    return;
}
__host__ void DipoleFieldTest(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{

    int Nx = 512, Ny = 512, Nz = 1;
    double dxyz = 1.0;

    GridSize(Nx, Ny, Nz);
    CellSize(dxyz, dxyz, dxyz);
    SetMaterialParameters(1 / (mu * (1e-6)), 0, 1.0);
    SetTime(1.0);
    SetSamplingPeriod(1e-1);
    SetMethod(RK54DP);

    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp, &M_temp, &H_temp, &DATA_d, &M_d, &H_d);

    Region SingleCell = SetRegion(255, 255, 255, 255, 0, 0);

    MagnetiseFilm3D_InRegion(M_h, SingleCell), UniformState_InRegion(M_h, 1, 0, 0, SingleCell);


    OutputFormat Out = NewOutput();

    CopyMagToDevice(M_h, M_d);
    ComputeFields(DATA_d, M_d, H_d, P_h, 0);

    Out.B_demag = true;

    CopyDemagFieldFromDevice(H_h, H_d);
    printDemagField_Outputs(H_h, 0, Out);


    return;
}
__host__ void LandauDomainTest(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{
    SinglePrecision(true);

    int Nx = 256, Ny =512, Nz =4;
    double dxyz = 5.0;

    GridSize(Nx, Ny, Nz), CellSize(dxyz, dxyz, dxyz);

    SetMaterialParameters(8.0e5, 1.3e-11, 1.0);

    SetTime(4.0),SetSamplingPeriod(1e-1);

    SetMethod(RK54DP);

    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp,
                     &M_temp, &H_temp, &DATA_d, &M_d, &H_d);

    OutputFormat Out = NewOutput();
    MaterialHandle Permalloy = DefineMaterial(8e5, 1.3e-11, 1.0);
    BlockMaterial(M_h, Out, Permalloy);
    UniformState_InMaterial(M_h, 1, 1,5,Permalloy);

    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    SetMaterialParameters(8.0e5, 1.3e-11,0.008);
    SetTime(4.0), SetSamplingPeriod(5e-3);

    SetMethod(RK54BS);
    ApplyBiasField(-6.0, -0.75, 0.0);

    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    return;
}
__host__ void VortexCoreGyration(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{
    SinglePrecision(true);
    PBC(0, 0);

    int Nx = 256, Ny = 256, Nz = 4;
    double dxyz = 3.0;

    GridSize(Nx, Ny, Nz);
    CellSize(dxyz, dxyz, dxyz);
    SetMaterialParameters(8.0e5, 1.3e-11, 1.0);
    SetTime(2.0);
    SetSamplingPeriod(1e-2);
    SetMethod(RK54DP);

    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp,
                     &M_temp, &H_temp, &DATA_d, &M_d, &H_d);

    OutputFormat Out = NewOutput();
    MaterialHandle Permalloy = DefineMaterial(8e5, 1.3e-11, 1.0);

    Disk(M_h, 120 * dxyz, Nz * dxyz, Permalloy);
    VortexState_InMaterial(M_h, 1, 1, 0, Permalloy);

    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    SetMaterialParameters(8.0e5, 1.3e-11, 0.1);
    SetTime(2.0);
    SetSamplingPeriod(5e-3);
    SetMethod(ESDIRK54);
    ApplySpinTransferTorque_InMaterial(Permalloy, 1.0, MakeVector(1e12, 0.0, 0.0), 0.05);

    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    return;
}
__host__ void StandardProblem4a(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{
    SinglePrecision(true);

    double LENGTH = 500.0, WIDTH = 125.0, THICKNESS = 3.0;
    int Nx = 100, Ny =25, Nz = 1;
    GridSize(Nx, Ny, Nz);
    CellSize(LENGTH / Nx, WIDTH / Ny, THICKNESS / Nz);
    SetMaterialParameters(8.0e5, 1.3e-11, 1.0);
    SetTime(2.0);
    SetSamplingPeriod(1e-2);
    SetMethod(ESDIRK54);
    SetMaxError(1e-6);

    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp,
                     &M_temp, &H_temp, &DATA_d, &M_d, &H_d);

    MagnetiseFilm3D(M_h, LENGTH, WIDTH, THICKNESS, 1);

    UniformState(M_h, 100, 10, 1);

    OutputFormat Out = NewOutput();


    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

   // double AMPX = -24.6, AMPY = 4.3; //Field 1

    double AMPX = -35.5, AMPY = -6.3; //Field 2

    ApplyBiasField(AMPX, AMPY, 0.0);
    SetMaterialParameters(8.0e5, 1.3e-11, 0.02);
    SetTime(1.0);
    SetSamplingPeriod(2e-3);
    SetMethod(RK54BS);
    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    return;
}
__host__ void PermalloyStripe_Switching(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{
    SinglePrecision(true);

    double LENGTH = 2560.0, WIDTH = 1280.0, THICKNESS = 10.0;
    int Nx = 1024, Ny = 256, Nz = 2;
    GridSize(Nx, Ny, Nz);
    CellSize(5.0,5.0,5.0);
    SetMaterialParameters(8.0e5, 1.3e-11, 1.0);
    SetTime(5.0);
    SetSamplingPeriod(1e-2);
    SetMethod(RK54DP);


    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp,
                     &M_temp, &H_temp, &DATA_d, &M_d, &H_d);

  
    OutputFormat Out = NewOutput();

    BlockGeometry(M_h, Out);

    UniformState(M_h, 100, 10, 1);

    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    double AMPX = -20.0, AMPY = -4.0; //Field 2

    ApplyBiasField(AMPX, AMPY, 0.0);

    SetMaterialParameters(8.0e5, 1.3e-11, 0.02);
    SetTime(0.14);
    SetSamplingPeriod(1e-2);
    SetMethod(RK54BS);

    Out.m_unit = true;
  
    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);
    
    return;
}
__host__ void PermalloySphere(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{
    SinglePrecision(true);

    
    int Nx = 128, Ny = 128, Nz = 128;
    GridSize(Nx, Ny, Nz);
    CellSize(5.0,5.0,5.0);
    SetMaterialParameters(8.0e5, 1.3e-11, 1.0);
    SetTime(5.0);
    SetSamplingPeriod(1e-2);
    SetMethod(RK54BS);


    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp,
                     &M_temp, &H_temp, &DATA_d, &M_d, &H_d);

    MagnetiseSphere3D(M_h, 60 * 5.0, 0);

    VortexState(M_h, 1, 1, 0);

    OutputFormat Out = NewOutput();


    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    double AMPX = -20.0, AMPY = -4.0; //Field 2

    ApplyBiasField(AMPX, AMPY, 0.0);

    SetMaterialParameters(8.0e5, 1.3e-11, 0.02);
    SetTime(1.0);
    SetSamplingPeriod(5e-3);
    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    return;
}
__host__ void StandardProblem3(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{
    SinglePrecision(true);
    int N = 16;
    double Ms = 8e5;
    double Aex = 1.3e-11;
    double Km = 0.5 * mu * (1e-6) * Ms * Ms;
    double Ku = 0.1 * Km;
    double Dc = 0.1;
    double lex = (1e9)*sqrt(2 * Aex / (mu * (1e-6) * Ms * Ms));
    double Length = 8*lex;
   
    double Cell =Length/16;
    GridSize(N,N,N);
    CellSize(Cell,Cell,Cell);     
    SetMethod(RK54BS);

    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp,
                     &M_temp, &H_temp, &DATA_d, &M_d, &H_d);

    SetMaterialParameters(Ms,Aex, 1.0);
    SetUniaxialAnisotropy(Ku, 0, 1, 0);

    OutputFormat Out = NewOutput();

    BlockGeometry(M_h, Out);

    SetTime(3.0);
    SetSamplingPeriod(2e-1);
    
    for (int n = 0; n < 6; n++)
    {     
        Length = (8.0 + n * Dc) * lex;
        Cell = Length / 16.0;
        CellSize(Cell,Cell,Cell);
        UniformState(M_h,1, 5, 1);
        Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

        VortexState(M_h, 1, 1, 0);
        Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);
    }

    
}
__host__ void DiskArray(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{
    // An Array of 4 disks 
    double THICKNESS = 20.0;
    double RADIUS = 464.0;
    int Nx = 512, Ny = 512, Nz = 1;
    double dxy = 4.0;
    double dz = 5.0;

    int RADIUS_c = 120;
    int SPACE = 10;
    GridSize(Nx, Ny, Nz);
    CellSize(dxy, dxy, dz);
    SetMaterialParameters(8.0e5, 1.3e-11, 1.0);
    SetTime(1.0);
    SetSamplingPeriod(1e-2);
    SetMethod(RK54BS);


    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp,
                     &M_temp, &H_temp, &DATA_d, &M_d, &H_d);


    MagnetiseDisk3D_bool(M_h, RADIUS, THICKNESS / 2, RADIUS_c + 1, RADIUS_c + 1, 5, true); //top left

    MagnetiseDisk3D_bool(M_h, RADIUS, THICKNESS / 2, 3 * (RADIUS_c + 1) + SPACE, RADIUS_c + 1, 5, true); //top right

    MagnetiseDisk3D_bool(M_h, RADIUS, THICKNESS / 2, (RADIUS_c + 1), 3 * (RADIUS_c + 1) + SPACE, 5, true); //bottom left

    MagnetiseDisk3D_bool(M_h, RADIUS, THICKNESS / 2, 3 * (RADIUS_c + 1) + SPACE, 3 * (RADIUS_c + 1) + SPACE, 5, true); //bottom right

    Region Disk1; //top left
    Disk1.x[0] = 0;
    Disk1.x[1] = 2 * (RADIUS_c + 1) - 1;
    Disk1.y[0] = 0;
    Disk1.y[1] = 2 * (RADIUS_c + 1) - 1;
    Disk1.z[0] = 0;
    Disk1.z[1] = Nz;

    Region Disk2; //top right
    Disk2.x[0] = 2 * (RADIUS_c + 1) + SPACE;
    Disk2.x[1] = 4 * (RADIUS_c + 1) - 1 + SPACE;
    Disk2.y[0] = 0;
    Disk2.y[1] = 2 * (RADIUS_c + 1) - 1;
    Disk2.z[0] = 0;
    Disk2.z[1] = Nz;

    Region Disk3; //bottom left
    Disk3.y[0] = 2 * (RADIUS_c + 1) + SPACE;
    Disk3.y[1] = 4 * (RADIUS_c + 1) - 1 + SPACE;
    Disk3.x[0] = 0;
    Disk3.x[1] = 2 * (RADIUS_c + 1) - 1;
    Disk3.z[0] = 0;
    Disk3.z[1] = Nz;

    Region Disk4; //bottom right
    Disk4.x[0] = 2 * (RADIUS_c + 1) + SPACE;
    Disk4.x[1] = 4 * (RADIUS_c + 1) - 1 + SPACE;
    Disk4.y[0] = 2 * (RADIUS_c + 1) + SPACE;
    Disk4.y[1] = 4 * (RADIUS_c + 1) - 1 + SPACE;
    Disk4.z[0] = 0;
    Disk4.z[1] = Nz;



    UniformState(M_h, 10, 1, 1);

    OutputFormat Out;

    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    return;
}
__host__ void DiskTest(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{
    SinglePrecision(true);
    int Nx = 200, Ny = 200, Nz = 1;
    double Diameter = 195.0;
    double dxy = 1.0;
    double dz = 5.0;
    double thickness = Nz * dz;
    GridSize(Nx, Ny, Nz);
    CellSize(dxy, dxy, dz);
    SetMaterialParameters(8e5, 1.3e-11,1.0);
    SetTime(1.0);
    SetSamplingPeriod(1e-3);
    SetMethod(ESDIRK54);

    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp,
                     &M_temp, &H_temp, &DATA_d, &M_d, &H_d);

    MagnetiseDisk3D(M_h, Diameter / 2., thickness, 1);
    UniformState(M_h, 100, 1, 1);


    ApplyBiasField(933 * Oersted, 0.0, 0.0);

    OutputFormat Out;
    Out = NewOutput();


    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);


    IncludeExternalField(true, &ExcitationFunc_CW);
    SetMaterialParameters(8e5, 1.3e-11, 0.01);

   // Out.m_unit = true;

    SetTime(5.0);
    SetSamplingPeriod(1.0e-2);
   // StrictTimeSampling(true);
   // SetMethod(RK54DP);
    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    return;
}
__host__ void DiskUniformSinc_Stiffness(MEMDATA DATA_h, MAG M_h, FIELD H_h, PLANS P_h,
    MEMDATA DATA_temp, MAG M_temp, FIELD H_temp,
    MEMDATA DATA_d, MAG M_d, FIELD H_d)
{
    SinglePrecision(true);
    int Nx = 200, Ny = 200, Nz = 1;
    double Diameter = 195.0;
    double dxy = 1.0;
    double dz = 5.0;
    double thickness = Nz * dz;
    GridSize(Nx, Ny, Nz);
    CellSize(dxy, dxy, dz);
    SetMaterialParameters(8e5, 1.3e-11, 1.0);
    SetTime(0.5);
    SetSamplingPeriod(1e-2);
    SetMethod(ESDIRK54);
    GlobalInitialise(&DATA_h, &M_h, &H_h, &P_h, &DATA_temp,
                     &M_temp, &H_temp, &DATA_d, &M_d, &H_d);

    MagnetiseDisk3D(M_h, Diameter / 2., thickness, 1);
    UniformState(M_h, 100, 1, 1);


    ApplyBiasField(933 * Oersted, 0.0, 0.0);

    OutputFormat Out;
    Out = NewOutput();


    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);


    SetTime(0.10);
    SetSamplingPeriod(1.0e-2);

    double damp = 0.001;
   
 /*  for (int n = 0; n < 4; n++)
    {
        SetMaterialParameters(8e5, 1.3e-11, damp);
        Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);
        damp /= 2.0;
    }*/

   // SetMaterialParameters(8e5, 1.3e-11, damp);
  //  Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

   // SetMethod(RK54DP);
    SetMaterialParameters(8e5, 1.3e-11,damp);
    Simulate(DATA_h, M_h, H_h, P_h, DATA_d, M_d, H_d, Out);

    return;
}
__host__ void Reduction_Sum_Test(int size)
{
    int maxThreads = 256;  // number of threads per block
    int whichKernel = 6;
    int maxBlocks = 64;

    unsigned int bytes = size * sizeof(double);

    double* h_idata = (double*)malloc(bytes);

    //init host array
    for (int i = 0; i < size; i++)
    {
        h_idata[i] = 1.0;
    }

    int numBlocks = 0;
    int numThreads = 0;
    getNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks,
        numThreads);

    // allocate device memory and data
    double* d_idata = NULL;
    double* d_odata = NULL;

    checkCudaErrors(cudaMalloc((void**)&d_idata, bytes));
    checkCudaErrors(cudaMalloc((void**)&d_odata, numBlocks * sizeof(double)));

    // copy data directly to device memory
    checkCudaErrors(
        cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_odata, h_idata, numBlocks * sizeof(double),
        cudaMemcpyHostToDevice));

    double result = 0;

    result = Reduce_Sum(size, numThreads, numBlocks, maxThreads, maxBlocks, d_idata, d_idata);
    printf("Expected Answer =%f, Reduction Answer = %f\n",(float)size,result);

    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
}
__host__ void Reduction_Max_Test(int size)
{
    int maxThreads = 256;  // number of threads per block
    int whichKernel = 6;
    int maxBlocks = 64;

    unsigned int bytes = size * sizeof(double);

    double* h_idata = (double*)malloc(bytes);

    //init host array
    for (int i = 0; i < size; i++)
    {
        h_idata[i] = i;
    }
    

    int numBlocks = 0;
    int numThreads = 0;
    getNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks,
        numThreads);

    // allocate device memory and data
    double* d_idata = NULL;
    double* d_odata = NULL;

    checkCudaErrors(cudaMalloc((void**)&d_idata, bytes));
    checkCudaErrors(cudaMalloc((void**)&d_odata, numBlocks * sizeof(double)));

    // copy data directly to device memory
    checkCudaErrors(
        cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_odata, h_idata, numBlocks * sizeof(double),
        cudaMemcpyHostToDevice));

    double result = 0;

    result = Reduce_Max(size, numThreads, numBlocks, maxThreads, maxBlocks, d_idata, d_odata);
    printf("Expected Answer =%f, Reduction Answer = %f\n",h_idata[size-1], result);

    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
}
__host__ void Reduction_Min_Test(int size)
{
    int maxThreads = 256;  // number of threads per block
    int whichKernel = 6;
    int maxBlocks = 64;

    unsigned int bytes = size * sizeof(double);

    double* h_idata = (double*)malloc(bytes);

    //init host array
    for (int i = 0; i < size; i++)
    {
        h_idata[i] = i +21.0;
    }


    int numBlocks = 0;
    int numThreads = 0;
    getNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks,
        numThreads);

    // allocate device memory and data
    double* d_idata = NULL;
    double* d_odata = NULL;

    checkCudaErrors(cudaMalloc((void**)&d_idata, bytes));
    checkCudaErrors(cudaMalloc((void**)&d_odata, numBlocks * sizeof(double)));

    // copy data directly to device memory
    checkCudaErrors(
        cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_odata, h_idata, numBlocks * sizeof(double),
        cudaMemcpyHostToDevice));

    double result = 0;

    result = Reduce_Min(size, numThreads, numBlocks, maxThreads, maxBlocks, d_idata, d_odata);
    printf("Expected Answer =%f, Reduction Answer = %f\n",21.0, result);

    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
}
__host__ void Reduction_ProdSumEnergy_Test(int size)
{
    int maxThreads = 256;  // number of threads per block
    int whichKernel = 6;
    int maxBlocks = 64;

    unsigned int bytes = size * sizeof(double);

    double* M_idata = (double*)malloc(bytes);
    double* H_idata = (double*)malloc(bytes);

    //init host array
    for (int i = 0; i < size; i++)
    {
        M_idata[i] = 1.0;
        H_idata[i] = 2.0;
    }

    int numBlocks = 0;
    int numThreads = 0;
    getNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks,
        numThreads);

    // allocate device memory and data
    double* Md_idata = NULL;
    double* Hd_idata = NULL;
    double* Ed_odata = NULL;

    checkCudaErrors(cudaMalloc((void**)&Md_idata, bytes));
    checkCudaErrors(cudaMalloc((void**)&Hd_idata, bytes));
    checkCudaErrors(cudaMalloc((void**)&Ed_odata, numBlocks * sizeof(double)));

    // copy data directly to device memory
    checkCudaErrors(
        cudaMemcpy(Md_idata, M_idata, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(Hd_idata, H_idata, bytes, cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(Ed_odata, h_idata, numBlocks * sizeof(double),
    //    cudaMemcpyHostToDevice));

    double result = 0;

    result = Reduce_Energy(size, numThreads, numBlocks, maxThreads, maxBlocks,
                           Md_idata,Hd_idata, Ed_odata);
    printf("Expected Answer =%f, Reduction Answer = %f\n", ((float)size)*3.0, result);

    cudaFree(Md_idata);
    cudaFree(Hd_idata);
    free(M_idata);
    free(H_idata);
    cudaFree(Ed_odata);
}
 


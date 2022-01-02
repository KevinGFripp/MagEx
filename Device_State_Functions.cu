#include "Device_State_Functions.cuh"
#include <device_launch_parameters.h>
#include "Device_Globals_Constants.cuh"
#include "Host_Globals.cuh"
#include <helper_cuda.h>
__host__ void UpdateDeviceTime(double time)
{
    checkCudaErrors(cudaMemcpyToSymbol(t_d, &time, sizeof(double)));
}
__host__ void SetCurrentTime(double time)
{
    t_h = time;
    return;
}
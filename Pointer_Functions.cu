#include "Pointer_Functions.cuh"

__host__ void PointerCheck(int a)
{
    if (a == 0)
    {
        printf("Memory Allocation Failure");
        exit(1);
    }
    return;
}

__host__ void PointerPlansCheck(int a)
{
    if (a == 0)
    {
        printf("FFT Plans Initialisation Failure");
        exit(1);
    }
    return;
}
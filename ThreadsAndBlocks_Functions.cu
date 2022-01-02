#include "ThreadsAndBlocks_Functions.cuh"
#include "Host_Globals.cuh"
#include "Reduction_Functions.cuh"

__host__ int Factorise(int N, int n)
{
    int Val_prev = N - 1;
    int Val = N - 1;
    int i = 0;

    if (N == 2)
    {
        return 2;
    }
    if (N == 0)
    {
        return 0;
    }

    if (N == 1)
    {
        return 1;
    }

    if (N == 3)
    {
        return 3;
    }

    while (i <= n)
    {
        Val--;
        if (N % Val == 0)
        {
            i++;
            if (Val == 1)
            {
                Val = Val_prev;
                break;
            }
            else
            {
                Val_prev = Val;
            }

        }
    }

    return Val;
}
bool IsPowerOfTwo(int x)
{
    return (x != 0) && ((x & (x - 1)) == 0);
}
__device__ bool IsPowerOfTwo_d(int x)
{
    return (x != 0) && ((x & (x - 1)) == 0);
}
unsigned int nextPowerOfTwo(unsigned int n)
{
  /*  --n;

    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
   
    return n + 1;*/

    double power = ceil(log2((double)n));
    double result = pow(2.0, power);
    return (int)result;
}

__host__ void CalculateThreadsAndBlocksReductions_1D()
{
    unsigned int tx = 1, ty = 1, tz = 1;
    unsigned int bx = 1, by = 1, bz = 1;

    NumberofBlocksReduction.x = 64;
    NumberofBlocksReduction.y = 1;
    NumberofBlocksReduction.z = 1;


    NumberofThreadsReduction.x = 256;
    NumberofThreadsReduction.y = 1;
    NumberofThreadsReduction.z = 1;

    return;
}
__host__ void Integration_GetThreadsAndBlocks()
{
    unsigned int tx = 1, ty = 1, tz = 1;
    unsigned int bx = 1, by = 1, bz = 1;


    const unsigned int ThreadsPerBlock = 128;
    const unsigned int MinThreads = 128;

    unsigned int Size_x = NUMZ_h;
    unsigned int Size_y = NUMY_h;
    unsigned int Size_z = NUM_h;

    unsigned int Nmin = 1;

    if(Size_x * Size_y * Size_z <= MinThreads)
    {
     //very small grid size 

     //Extend size to next power of 2 if needed

        Size_x = nextPowerOfTwo(Size_x);
        Size_y = nextPowerOfTwo(Size_y);
        Size_z = nextPowerOfTwo(Size_z);

        //find smallest grid size
        Nmin = Size_x;

        if (Size_y < Nmin) { Nmin = Size_y; }
        if (Size_z < Nmin) { Nmin = Size_z; }

        if (Nmin <= 2)
        {
            bx = 1, by = 1, bz = 1;
        }
        else {
                bx = Size_x / Nmin,
                by = Size_y / Nmin,
                bz = Size_z / Nmin;
        }

        tx = Size_x/bx,
        ty = Size_y/by,
        tz = Size_z/bz;
    }
    else {

        //Extend size to next power of 2 if needed
        Size_x = nextPowerOfTwo(Size_x);
        Size_y = nextPowerOfTwo(Size_y);
        Size_z = nextPowerOfTwo(Size_z);

        //find smallest grid size
        unsigned int Nblocks = (Size_x * Size_y * Size_z) / ThreadsPerBlock;

        Nmin = Size_x;

        if (Size_y < Nmin) {Nmin = Size_y;}
        if (Size_z < Nmin) {Nmin = Size_z;}

        bx = Size_x / Nmin,
        by = Size_y / Nmin,
        bz = Size_z / Nmin;

        unsigned int CurrentNumberOfBlocks = bx * by * bz;

        while (CurrentNumberOfBlocks > Nblocks)
        {
            if (bz > 1)
            {
                bz /= 2;
            }
            CurrentNumberOfBlocks = bx * by * bz;
            if(CurrentNumberOfBlocks == Nblocks)
            {
                break;
            }

            if (by > 1)
            {
                by /= 2;
            }
            CurrentNumberOfBlocks = bx * by * bz;
            if (CurrentNumberOfBlocks == Nblocks)
            {
                break;
            }

            if (bx > 1)
            {
                bx /= 2;
            }
            CurrentNumberOfBlocks = bx * by * bz;
            if (CurrentNumberOfBlocks == Nblocks)
            {
                break;
            }
        }

        //threads
        tx = Size_x / bx,
        ty = Size_y / by,
        tz = Size_z / bz;

        unsigned int CurrentNumberOfThreads = tx * ty * tz;
        //Make sure number of threads is 128
        while (CurrentNumberOfThreads > ThreadsPerBlock)
        {
            tx /= 2;
            bx *= 2;

            CurrentNumberOfThreads = tx * ty * tz;
            if (CurrentNumberOfThreads == ThreadsPerBlock)
            {
                break;
            }


            ty /= 2;
            by *= 2;

            CurrentNumberOfThreads = tx * ty * tz;
            if (CurrentNumberOfThreads == ThreadsPerBlock)
            {
                break;
            }

            tz /= 2;
            bz *= 2;

            CurrentNumberOfThreads = tx * ty * tz;
            if (CurrentNumberOfThreads == ThreadsPerBlock)
            {
                break;
            }
        }
    
    }

    NumberofBlocksIntegrator.x = (int)(bx);
    NumberofBlocksIntegrator.y = (int)(by);
    NumberofBlocksIntegrator.z = (int)(bz);


    NumberofThreadsIntegrator.x = (int)(tx);
    NumberofThreadsIntegrator.y = (int)(ty);
    NumberofThreadsIntegrator.z = (int)(tz);

}
__host__ void Fields_GetThreadsAndBlocks()
{
    unsigned int tx = 1, ty = 1, tz = 1;
    unsigned int bx = 1, by = 1, bz = 1;


    const unsigned int ThreadsPerBlock = 256;
    const unsigned int MinThreads = 256;

    unsigned int Size_x = NUMZ_h;
    unsigned int Size_y = NUMY_h;
    unsigned int Size_z = NUM_h;

    unsigned int Nmin = 1;

    if (Size_x * Size_y * Size_z <= MinThreads)
    {
        //very small grid size 

        //Extend size to next power of 2 if needed

        Size_x = nextPowerOfTwo(Size_x);
        Size_y = nextPowerOfTwo(Size_y);
        Size_z = nextPowerOfTwo(Size_z);

        //find smallest grid size
        Nmin = Size_x;

        if (Size_y < Nmin) { Nmin = Size_y; }
        if (Size_z < Nmin) { Nmin = Size_z; }

        if (Nmin <= 2)
        {
            bx = 1, by = 1, bz = 1;
        }
        else {
            bx = Size_x / Nmin,
            by = Size_y / Nmin,
            bz = Size_z / Nmin;
        }

        tx = Size_x / bx,
        ty = Size_y / by,
        tz = Size_z / bz;
    }
    else {

        //Extend size to next power of 2 if needed
        Size_x = nextPowerOfTwo(Size_x);
        Size_y = nextPowerOfTwo(Size_y);
        Size_z = nextPowerOfTwo(Size_z);

        //find smallest grid size
        unsigned int Nblocks = (Size_x * Size_y * Size_z) / ThreadsPerBlock;

        Nmin = Size_x;

        if (Size_y < Nmin) { Nmin = Size_y; }
        if (Size_z < Nmin) { Nmin = Size_z; }

        bx = Size_x / Nmin,
            by = Size_y / Nmin,
            bz = Size_z / Nmin;

        unsigned int CurrentNumberOfBlocks = bx * by * bz;

        while (CurrentNumberOfBlocks > Nblocks)
        {
            if (bz > 1)
            {
                bz /= 2;
            }
            CurrentNumberOfBlocks = bx * by * bz;
            if (CurrentNumberOfBlocks == Nblocks)
            {
                break;
            }

            if (by > 1)
            {
                by /= 2;
            }
            CurrentNumberOfBlocks = bx * by * bz;
            if (CurrentNumberOfBlocks == Nblocks)
            {
                break;
            }

            if (bx > 1)
            {
                bx /= 2;
            }
            CurrentNumberOfBlocks = bx * by * bz;
            if (CurrentNumberOfBlocks == Nblocks)
            {
                break;
            }
        }

        //threads
            tx = Size_x / bx,
            ty = Size_y / by,
            tz = Size_z / bz;

            unsigned int CurrentNumberOfThreads = tx * ty * tz;
      //Make sure number of threads is 128
            while (CurrentNumberOfThreads > ThreadsPerBlock)
            {
                tx /= 2;
                bx *= 2;

                CurrentNumberOfThreads = tx * ty * tz;
                if(CurrentNumberOfThreads == ThreadsPerBlock)
                {
                    break;
                }


                ty /= 2;
                by *= 2;

                CurrentNumberOfThreads = tx * ty * tz;
                if (CurrentNumberOfThreads == ThreadsPerBlock)
                {
                    break;
                }

                tz /= 2;
                bz *= 2;

                CurrentNumberOfThreads = tx * ty * tz;
                if (CurrentNumberOfThreads == ThreadsPerBlock)
                {
                    break;
                }
            }

    }

    NumberofBlocks.x = (int)(bx);
    NumberofBlocks.y = (int)(by);
    NumberofBlocks.z = (int)(bz);


    NumberofThreads.x = (int)(tx);
    NumberofThreads.y = (int)(ty);
    NumberofThreads.z = (int)(tz);

    //Demag Padded
   
   
    NumberofBlocksPadded.x = 2 * bx,
    NumberofBlocksPadded.y = 2 * by,
    NumberofBlocksPadded.z = 2 * bz;

    if (PBC_x_h != 0)
    {
        NumberofBlocksPadded.x = bx,
        NumberofBlocksPadded.y = 2 * by,
        NumberofBlocksPadded.z = 2 * bz;
    }
    if (PBC_y_h != 0)
    {
        NumberofBlocksPadded.x = 2 * bx,
        NumberofBlocksPadded.y = by,
        NumberofBlocksPadded.z = 2 * bz;
    }

    if (PBC_x_h != 0 && PBC_y_h != 0)
    {
        NumberofBlocksPadded.x =bx,
        NumberofBlocksPadded.y = by,
        NumberofBlocksPadded.z = 2 * bz;
    }

    NumberofThreadsPadded.x = tx,
    NumberofThreadsPadded.y = ty,
    NumberofThreadsPadded.z = tz;

}
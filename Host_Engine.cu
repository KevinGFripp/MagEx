#include "Host_Engine.cuh"
#include <stdio.h>
#include <stdlib.h>
#include "Host_OpenGL_RenderWindow.cuh"
#include "Data_Transfer_Functions.cuh"
#include "Device_State_Functions.cuh"
#include "Print_And_Log_Functions.cuh"
#include "Host_OpenGL_Globals.cuh"
#include "Simulation_Parameter_Wrapper_Functions.cuh"
#include "Host_Globals.cuh"
#include "EffectiveField.cuh"
#include "ODE_Integrator.cuh"
#include <ctime>
#include "DefineMaterials_Functions.cuh"
#include "ExchangeField.cuh"

__host__ void Simulate(MEMDATA DATA, MAG M, FIELD H, PLANS P,
    MEMDATA DATA_d, MAG M_d, FIELD H_d, OutputFormat Out)
{
    ExchangeStencilParameters();
    WriteMaterialsArrayToDevice();
    MeshBoundaries(M);
    CopyMagToDevice(M, M_d);

    std::clock_t start;
    int IsFinished = 0, GUIcount = 0, Flag = 0, stepcount = 0.0, Count = 0, ControlCount = 0;
    int consolestepcount = 20;

    double h, t_previous = 0.0, MaxTorque = 0.0, MinTorque = 0.0, MaximumErr = 0.0;

    bool IsFirstStep = true;


    Vector StepData;
    StepData.X[0] = 0.0;
    StepData.X[1] = 0.0;
    StepData.X[2] = 0.0;

    FILE* NewtonFile = CreateLog();

    SetCurrentTime(0.0);
    UpdateDeviceTime(0.0);

    start = std::clock();
    printf("Run(%e s)\n", TIME * (1e-9));

    char viewerwindow_name[160];
    int viewerwindow = initGlutDisplay(ARG_c, ARG_v);

    initPixelBuffer();

    h = h_h;

    //t=0

    ScheduleSampling(NewtonFile, &Count, &ControlCount,
        Sampling_Period, Out, DATA_d, M_d,
        M, H_d, H, StepData, h);
    Count = 1;
    while (t_h < TIME)
    {
        t_previous = t_h;


        if (StrictSampling == true)
        {
            if ((t_h + h) > Count * Sampling_Period)
            {
                h = Count * Sampling_Period - t_h;
            }
        }

        //final step
        if ((t_h + h) > TIME)
        {
            h = TIME - t_h;
        }

        if (FixedTimeStep == false) {
            SetStepSize(h);
        }

        if (IsFirstStep) {
            ComputeFields(DATA_d, M_d, H_d, P, 0);
            IsFirstStep = false;
        }

        MakeStep_RK(&StepData, &h, M_d, H_d, DATA_d, P, &Flag);

        MaximumErr = StepData.X[0],
        MaxTorque = StepData.X[1],
        MinTorque = StepData.X[2];


        if (Flag == 1) // If error exceeds tolerance, reset has taken place so continue
        {
            ControlCount = ControlCount + 1;
            t_h = t_previous;
            UpdateDeviceTime(t_previous);
            continue;
        }
        else {
            stepcount++;
            consolestepcount++;
            ScheduleSampling(NewtonFile, &Count, &ControlCount,
                Sampling_Period, Out, DATA_d, M_d,
                M, H_d, H, StepData, h);
        }



        if (consolestepcount >= 20)
        {
           // printf("Elapsed time = %e (s) | Stepsize = %e (s) | Max Torque %e | Steps taken =  %d |\r", (1e-9) * t_h, (1e-9) * h, MaxTorque / ((1e3)), stepcount);
            
            consolestepcount = 0;
        }
        if ((((double)std::clock() - (double)start) / (double)CLOCKS_PER_SEC) >= GUI_Sampling_Period)
        {
            sprintf(viewerwindow_name,"MagEx:: t = %.4e (s) | Step = %.4e (s) | dMdt %.4e (T) | Steps = %d |\r",
                                       (1e-9) * t_h, (1e-9) * h, MaxTorque / ((1e3)), stepcount);
            //OpenGL
            CreateTexture(M_d, H_d);
            glutPostRedisplay();
            glutMainLoopEvent();
            glutSetWindowTitle(viewerwindow_name);
            start = std::clock();
            GUIcount++;
        }

    }
    fclose(NewtonFile);
    UpdateDeviceTime(0.0);

    CopyMagFromDevice(M, M_d);
    printf("\n");
    IsFinished = 1;
    // glutDestroyWindow(viewerwindow);
    h_h = 1e-5;
    return;
}
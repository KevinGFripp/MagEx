#include "Print_and_Log_Functions.cuh"
#include "Host_Globals.cuh"
#include "GlobalDefines.cuh"
#include "Array_Indexing_Functions.cuh"
#include "Average_Quantities.cuh"
#include "Data_Transfer_Functions.cuh"

//OOMMF OVF 2.0
__host__ void WriteOVF_Mag_Binary_4(OutputFormat Out, MAG M, int c)
{

    //File writes binary 4, float data type
    if (Out.m_unit == false)
    {
        return;
    }
    FILE* M_ovf;
    char filename[50];
    sprintf(filename, "m_full%d.ovf", c);

    //write header
    M_ovf = fopen(filename, "w+");
    fprintf(M_ovf, "# OOMMF OVF 2.0\n");
    fprintf(M_ovf, "# Segment count: 1\n");
    fprintf(M_ovf, "# Begin: Segment\n");
    fprintf(M_ovf, "# Begin: Header\n");
    fprintf(M_ovf, "# Title: m_full\n");
    fprintf(M_ovf, "# meshtype: rectangular\n");
    fprintf(M_ovf, "# meshunit: m\n");
    fprintf(M_ovf, "# xmin: 0\n");
    fprintf(M_ovf, "# ymin: 0\n");
    fprintf(M_ovf, "# zmin: 0\n");
    fprintf(M_ovf, "# xmax: %e\n", (1e-9) * (CELL_h * NUM_h));
    fprintf(M_ovf, "# ymax: %e\n", (1e-9) * (CELLY_h * NUMY_h));
    fprintf(M_ovf, "# zmax: %e\n", (1e-9) * (CELLZ_h * NUMZ_h));
    fprintf(M_ovf, "# valuedim: 3\n");
    fprintf(M_ovf, "# valuelabels: m_full_x m_full_y m_full_z\n");
    fprintf(M_ovf, "# valueunits: A/m A/m A/m\n");
    fprintf(M_ovf, "# Desc: Total simulation time: %e  s\n", TIME * (1e-9));
    fprintf(M_ovf, "# xbase: %e\n", (1e-9) * (CELL_h / 2.));
    fprintf(M_ovf, "# ybase: %e\n", (1e-9) * (CELLY_h / 2.));
    fprintf(M_ovf, "# zbase: %e\n", (1e-9) * (CELLZ_h / 2.));
    fprintf(M_ovf, "# xnodes: %d\n", NUM_h);
    fprintf(M_ovf, "# ynodes: %d\n", NUMY_h);
    fprintf(M_ovf, "# znodes: %d\n", NUMZ_h);
    fprintf(M_ovf, "# xstepsize: %e\n", (1e-9) * (CELL_h));
    fprintf(M_ovf, "# ystepsize: %e\n", (1e-9) * (CELLY_h));
    fprintf(M_ovf, "# zstepsize: %e\n", (1e-9) * (CELLZ_h));
    fprintf(M_ovf, "# End: Header\n");
    fprintf(M_ovf, "# Begin: Data Binary 4\n");

    fclose(M_ovf);

    //write binary
    M_ovf = fopen(filename, "ab");

    //write data type control number
    const float OOMMF_CONTROL_NUM = 1234567.0;
    fwrite(&OOMMF_CONTROL_NUM, sizeof(float), 1, M_ovf);

    //write array
    //Format expects column-major order, x-y-z triples
    float Mx, My, Mz;
    float Ms = (float)MSAT_h;


    for (int k = 0; k < NUMZ_h; k++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {
            for (int i = 0; i < NUM_h; i++)
            {
                Mx = (float)(M->M[mind_h(0, i, j, k, 0)]);
                Mx *= Ms * 1000.0;
                fwrite(&Mx, sizeof(float), 1, M_ovf);

                My = (float)(M->M[mind_h(0, i, j, k, 1)]);
                My *= Ms * 1000.0;
                fwrite(&My, sizeof(float), 1, M_ovf);

                Mz = (float)(M->M[mind_h(0, i, j, k, 2)]);
                Mz *= Ms * 1000.0;
                fwrite(&Mz, sizeof(float), 1, M_ovf);
            }
        }
    }
    fclose(M_ovf);

    M_ovf = fopen(filename, "a");
    fprintf(M_ovf, " # End: Data Binary 4\n");
    fprintf(M_ovf, "# End: Segment\n");

    fclose(M_ovf);
}
__host__ void WriteOVF_Mag_Binary_8(OutputFormat Out, MAG M, int c)
{

    //File writes binary 8, double data type
    if (Out.m_unit == false)
    {
        return;
    }
    FILE* M_ovf;
    char filename[50];
    sprintf(filename, "m_full%d.ovf", c);

    //write header
    M_ovf = fopen(filename, "w+");
    fprintf(M_ovf, "# OOMMF OVF 2.0\n");
    fprintf(M_ovf, "# Segment count: 1\n");
    fprintf(M_ovf, "# Begin: Segment\n");
    fprintf(M_ovf, "# Begin: Header\n");
    fprintf(M_ovf, "# Title: m_full\n");
    fprintf(M_ovf, "# meshtype: rectangular\n");
    fprintf(M_ovf, "# meshunit: m\n");
    fprintf(M_ovf, "# xmin: 0\n");
    fprintf(M_ovf, "# ymin: 0\n");
    fprintf(M_ovf, "# zmin: 0\n");
    fprintf(M_ovf, "# xmax: %e\n", (1e-9) * (CELL_h * NUM_h));
    fprintf(M_ovf, "# ymax: %e\n", (1e-9) * (CELLY_h * NUMY_h));
    fprintf(M_ovf, "# zmax: %e\n", (1e-9) * (CELLZ_h * NUMZ_h));
    fprintf(M_ovf, "# valuedim: 3\n");
    fprintf(M_ovf, "# valuelabels: m_full_x m_full_y m_full_z\n");
    fprintf(M_ovf, "# valueunits: A/m A/m A/m\n");
    fprintf(M_ovf, "# Desc: Total simulation time: %e  s\n", TIME * (1e-9));
    fprintf(M_ovf, "# xbase: %e\n", (1e-9) * (CELL_h / 2.));
    fprintf(M_ovf, "# ybase: %e\n", (1e-9) * (CELLY_h / 2.));
    fprintf(M_ovf, "# zbase: %e\n", (1e-9) * (CELLZ_h / 2.));
    fprintf(M_ovf, "# xnodes: %d\n", NUM_h);
    fprintf(M_ovf, "# ynodes: %d\n", NUMY_h);
    fprintf(M_ovf, "# znodes: %d\n", NUMZ_h);
    fprintf(M_ovf, "# xstepsize: %e\n", (1e-9) * (CELL_h));
    fprintf(M_ovf, "# ystepsize: %e\n", (1e-9) * (CELLY_h));
    fprintf(M_ovf, "# zstepsize: %e\n", (1e-9) * (CELLZ_h));
    fprintf(M_ovf, "# End: Header\n");
    fprintf(M_ovf, "# Begin: Data Binary 8\n");

    fclose(M_ovf);

    //write binary
    M_ovf = fopen(filename, "ab");

    //write data type control number
    const double OOMMF_CONTROL_NUM = 123456789012345.0;
    fwrite(&OOMMF_CONTROL_NUM, sizeof(double), 1, M_ovf);

    //write array
    //Format expects column-major order, x-y-z triples
    double Mx, My, Mz;


    for (int k = 0; k < NUMZ_h; k++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {
            for (int i = 0; i < NUM_h; i++)
            {

                Mx = (M->M[mind_h(0, i, j, k, 0)] * MSAT_h * (1e3));

                fwrite(&Mx, sizeof(double), 1, M_ovf);

                My = (M->M[mind_h(0, i, j, k, 1)] * MSAT_h * (1e3));

                fwrite(&My, sizeof(double), 1, M_ovf);

                Mz = (M->M[mind_h(0, i, j, k, 2)] * MSAT_h * (1e3));

                fwrite(&Mz, sizeof(double), 1, M_ovf);
            }
        }
    }
    fclose(M_ovf);

    M_ovf = fopen(filename, "a");
    fprintf(M_ovf, "# End: Data Binary 8\n");
    fprintf(M_ovf, "# End: Segment\n");

    fclose(M_ovf);
}
__host__ void ReverseByteOrder_float(char* data)
{
    char temp[4];
    temp[0] = data[0];
    temp[1] = data[1];
    temp[2] = data[2];
    temp[3] = data[3];

    data[0] = temp[3];
    data[1] = temp[2];
    data[2] = temp[1];
    data[3] = temp[0];

}

//Mag Binary File Format (mbf)
__host__ void Write_mbf(OutputFormat Out, MAG M, int c)
{

    //File writes single data type
    if (Out.m_unit == false)
    {
        return;
    }

    FILE* M_mbf;
    char filename[50];
    sprintf(filename, "Mag%d.mbf", c);


    //write binary
    M_mbf = fopen(filename, "ab+");

    //write array
    //Format expects column-major order, x-y-z triples
    float Mx, My, Mz;
    float Ms = (float)MSAT_h;


    for (int k = 0; k < NUMZ_h; k++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {
            for (int i = 0; i < NUM_h; i++)
            {
                Mx = (float)(M->M[mind_h(0, i, j, k, 0)]);
                Mx *= Ms * 1000.0;
                fwrite(&Mx, sizeof(float), 1, M_mbf);

                My = (float)(M->M[mind_h(0, i, j, k, 1)]);
                My *= Ms * 1000.0;
                fwrite(&My, sizeof(float), 1, M_mbf);

                Mz = (float)(M->M[mind_h(0, i, j, k, 2)]);
                Mz *= Ms * 1000.0;
                fwrite(&Mz, sizeof(float), 1, M_mbf);
            }
        }
    }

    fclose(M_mbf);
}
__host__ void Write_mbf_single(OutputFormat Out, MAG M, int c)
{

    //File writes float data type
    if (Out.m_unit == false)
    {
        return;
    }

    FILE* M_mbf;
    char filename[50];
    sprintf(filename, "Mag%d.mbf", c);

    //Write Header
    M_mbf = fopen(filename, "w+");
    fprintf(M_mbf, "#Magnetisation binary format 1.0\n");
    fprintf(M_mbf, "%d\n", NUM_h);
    fprintf(M_mbf, "%d\n", NUMY_h);
    fprintf(M_mbf, "%d\n", NUMZ_h);
    fprintf(M_mbf, "%e\n", (1e-9) * CELL_h);
    fprintf(M_mbf, "%e\n", (1e-9) * CELLY_h);
    fprintf(M_mbf, "%e\n", (1e-9) * CELLZ_h);
    fprintf(M_mbf, "#Begin data\n");
    fclose(M_mbf);

    //write binary
    M_mbf = fopen(filename, "ab");

    //write array
    //Format expects column-major order, x-y-z triples
    float Mx, My, Mz;
    float Ms = (float)MSAT_h;


    for (int k = 0; k < NUMZ_h; k++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {
            for (int i = 0; i < NUM_h; i++)
            {
                Mx = (float)(M->M[mind_h(0, i, j, k, 0)]);
                Mx *= Ms * 1000.0;
                fwrite(&Mx, sizeof(float), 1, M_mbf);

                My = (float)(M->M[mind_h(0, i, j, k, 1)]);
                My *= Ms * 1000.0;
                fwrite(&My, sizeof(float), 1, M_mbf);

                Mz = (float)(M->M[mind_h(0, i, j, k, 2)]);
                Mz *= Ms * 1000.0;
                fwrite(&Mz, sizeof(float), 1, M_mbf);
            }
        }
    }
    fclose(M_mbf);
    M_mbf = fopen(filename, "a");
    fprintf(M_mbf, "\n#End data\n");
    fclose(M_mbf);
}

//Field Binary File Format (Bmf)
__host__ void Write_bmf(OutputFormat Out, FIELD H, int c)
{

    //File writes single data type
    if (Out.B_demag == false)
    {
        return;
    }

    FILE* M_mbf;
    char filename[50];
    sprintf(filename, "Demag%d.bmf", c);


    //write binary
    M_mbf = fopen(filename, "ab+");

    //write array
    //Format expects column-major order, x-y-z triples
    float Hx, Hy, Hz;
 
    for (int k = 0; k < NUMZ_h; k++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {
            for (int i = 0; i < NUM_h; i++)
            {
                Hx = (float)(H->H_m[find_h( i, j, k, 0)]);
                Hx *=mu*(1e-6);
                fwrite(&Hx, sizeof(float), 1, M_mbf);

                Hy = (float)(H->H_m[find_h( i, j, k, 1)]);
                Hy *= mu*(1e-6);
                fwrite(&Hy, sizeof(float), 1, M_mbf);

                Hz = (float)(H->H_m[find_h( i, j, k, 2)]);
                Hz *= mu*(1e-6);
                fwrite(&Hz, sizeof(float), 1, M_mbf);
            }
        }
    }

    fclose(M_mbf);
}
__host__ void Write_emf(OutputFormat Out, FIELD H, int c)
{

    //File writes single data type
    if (Out.B_exch == false)
    {
        return;
    }

    FILE* M_mbf;
    char filename[50];
    sprintf(filename, "Demag%d.bmf", c);


    //write binary
    M_mbf = fopen(filename, "ab+");

    //write array
    //Format expects column-major order, x-y-z triples
    float Hx, Hy, Hz;

    for (int k = 0; k < NUMZ_h; k++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {
            for (int i = 0; i < NUM_h; i++)
            {
                Hx = (float)(H->H_ex[find_h(i, j, k, 0)]);
                Hx *= mu*(1e-6);
                fwrite(&Hx, sizeof(float), 1, M_mbf);

                Hy = (float)(H->H_ex[find_h(i, j, k, 1)]);
                Hy *= mu*(1e-6);
                fwrite(&Hy, sizeof(float), 1, M_mbf);

                Hz = (float)(H->H_ex[find_h(i, j, k, 2)]);
                Hz *= mu*(1e-6);
                fwrite(&Hz, sizeof(float), 1, M_mbf);
            }
        }
    }

    fclose(M_mbf);
}
__host__ void ScheduleSampling(FILE* log, int* Count, int* ControlCount,
    double Period, OutputFormat Out, MEMDATA DATA_d, MAG M_d, MAG M,
    FIELD H_d, FIELD H, Vector Step, double stepsize)
{
    Vector AvgMag;
    double ExEnergy=0.0;
    double DmEnergy=0.0;
    double ZmnEnergy=0.0;
    double AnisEnergy=0.0;
    double TotalEnergy = 0.0;

    if (((*Count) * Period) <= t_h)
    {
        
        AvgMag = AverageMag_Reduction();
        AvgMag.X[0] /= M->NUMCELLS[0];
        AvgMag.X[1] /= M->NUMCELLS[0];
        AvgMag.X[2] /= M->NUMCELLS[0];

        ExEnergy = ExchangeEnergy(M_d, H_d, DATA_d);

        DmEnergy = DemagEnergy(M_d, H_d, DATA_d);  

        if (ExternalField_h == 1 || BiasField_h == 1)
        {
            ZmnEnergy = ZeemanEnergy(M_d, H_d, DATA_d);
        }

        if (UniAnisotropy_h == 1)
        {
            AnisEnergy = UniAnisotropyEnergy(M_d, H_d, DATA_d);
        }

        TotalEnergy = ExEnergy + DmEnergy + ZmnEnergy + AnisEnergy;

        UpdateLog(log, Step.X[1] / ((1e3)), (1e-9) * stepsize,
            (double)(*ControlCount), Step.X[0], ExEnergy, DmEnergy,TotalEnergy,
            AvgMag.X[0], AvgMag.X[1], AvgMag.X[2], (1e-9) * t_h);

        if (Out.m_unit == true)
        {
            CopyMagComponentsFromDevice(M, M_d);        
            Write_mbf(Out, M, *Count);
        }

        if (Out.B_exch == true)
        {
            CopyExchangeFieldFromDevice(H, H_d);
            Write_emf(Out, H, *Count);

        }
        if (Out.B_demag == true)
        {
            CopyDemagComponentsFromDevice(H, H_d);
            Write_bmf(Out, H, *Count);
        }

        *ControlCount = 0;
        (*Count)++;
    }
    return;
}

__host__ FILE* CreateLog()
{
    LoggingStage += 1;
    char fname[100];
    sprintf(fname, "SimLog_Stage%d.txt", LoggingStage);
    FILE* NewtonFile = fopen(fname, "w+");
    fprintf(NewtonFile, "---------------------------------- MagEx:: Log File ------------------------------------------\n");
    fprintf(NewtonFile, "Grid Size: %d x %d x %d\nCell Size: %f (nm), %f (nm), %f (nm)\n",
        NUM_h, NUMY_h, NUMZ_h, CELL_h, CELLY_h, CELLZ_h);
    fprintf(NewtonFile, "Gyromagnetic Ratio: %e\n", (Gamma / mu) * 1e12);
    fprintf(NewtonFile, "Ms: %e A/m\nAex: %e J/m\nGilbert Damping: %f\n", MSAT_h * (1e3), A_ex_h * (1e-18), alpha_h);
    fprintf(NewtonFile, "----------------------------------------------------------------------------------------------\n");
    fprintf(NewtonFile, "MaxTorque    Stepsize     StepReset   MaximumError ExchangeEnergy DemagEnergy");
    fprintf(NewtonFile, " TotalEnergy    MxAvg        MyAvg         MzAvg        Time \n");

    return NewtonFile;
}
__host__ void UpdateLog(FILE* log, double MaxTorque, double step, double ControlCount, double MaxErr,
    double ExEnergy, double DmEnergy, double TotalEnergy, double Mx, double My,
    double Mz, double time)
{
    fprintf(log, "%.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n", MaxTorque, step,
        ControlCount, MaxErr, ExEnergy, DmEnergy, TotalEnergy, Mx, My, Mz, time);
    fflush(log);
}
__host__ OutputFormat NewOutput()
{
    OutputFormat Out;
    Out.m_unit = false;
    Out.B_demag = false;
    Out.B_exch = false;
    Out.B_full = false;
    Out.xrange[0] = 0;
    Out.xrange[1] = NUM_h - 1;
    Out.yrange[0] = 0;
    Out.yrange[1] = NUMY_h - 1;
    Out.zrange[0] = 0;
    Out.zrange[1] = NUMZ_h - 1;
    return Out;
}
__host__ void printDemagTensorSlice(MEMDATA DATA, int k)
{
    FILE* xFile;
    FILE* yFile;
    FILE* zFile;
    char FileNamex[20];
    char FileNamey[20];
    char FileNamez[20];
    sprintf(FileNamex, "%s%d.txt", "kxx", k);
    xFile = fopen(FileNamex, "w+");
    sprintf(FileNamey, "%s%d.txt", "kxy", k);
    yFile = fopen(FileNamey, "w+");
    sprintf(FileNamez, "%s%d.txt", "kxz", k);
    zFile = fopen(FileNamez, "w+");

    for (int i = 0; i < 2 * NUM_h; i++)
    {
        for (int j = 0; j < 2 * NUMY_h; j++)
        {

            fprintf(xFile, "%e ", DATA->kxx[FFTind_h(i, j, k)][0]);
            fprintf(yFile, "%e ", DATA->kxy[FFTind_h(i, j, k)][0]);
            fprintf(zFile, "%e ", DATA->kxz[FFTind_h(i, j, k)][0]);
        }
        fprintf(xFile, "\n");
        fprintf(yFile, "\n");
        fprintf(zFile, "\n");
    }


    fclose(xFile);
    fclose(yFile);
    fclose(zFile);

}
__host__ void printFFT(MEMDATA DATA, int k)
{
    FILE* xFile;
    FILE* yFile;
    FILE* zFile;
    char FileNamex[20];
    char FileNamey[20];
    char FileNamez[20];
    sprintf(FileNamex, "%s%d.txt", "xFFT", k);
    xFile = fopen(FileNamex, "w+");
    sprintf(FileNamey, "%s%d.txt", "yFFT", k);
    yFile = fopen(FileNamey, "w+");
    sprintf(FileNamez, "%s%d.txt", "zFFT", k);
    zFile = fopen(FileNamez, "w+");

    for (int i = 0; i < (2 - PBC_x_h) * NUM_h; i++)
    {
        for (int j = 0; j < (2 - PBC_y_h) * NUMY_h; j++)
        {

            fprintf(xFile, "%10lf ", DATA->xFFT[FFTind_h(i, j, k)][0]);
            fprintf(yFile, "%10lf ", DATA->yFFT[FFTind_h(i, j, k)][0]);
            fprintf(zFile, "%10lf ", DATA->zFFT[FFTind_h(i, j, k)][0]);
        }
        fprintf(xFile, "\n");
        fprintf(yFile, "\n");
        fprintf(zFile, "\n");
    }


    fclose(xFile);
    fclose(yFile);
    fclose(zFile);
}
__host__ void printField(FIELD H, int k)
{
    FILE* HxFile;
    FILE* HyFile;
    FILE* HzFile;
    char FileNamex[20];
    char FileNamey[20];
    char FileNamez[20];
    sprintf(FileNamex, "%s%d.txt", "Hx", k);
    HxFile = fopen(FileNamex, "w+");
    sprintf(FileNamey, "%s%d.txt", "Hy", k);
    HyFile = fopen(FileNamey, "w+");
    sprintf(FileNamez, "%s%d.txt", "Hz", k);
    HzFile = fopen(FileNamez, "w+");


    for (int i = 0; i < NUM_h; i++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {

            fprintf(HxFile, "%lf ", H->H_m[find_h(i, j, k, 0)]);
            fprintf(HyFile, "%lf ", H->H_m[find_h(i, j, k, 1)]);
            fprintf(HzFile, "%lf ", H->H_m[find_h(i, j, k, 2)]);
        }
        fprintf(HxFile, "\n");
        fprintf(HyFile, "\n");
        fprintf(HzFile, "\n");
    }


    fclose(HxFile);
    fclose(HyFile);
    fclose(HzFile);
}
__host__  void printExchangeField(FIELD H, int k)
{
    FILE* HxFile;
    FILE* HyFile;
    FILE* HzFile;
    char FileNamex[20];
    char FileNamey[20];
    char FileNamez[20];
    sprintf(FileNamex, "%s%d.txt", "Hx_ex", k);
    HxFile = fopen(FileNamex, "w+");
    sprintf(FileNamey, "%s%d.txt", "Hy_ex", k);
    HyFile = fopen(FileNamey, "w+");
    sprintf(FileNamez, "%s%d.txt", "Hz_ex", k);
    HzFile = fopen(FileNamez, "w+");

    for (int i = 0; i < NUM_h; i++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {

            fprintf(HxFile, "%lf ", H->H_ex[find_h(i, j, k, 0)]);
            fprintf(HyFile, "%lf ", H->H_ex[find_h(i, j, k, 1)]);
            fprintf(HzFile, "%lf ", H->H_ex[find_h(i, j, k, 2)]);
        }
        fprintf(HxFile, "\n");
        fprintf(HyFile, "\n");
        fprintf(HzFile, "\n");
    }


    fclose(HxFile);
    fclose(HyFile);
    fclose(HzFile);
}
__host__ void printmagnetisationframe(MAG M, int c, int k)
{
    FILE* xFile;
    FILE* yFile;
    FILE* zFile;
    char FileNamex[20];
    char FileNamey[20];
    char FileNamez[20];
    sprintf(FileNamex, "%s%d.txt", "Mx", c);
    xFile = fopen(FileNamex, "w+");
    sprintf(FileNamey, "%s%d.txt", "My", c);
    yFile = fopen(FileNamey, "w+");
    sprintf(FileNamez, "%s%d.txt", "Mz", c);
    zFile = fopen(FileNamez, "w+");


    for (int i = 0; i < NUM_h; i++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {

            fprintf(xFile, "%lf ", M->M[mind_h(0, i, j, k, 0)]);
            fprintf(yFile, "%lf ", M->M[mind_h(0, i, j, k, 1)]);
            fprintf(zFile, "%lf ", M->M[mind_h(0, i, j, k, 2)]);

        }
        fprintf(xFile, "\n");
        fprintf(yFile, "\n");
        fprintf(zFile, "\n");
    }
    fclose(xFile);
    fclose(yFile);
    fclose(zFile);
    return;
}
__host__ void printDemagTensorFFT(MEMDATA DATA, int k)
{
    FILE* xFile;
    FILE* yFile;
    FILE* zFile;

    FILE* x2File;
    FILE* y2File;
    FILE* z2File;
    char FileNamex[20];
    char FileNamey[20];
    char FileNamez[20];
    char FileNamex2[20];
    char FileNamey2[20];
    char FileNamez2[20];

    sprintf(FileNamex, "%s%d.txt", "kxx", k);
    xFile = fopen(FileNamex, "w+");

    sprintf(FileNamey, "%s%d.txt", "kxy", k);
    yFile = fopen(FileNamey, "w+");

    sprintf(FileNamez, "%s%d.txt", "kxz", k);
    zFile = fopen(FileNamez, "w+");

    sprintf(FileNamex2, "%s%d.txt", "kyy", k);
    x2File = fopen(FileNamex2, "w+");

    sprintf(FileNamey2, "%s%d.txt", "kyz", k);
    y2File = fopen(FileNamey2, "w+");

    sprintf(FileNamez2, "%s%d.txt", "kzz", k);
    z2File = fopen(FileNamez2, "w+");

    for (int i = 0; i < (2) * NUM_h; i++)
    {
        for (int j = 0; j < (2) * NUMY_h; j++)
        {

            fprintf(xFile, "%e ", DATA->kxx[FFTind_h(i, j, k)][0]);
            fprintf(yFile, "%e ", DATA->kxy[FFTind_h(i, j, k)][0]);
            fprintf(zFile, "%e ", DATA->kxz[FFTind_h(i, j, k)][0]);

            fprintf(x2File, "%e ", DATA->kyy[FFTind_h(i, j, k)][0]);
            fprintf(y2File, "%e ", DATA->kyz[FFTind_h(i, j, k)][0]);
            fprintf(z2File, "%e ", DATA->kzz[FFTind_h(i, j, k)][0]);
        }
        fprintf(xFile, "\n");
        fprintf(yFile, "\n");
        fprintf(zFile, "\n");

        fprintf(x2File, "\n");
        fprintf(y2File, "\n");
        fprintf(z2File, "\n");
    }


    fclose(xFile);
    fclose(yFile);
    fclose(zFile);
    fclose(x2File);
    fclose(y2File);
    fclose(z2File);
}
__host__ void printmagnetisation_Outputs(MAG M, int c, OutputFormat Out)
{
    if (Out.m_unit == false)
    {
        return;
    }

    FILE* xFile;
    FILE* yFile;
    FILE* zFile;
    char FileNamex[20];
    char FileNamey[20];
    char FileNamez[20];

    sprintf(FileNamex, "%s%d.txt", "Mx", c);
    xFile = fopen(FileNamex, "w+");
    sprintf(FileNamey, "%s%d.txt", "My", c);
    yFile = fopen(FileNamey, "w+");
    sprintf(FileNamez, "%s%d.txt", "Mz", c);
    zFile = fopen(FileNamez, "w+");


    for (int i = Out.xrange[0]; i <= Out.xrange[1]; i++)
    {
        if (i >= NUM_h)
        {
            continue;
        }
        if (i < 0)
        {
            i = 0;
        }

        for (int j = Out.yrange[0]; j <= Out.yrange[1]; j++)
        {
            if (j >= NUMY_h)
            {
                continue;
            }
            if (j < 0)
            {
                j = 0;
            }

            fprintf(xFile, "%.16e ", 1000 * MSAT_h * M->M[mind_h(0, i, j, Out.zrange[0], 0)]);
            fprintf(yFile, "%.16e ", 1000 * MSAT_h * M->M[mind_h(0, i, j, Out.zrange[0], 1)]);
            fprintf(zFile, "%.16e ", 1000 * MSAT_h * M->M[mind_h(0, i, j, Out.zrange[0], 2)]);

        }
        fprintf(xFile, "\n");
        fprintf(yFile, "\n");
        fprintf(zFile, "\n");
    }
    fclose(xFile);
    fclose(yFile);
    fclose(zFile);
    return;
}
__host__  void printExchangeField_Outputs(FIELD H, int c, OutputFormat Out)
{
    if (Out.B_exch == false)
    {
        return;
    }

    FILE* HxFile;
    FILE* HyFile;
    FILE* HzFile;
    char FileNamex[20];
    char FileNamey[20];
    char FileNamez[20];
    sprintf(FileNamex, "%s%d.txt", "Bx_ex", c);
    HxFile = fopen(FileNamex, "w+");
    sprintf(FileNamey, "%s%d.txt", "By_ex", c);
    HyFile = fopen(FileNamey, "w+");
    sprintf(FileNamez, "%s%d.txt", "Bz_ex", c);
    HzFile = fopen(FileNamez, "w+");

    for (int i = Out.xrange[0]; i <= Out.xrange[1]; i++)
    {
        if (i >= NUM_h)
        {
            continue;
        }
        if (i < 0)
        {
            i = 0;
        }
        for (int j = Out.yrange[0]; j <= Out.yrange[1]; j++)
        {
            if (j >= NUMY_h)
            {
                continue;
            }
            if (j < 0)
            {
                j = 0;
            }

            fprintf(HxFile, "%lf ", 1000 * mu * (1e-6) * H->H_ex[find_h(i, j, Out.zrange[0], 0)]);
            fprintf(HyFile, "%lf ", 1000 * mu * (1e-6) * H->H_ex[find_h(i, j, Out.zrange[0], 1)]);
            fprintf(HzFile, "%lf ", 1000 * mu * (1e-6) * H->H_ex[find_h(i, j, Out.zrange[0], 2)]);
        }
        fprintf(HxFile, "\n");
        fprintf(HyFile, "\n");
        fprintf(HzFile, "\n");
    }


    fclose(HxFile);
    fclose(HyFile);
    fclose(HzFile);
}
__host__  void printDemagField_Outputs(FIELD H, int c, OutputFormat Out)
{
    if (Out.B_demag == false)
    {
        return;
    }
    FILE* HxFile;
    FILE* HyFile;
    FILE* HzFile;
    char FileNamex[20], FileNamey[20], FileNamez[20];
    sprintf(FileNamex, "%s%d.txt", "Bmx", c);
    HxFile = fopen(FileNamex, "w+");
    sprintf(FileNamey, "%s%d.txt", "Bmy", c);
    HyFile = fopen(FileNamey, "w+");
    sprintf(FileNamez, "%s%d.txt", "Bmz", c);
    HzFile = fopen(FileNamez, "w+");

    for (int i = Out.xrange[0]; i <= Out.xrange[1]; i++)
    {
        if (i >= NUM_h)
        {
            continue;
        }
        if (i < 0)
        {
            i = 0;
        }
        for (int j = Out.yrange[0]; j <= Out.yrange[1]; j++)
        {
            if (j >= NUMY_h)
            {
                continue;
            }
            if (j < 0)
            {
                j = 0;
            }

            fprintf(HxFile, "%.16e ", 1000 * mu * (1e-6) * H->H_m[find_h(i, j, Out.zrange[0], 0)]);
            fprintf(HyFile, "%.16e ", 1000 * mu * (1e-6) * H->H_m[find_h(i, j, Out.zrange[0], 1)]);
            fprintf(HzFile, "%.16e ", 1000 * mu * (1e-6) * H->H_m[find_h(i, j, Out.zrange[0], 2)]);
        }
        fprintf(HxFile, "\n"), fprintf(HyFile, "\n"), fprintf(HzFile, "\n");
    }

    fclose(HxFile), fclose(HyFile), fclose(HzFile);
}
__host__  void printEffectiveField_Outputs(FIELD H, int c, OutputFormat Out)
{
    if (Out.B_full == false)
    {
        return;
    }

    FILE* HxFile;
    FILE* HyFile;
    FILE* HzFile;
    char FileNamex[20];
    char FileNamey[20];
    char FileNamez[20];
    sprintf(FileNamex, "%s%d.txt", "Heff_x", c);
    HxFile = fopen(FileNamex, "w+");
    sprintf(FileNamey, "%s%d.txt", "Heff_y", c);
    HyFile = fopen(FileNamey, "w+");
    sprintf(FileNamez, "%s%d.txt", "Heff_z", c);
    HzFile = fopen(FileNamez, "w+");

    for (int i = Out.xrange[0]; i <= Out.xrange[1]; i++)
    {
        if (i >= NUM_h)
        {
            continue;
        }
        if (i < 0)
        {
            i = 0;
        }
        for (int j = Out.yrange[0]; j <= Out.yrange[1]; j++)
        {
            if (j >= NUMY_h)
            {
                continue;
            }
            if (j < 0)
            {
                j = 0;
            }

            fprintf(HxFile, "%lf ", H->H_eff[find_h(i, j, Out.zrange[0], 0)]);
            fprintf(HyFile, "%lf ", H->H_eff[find_h(i, j, Out.zrange[0], 1)]);
            fprintf(HzFile, "%lf ", H->H_eff[find_h(i, j, Out.zrange[0], 2)]);
        }
        fprintf(HxFile, "\n");
        fprintf(HyFile, "\n");
        fprintf(HzFile, "\n");
    }


    fclose(HxFile);
    fclose(HyFile);
    fclose(HzFile);
}
__host__ void printmagnetisationFull_Outputs(MAG M, int c, OutputFormat Out)
{
    if (Out.m_unit == false)
    {
        return;
    }
    FILE* xFile;

    char FileNamex[20];
    sprintf(FileNamex, "%s%d.txt", "Mfull", c);
    xFile = fopen(FileNamex, "w+");

    for (int i = Out.xrange[0]; i <= Out.xrange[1]; i++)
    {
        if (i >= NUM_h)
        {
            continue;
        }
        if (i < 0)
        {
            i = 0;
        }

        for (int j = Out.yrange[0]; j <= Out.yrange[1]; j++)
        {
            if (j >= NUMY_h)
            {
                continue;
            }
            if (j < 0)
            {
                j = 0;
            }

            for (int k = Out.zrange[0]; k <= Out.zrange[1]; k++)
            {
                if (k >= NUMZ_h)
                {
                    continue;
                }
                if (k < 0)
                {
                    k = 0;
                }

                fprintf(xFile, "%.16e %.16e %.16e ", MSAT_h * (1e3) * M->M[mind_h(0, i, j, k, 0)],
                    MSAT_h * (1e3) * M->M[mind_h(0, i, j, k, 1)],
                    MSAT_h * (1e3) * M->M[mind_h(0, i, j, k, 2)]);
            }
        }
        fprintf(xFile, "\n");
    }
    fclose(xFile);
    return;
}

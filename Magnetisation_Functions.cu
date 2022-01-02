#include "Magnetisation_Functions.cuh"

__host__ void MeshBoundaries(MAG M)
{
    int dx = 0, dy = 0, dz = 0;
    int Fx = 0, Fy = 0, Fz = 0;
    double norm = 0.0;
    double m1, m2;
   // printf("Computing boundary conditions... \n");
    for (int i = 0; i < NUM_h; i++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {
            for (int k = 0; k < NUMZ_h; k++)
            {
                norm = sqrt(M->M[mind_h(0, i, j, k, 0)] * M->M[mind_h(0, i, j, k, 0)]
                    + M->M[mind_h(0, i, j, k, 1)] * M->M[mind_h(0, i, j, k, 1)]
                    + M->M[mind_h(0, i, j, k, 2)] * M->M[mind_h(0, i, j, k, 2)]);

                if (M->Mat[ind_h(i, j, k)] == 0 && norm < (1e-10))
                {
                    continue;
                }
                if (i == 0)
                {
                    M->Bd[bind_h(i, j, k, 0)] = -1;
                    Fx = 1;
                }

                if (j == 0)
                {
                    M->Bd[bind_h(i, j, k, 1)] = -1;
                    Fy = 1;
                }
                if (k == 0)
                {
                    M->Bd[bind_h(i, j, k, 2)] = -1;
                    Fz = 1;
                }
                if (i == NUM_h - 1)
                {
                    M->Bd[bind_h(i, j, k, 0)] = 1;
                    Fx = 1;
                }

                if (j == (NUMY_h - 1))
                {
                    M->Bd[bind_h(i, j, k, 1)] = 1;
                    Fy = 1;
                }

                if (k == NUMZ_h - 1)
                {
                    M->Bd[bind_h(i, j, k, 2)] = 1;
                    Fz = 1;
                }
                if (Fx == 0)
                {   
                    m1 = M->Mat[ind_h(i - 1, j, k)];
                    m2 = M->Mat[ind_h(i + 1, j, k)];
                    m1 != 0 ? (m1 /= m1) :(m1=m1);
                    m2 != 0 ? (m2 /= m2) : (m2 = m2);
                    dx = m1 - m2;
                    M->Bd[bind_h(i, j, k, 0)] = dx;
                }
                else {}

                if (Fy == 0)
                {
                    m1 = M->Mat[ind_h(i, j - 1, k)];
                    m2 = M->Mat[ind_h(i, j + 1, k)];
                    m1 != 0 ? (m1 /= m1) : (m1 = m1);
                    m2 != 0 ? (m2 /= m2) : (m2 = m2);
                    dy =m1 - m2;
                    M->Bd[bind_h(i, j, k, 1)] = dy;
                }
                else {}

                if (Fz == 0)
                {
                    m1 = M->Mat[ind_h(i, j, k - 1)];
                    m2 = M->Mat[ind_h(i, j, k + 1)];
                    m1 != 0 ? (m1 /= m1) : (m1 = m1);
                    m2 != 0 ? (m2 /= m2) : (m2 = m2);
                    dz = m1 - m2;
                    M->Bd[bind_h(i, j, k, 2)] = dz;
                }
                else {}
                Fx = 0, Fy = 0, Fz = 0;
                m1 = 0.0, m2 = 0.0;
            }
        }
    }
}
__host__ void MagnetiseFilm3D(MAG M, double L, double H, double D, int Flag)
{
    double Magnitude = 0.0;
    for (int i = 0; i < NUM_h; i++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {
            for (int k = 0; k < NUMZ_h; k++)
            {
                if ((fabs((i - floor((double)NUM_h / 2))) * CELL_h) <= (L) / 2
                    && (fabs((j - floor((double)NUMY_h / 2))) * CELLY_h) <= (H) / 2
                    && (k * CELLZ_h <= D))
                {
                    if (Flag == 1)
                    {
                        M->M[mind_h(0, i, j, k, 0)] = 1.0;
                        M->M[mind_h(0, i, j, k, 1)] = 0.1;
                        M->M[mind_h(0, i, j, k, 2)] = 0.0;
                    }
                    if (Flag == 2)
                    {
                        M->M[mind_h(0, i, j, k, 0)] = 0.0;
                        M->M[mind_h(0, i, j, k, 1)] = -cos((M_PI / NUMY_h) * j);
                        M->M[mind_h(0, i, j, k, 2)] = cos((M_PI / NUM_h) * i);
                    }
                    Magnitude = sqrt(pow(M->M[mind_h(0, i, j, k, 0)], 2)
                        + pow(M->M[mind_h(0, i, j, k, 1)], 2)
                        + pow(M->M[mind_h(0, i, j, k, 2)], 2));

                    M->M[mind_h(0, i, j, k, 0)] /= Magnitude;
                    M->M[mind_h(0, i, j, k, 1)] /= Magnitude;
                    M->M[mind_h(0, i, j, k, 2)] /= Magnitude;

                    M->Mat[ind_h(i, j, k)] = 1;
                    M->NUMCELLS[0] += 1;
                }
            }
        }
    }
    return;
}
__host__ void MagnetiseDisk3D(MAG M, double RADIUS, double HEIGHT, int Flag)
{
    double Magnitude = 0.0;
    for (int i = 0; i < NUM_h; i++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {
            for (int k = 0; k < NUMZ_h; k++)
            {
                if ((pow(((double)i * CELL_h - ((double)NUM_h / 2.) * CELL_h), 2)
                    + pow(((double)j * CELLY_h - ((double)NUMY_h / 2.) * CELLY_h), 2))
                    <= pow(RADIUS, 2) && k * CELLZ_h <= HEIGHT)
                {
                    if (Flag == 1)
                    {
                        M->M[mind_h(0, i, j, k, 0)] = 1.0;
                        M->M[mind_h(0, i, j, k, 1)] = 1.0;
                        M->M[mind_h(0, i, j, k, 2)] = 1.0;
                    }
                    Magnitude = sqrt(pow(M->M[mind_h(0, i, j, k, 0)], 2) + pow(M->M[mind_h(0, i, j, k, 1)], 2)
                        + pow(M->M[mind_h(0, i, j, k, 2)], 2));

                    M->M[mind_h(0, i, j, k, 0)] /= Magnitude;
                    M->M[mind_h(0, i, j, k, 1)] /= Magnitude;
                    M->M[mind_h(0, i, j, k, 2)] /= Magnitude;

                    M->M[mind_h(1, i, j, k, 0)] = M->M[mind_h(0, i, j, k, 0)];
                    M->M[mind_h(1, i, j, k, 1)] = M->M[mind_h(0, i, j, k, 1)];
                    M->M[mind_h(1, i, j, k, 2)] = M->M[mind_h(0, i, j, k, 2)];

                    M->Mat[ind_h(i, j, k)] = 1;
                    M->NUMCELLS[0]++;
                }
            }
        }
    }
    
    return;
}
__host__ void MagnetiseFilm3D_bool(MAG M, double L, double H, double D,
    int xpos, int ypos, int zpos, bool add)
{
    double Magnitude = 0.0;
    for (int i = (xpos - (int)(floor((double)((L / 2) / CELL_h)))); i < (xpos + (int)(floor((double)((L / 2) / CELL_h)))); i++)
    {
        for (int j = (ypos - (int)(floor((double)((H / 2) / CELLY_h)))); j < (ypos + (int)(floor((double)((H / 2) / CELLY_h)))); j++)
        {
            for (int k = (zpos - (int)(floor((double)((D / 2) / CELLZ_h)))); k < (zpos + (int)(floor((double)((D / 2) / CELLZ_h)))); k++)
            {
                {
                    if (add == true)
                    {
                        M->M[mind_h(0, i, j, k, 0)] = 1.0;
                        M->M[mind_h(0, i, j, k, 1)] = 1.0;
                        M->M[mind_h(0, i, j, k, 2)] = 1.0;
                        Magnitude = sqrt(pow(M->M[mind_h(0, i, j, k, 0)], 2)
                            + pow(M->M[mind_h(0, i, j, k, 1)], 2)
                            + pow(M->M[mind_h(0, i, j, k, 2)], 2));

                        M->M[mind_h(0, i, j, k, 0)] /= Magnitude;
                        M->M[mind_h(0, i, j, k, 1)] /= Magnitude;
                        M->M[mind_h(0, i, j, k, 2)] /= Magnitude;

                        M->Mat[ind_h(i, j, k)] = 1;
                        M->NUMCELLS[0] += 1;
                    }
                    else {
                        M->M[mind_h(0, i, j, k, 0)] = 0.0;
                        M->M[mind_h(0, i, j, k, 1)] = 0.0;
                        M->M[mind_h(0, i, j, k, 2)] = 0.0;

                        M->Mat[ind_h(i, j, k)] = 0;
                        if (M->NUMCELLS[0] == 0)
                        {
                        }
                        else
                        {
                            M->NUMCELLS[0] -= 1;
                        }
                    }
                }
            }
        }
    }
    return;
}
__host__ void MagnetiseDisk3D_bool(MAG M, double RADIUS, double HEIGHT,
    int xpos, int ypos, int zpos, bool add)
{
    double Magnitude = 0.0;
    for (int i = 0; i < NUM_h; i++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {
            for (int k = (zpos - (HEIGHT / 2) / CELLZ_h); k <= (zpos + (HEIGHT / 2) / CELLZ_h); k++)
            {
                if ((pow(((double)i * CELL_h - (xpos)*CELL_h), 2)
                    + pow(((double)j * CELLY_h - (ypos)*CELLY_h), 2))
                    < pow(RADIUS, 2))
                {
                    if (add == true)
                    {
                        M->M[mind_h(0, i, j, k, 0)] = 1.0;
                        M->M[mind_h(0, i, j, k, 1)] = 1.0;
                        M->M[mind_h(0, i, j, k, 2)] = 1.0;
                        Magnitude = sqrt(pow(M->M[mind_h(0, i, j, k, 0)], 2) + pow(M->M[mind_h(0, i, j, k, 1)], 2)
                            + pow(M->M[mind_h(0, i, j, k, 2)], 2));

                        M->M[mind_h(0, i, j, k, 0)] /= Magnitude;
                        M->M[mind_h(0, i, j, k, 1)] /= Magnitude;
                        M->M[mind_h(0, i, j, k, 2)] /= Magnitude;



                        M->Mat[ind_h(i, j, k)] = 1;
                        M->NUMCELLS[0] += 1;
                    }
                    else {
                        M->M[mind_h(0, i, j, k, 0)] = 0.0;
                        M->M[mind_h(0, i, j, k, 1)] = 0.0;
                        M->M[mind_h(0, i, j, k, 2)] = 0.0;
                        M->Mat[ind_h(i, j, k)] = 0;
                        if (M->NUMCELLS[0] == 0)
                        {
                        }
                        else {
                            M->NUMCELLS[0] -= 1;
                        }
                    }
                }
            }
        }
    }
    return;
}
__host__ void UniformState(MAG M, int x, int y, int z)
{
    double norm = sqrt((double)x * x + (double)y * y + (double)z * z);
    for (int i = 0; i < NUM_h; i++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {
            for (int k = 0; k < NUMZ_h; k++)
            {
                if (M->Mat[ind_h(i, j, k)] == 0)
                {
                    continue;
                }

                M->M[mind_h(0, i, j, k, 0)] = (double)x;
                M->M[mind_h(0, i, j, k, 1)] = (double)y;
                M->M[mind_h(0, i, j, k, 2)] = (double)z;

                M->M[mind_h(0, i, j, k, 0)] /= norm;
                M->M[mind_h(0, i, j, k, 1)] /= norm;
                M->M[mind_h(0, i, j, k, 2)] /= norm;

                M->M[mind_h(1, i, j, k, 0)] = M->M[mind_h(0, i, j, k, 0)];
                M->M[mind_h(1, i, j, k, 1)] = M->M[mind_h(0, i, j, k, 1)];
                M->M[mind_h(1, i, j, k, 2)] = M->M[mind_h(0, i, j, k, 2)];

                M->M[mind_h(2, i, j, k, 0)] = M->M[mind_h(0, i, j, k, 0)];
                M->M[mind_h(2, i, j, k, 1)] = M->M[mind_h(0, i, j, k, 1)];
                M->M[mind_h(2, i, j, k, 2)] = M->M[mind_h(0, i, j, k, 2)];
            }
        }
    }
    return;
}
__host__ void UniformState_InRegion(MAG M, int x, int y, int z, Region R)
{
    double Magnitude;
    for (int i = R.x[0]; i <= R.x[1]; i++)
    {
        for (int j = R.y[0]; j <= R.y[1]; j++)
        {
            for (int k = R.z[0]; k <= R.z[1]; k++)
            {
                if (M->Mat[ind_h(i, j, k)] == 0)
                {
                    continue;
                }

                if (i >= NUM_h || j >= NUMY_h || k >= NUMZ_h)
                {
                    continue;
                }

                M->M[mind_h(0, i, j, k, 0)] = (double)x;
                M->M[mind_h(0, i, j, k, 1)] = (double)y;
                M->M[mind_h(0, i, j, k, 2)] = (double)z;

                if (x == 0 && y == 0 && z == 0)
                {
                    M->M[mind_h(0, i, j, k, 0)] = 1.0;
                    M->M[mind_h(0, i, j, k, 1)] = 1.0;
                    M->M[mind_h(0, i, j, k, 2)] = 1.0;
                }

                Magnitude = sqrt(pow((double)x, 2) + pow((double)y, 2) + pow((double)z, 2));

                M->M[mind_h(0, i, j, k, 0)] /= Magnitude;
                M->M[mind_h(0, i, j, k, 1)] /= Magnitude;
                M->M[mind_h(0, i, j, k, 2)] /= Magnitude;

                M->M[mind_h(1, i, j, k, 0)] = M->M[mind_h(0, i, j, k, 0)];
                M->M[mind_h(1, i, j, k, 1)] = M->M[mind_h(0, i, j, k, 1)];
                M->M[mind_h(1, i, j, k, 2)] = M->M[mind_h(0, i, j, k, 2)];
            }
        }
    }
    return;
}
__host__ void VortexState(MAG M, int x, int y, int z)
{
    double Magnitude;
    for (int i = 0; i < NUM_h; i++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {
            for (int k = 0; k < NUMZ_h; k++)
            {
                if (M->Mat[ind_h(i, j, k)] == 0)
                {
                    continue;
                }
                if (z != 0 && y != 0) // Vortex in yz plane
                {
                    M->M[mind_h(0, i, j, k, 0)] = -0.1;
                    M->M[mind_h(0, i, j, k, 1)] = -cos((M_PI / NUMY_h) * k);
                    M->M[mind_h(0, i, j, k, 2)] = cos((M_PI / NUM_h) * j);
                }

                if (z != 0 && x != 0) // Vortex in xz plane
                {
                    M->M[mind_h(0, i, j, k, 1)] = -0.1;
                    M->M[mind_h(0, i, j, k, 0)] = -cos((M_PI / NUMY_h) * k);
                    M->M[mind_h(0, i, j, k, 2)] = cos((M_PI / NUM_h) * i);
                }

                if (x != 0 && y != 0) // Vortex in xy plane
                {
                    M->M[mind_h(0, i, j, k, 2)] = exp(-pow((j - NUMY_h / 2), 2)) * exp(-pow((i - NUM_h / 2), 2));
                    M->M[mind_h(0, i, j, k, 0)] = cos((M_PI / NUMY_h) * j);
                    M->M[mind_h(0, i, j, k, 1)] = -cos((M_PI / NUM_h) * i);
                }

                if (x == 0 && y == 0 && z == 0)
                {
                    M->M[mind_h(0, i, j, k, 2)] = -0.1;
                    M->M[mind_h(0, i, j, k, 0)] = -cos((M_PI / NUMY_h) * j);
                    M->M[mind_h(0, i, j, k, 1)] = cos((M_PI / NUM_h) * i);
                }

                if (x != 0 && y != 0 && z != 0)
                {
                    M->M[mind_h(0, i, j, k, 2)] = -0.1;
                    M->M[mind_h(0, i, j, k, 0)] = -cos((M_PI / NUMY_h) * j);
                    M->M[mind_h(0, i, j, k, 1)] = cos((M_PI / NUM_h) * i);
                }

                Magnitude = sqrt(pow(M->M[mind_h(0, i, j, k, 0)], 2)
                    + pow(M->M[mind_h(0, i, j, k, 1)], 2)
                    + pow(M->M[mind_h(0, i, j, k, 2)], 2));

                M->M[mind_h(0, i, j, k, 0)] /= Magnitude;
                M->M[mind_h(0, i, j, k, 1)] /= Magnitude;
                M->M[mind_h(0, i, j, k, 2)] /= Magnitude;

                M->M[mind_h(1, i, j, k, 0)] = M->M[mind_h(0, i, j, k, 0)];
                M->M[mind_h(1, i, j, k, 1)] = M->M[mind_h(0, i, j, k, 1)];
                M->M[mind_h(1, i, j, k, 2)] = M->M[mind_h(0, i, j, k, 2)];
                M->M[mind_h(2, i, j, k, 0)] = M->M[mind_h(0, i, j, k, 0)];
                M->M[mind_h(2, i, j, k, 1)] = M->M[mind_h(0, i, j, k, 1)];
                M->M[mind_h(2, i, j, k, 2)] = M->M[mind_h(0, i, j, k, 2)];

            }
        }
    }
}
__host__ void VortexState_InRegion(MAG M, int x, int y, int z, Region R)
{
    double Magnitude;
    int i_midpoint = R.x[1] / 2;
    int j_midpoint = R.y[1] / 2;
    int k_midpoint = R.z[1] / 2;

    for (int i = R.x[0]; i <= R.x[1]; i++)
    {
        for (int j = R.y[0]; j <= R.y[1]; j++)
        {
            for (int k = R.z[0]; k <= R.z[1]; k++)
            {
                if (M->Mat[ind_h(i, j, k)] == 0)
                {
                    continue;
                }

                if (i >= NUM_h || j >= NUMY_h || k >= NUMZ_h)
                {
                    continue;
                }

                if (z != 0 && y != 0) // Vortex in yz plane
                {
                    M->M[mind_h(0, i, j, k, 0)] = -0.1;
                    M->M[mind_h(0, i, j, k, 1)] = -cos((M_PI / NUMY_h) * ((double)k - (double)k_midpoint));
                    M->M[mind_h(0, i, j, k, 2)] = cos((M_PI / NUM_h) * ((double)j - (double)j_midpoint));
                }

                if (z != 0 && x != 0) // Vortex in xz plane
                {
                    M->M[mind_h(0, i, j, k, 1)] = -0.1;
                    M->M[mind_h(0, i, j, k, 0)] = -cos((M_PI / NUMY_h) * ((double)k - (double)k_midpoint));
                    M->M[mind_h(0, i, j, k, 2)] = cos((M_PI / NUM_h) * ((double)i - (double)i_midpoint));
                }

                if (x != 0 && y != 0 && z == 0) // Vortex in xy plane
                {
                    M->M[mind_h(0, i, j, k, 2)] = exp(-pow(((j - R.y[0]) / 2), 2)) * exp(-pow(((i - R.x[0]) / 2), 2));
                    M->M[mind_h(0, i, j, k, 0)] = cos((M_PI / ((double)R.y[1] - R.y[0])) * ((double)j - R.y[0]));
                    M->M[mind_h(0, i, j, k, 1)] = -cos((M_PI / ((double)R.x[1] - R.x[0])) * ((double)i - R.x[0]));
                }

                if (x == 0 && y == 0 && z == 0)
                {
                    M->M[mind_h(0, i, j, k, 2)] = -0.1;
                    M->M[mind_h(0, i, j, k, 0)] = -cos((M_PI / NUMY_h) * j);
                    M->M[mind_h(0, i, j, k, 1)] = cos((M_PI / NUM_h) * i);
                }

                if (x != 0 && y != 0 && z != 0)
                {
                    M->M[mind_h(0, i, j, k, 2)] = -0.1;
                    M->M[mind_h(0, i, j, k, 0)] = -cos((M_PI / NUMY_h) * j);
                    M->M[mind_h(0, i, j, k, 1)] = cos((M_PI / NUM_h) * i);
                }

                Magnitude = sqrt(pow(M->M[mind_h(0, i, j, k, 0)], 2)
                    + pow(M->M[mind_h(0, i, j, k, 1)], 2)
                    + pow(M->M[mind_h(0, i, j, k, 2)], 2));

                M->M[mind_h(0, i, j, k, 0)] /= Magnitude;
                M->M[mind_h(0, i, j, k, 1)] /= Magnitude;
                M->M[mind_h(0, i, j, k, 2)] /= Magnitude;

                M->M[mind_h(1, i, j, k, 0)] = M->M[mind_h(0, i, j, k, 0)];
                M->M[mind_h(1, i, j, k, 1)] = M->M[mind_h(0, i, j, k, 1)];
                M->M[mind_h(1, i, j, k, 2)] = M->M[mind_h(0, i, j, k, 2)];
            }
        }
    }
}
__host__ void MagnetisationInitialise(MAG M)
{
    (M->NUMCELLS[0]) = 0;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < NUM_h; i++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {
            for (int k = 0; k < NUMZ_h; k++)
            {
                M->Mat[ind_h(i, j, k)] = 0;

                for (int n = 0; n < DIM; n++)
                {
                    M->M[mind_h(0, i, j, k, n)] = 0.0;
                    M->M[mind_h(1, i, j, k, n)] = 0.0;
                    M->M[mind_h(2, i, j, k, n)] = 0.0;
                    M->Bd[bind_h(i, j, k, n)] = 0;
                }
            }
        }
    }

    return;
}
__host__ void MagnetiseFilm3D_InRegion(MAG M, Region R)
{
    double Magnitude = 0.0;
    for (int i = R.x[0]; i <= (R.x[1] % NUM_h); i++)
    {
        for (int j = R.y[0]; j <= (R.y[1] % NUMY_h); j++)
        {
            for (int k = R.z[0]; k <= (R.z[1] % NUMZ_h); k++)
            {

                {
                    M->M[mind_h(0, i, j, k, 0)] = 1.0;
                    M->M[mind_h(0, i, j, k, 1)] = 1.0;
                    M->M[mind_h(0, i, j, k, 2)] = 1.0;
                    Magnitude = sqrt(pow(M->M[mind_h(0, i, j, k, 0)], 2)
                        + pow(M->M[mind_h(0, i, j, k, 1)], 2)
                        + pow(M->M[mind_h(0, i, j, k, 2)], 2));

                    M->M[mind_h(0, i, j, k, 0)] /= Magnitude;
                    M->M[mind_h(0, i, j, k, 1)] /= Magnitude;
                    M->M[mind_h(0, i, j, k, 2)] /= Magnitude;

                    M->Mat[ind_h(i, j, k)] = 1;
                    M->NUMCELLS[0] += 1;
                }
            }
        }
    }

    return;
}
__host__ void MagnetiseFilm3D_InRegion_bool(MAG M, Region R, int A)
{
    double Magnitude = 0.0;
    for (int i = R.x[0]; i <= (R.x[1] % NUM_h); i++)
    {
        for (int j = R.y[0]; j <= (R.y[1] % NUMY_h); j++)
        {
            for (int k = R.z[0]; k <= (R.z[1] % NUMZ_h); k++)
            {
                if (A == M_ADD)
                {
                    M->M[mind_h(0, i, j, k, 0)] = 1.0;
                    M->M[mind_h(0, i, j, k, 1)] = 1.0;
                    M->M[mind_h(0, i, j, k, 2)] = 1.0;
                    Magnitude = sqrt(pow(M->M[mind_h(0, i, j, k, 0)], 2)
                        + pow(M->M[mind_h(0, i, j, k, 1)], 2)
                        + pow(M->M[mind_h(0, i, j, k, 2)], 2));

                    M->M[mind_h(0, i, j, k, 0)] /= Magnitude;
                    M->M[mind_h(0, i, j, k, 1)] /= Magnitude;
                    M->M[mind_h(0, i, j, k, 2)] /= Magnitude;

                    M->Mat[ind_h(i, j, k)] = 1;
                    M->NUMCELLS[0] += 1;
                }

                if (A == M_SUB)
                {
                    if (M->Mat[ind_h(i, j, k)] == 1)
                    {
                        M->M[mind_h(0, i, j, k, 0)] = 0.0;
                        M->M[mind_h(0, i, j, k, 1)] = 0.0;
                        M->M[mind_h(0, i, j, k, 2)] = 0.0;


                        M->Mat[ind_h(i, j, k)] = 0;
                        M->NUMCELLS[0] -= 1;
                    }
                }

            }
        }
    }

    return;
}
__host__ void BlockGeometry(MAG M, OutputFormat Out)
{
    double Magnitude = 0.0;
    for (int i = Out.xrange[0]; i <= Out.xrange[1]; i++)
    {
        for (int j = Out.yrange[0]; j <= Out.yrange[1]; j++)
        {
            for (int k = Out.zrange[0]; k <= Out.zrange[1]; k++)
            {
                Magnitude = sqrt(pow(M->M[mind_h(0, i, j, k, 0)], 2)
                    + pow(M->M[mind_h(0, i, j, k, 1)], 2)
                    + pow(M->M[mind_h(0, i, j, k, 2)], 2));

                M->M[mind_h(0, i, j, k, 0)] /= Magnitude;
                M->M[mind_h(0, i, j, k, 1)] /= Magnitude;
                M->M[mind_h(0, i, j, k, 2)] /= Magnitude;

                M->Mat[ind_h(i, j, k)] = 1;
                M->NUMCELLS[0] += 1;
            }
        }
    }
    return;
}
__host__ void MagnetiseSphere3D(MAG M, double RADIUS, int Flag)
{
    double Magnitude = 0.0;
    for (int i = 0; i < NUM_h; i++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {
            for (int k = 0; k < NUMZ_h; k++)
            {
                if ((pow(((double)i * CELL_h - ((double)NUM_h / 2.) * CELL_h), 2)
                    + pow(((double)j * CELLY_h - ((double)NUMY_h / 2.) * CELLY_h), 2)
                    + pow(((double)k * CELLZ_h - ((double)NUMZ_h / 2.) * CELLZ_h), 2))
                    <= pow(RADIUS, 2))
                {
                    if (Flag == 1)
                    {
                        M->M[mind_h(0, i, j, k, 0)] = 1.0;
                        M->M[mind_h(0, i, j, k, 1)] = 1.0;
                        M->M[mind_h(0, i, j, k, 2)] = 1.0;
                    }
                    Magnitude = sqrt(pow(M->M[mind_h(0, i, j, k, 0)], 2) + pow(M->M[mind_h(0, i, j, k, 1)], 2)
                        + pow(M->M[mind_h(0, i, j, k, 2)], 2));

                    M->M[mind_h(0, i, j, k, 0)] /= Magnitude;
                    M->M[mind_h(0, i, j, k, 1)] /= Magnitude;
                    M->M[mind_h(0, i, j, k, 2)] /= Magnitude;

                    M->M[mind_h(1, i, j, k, 0)] = M->M[mind_h(0, i, j, k, 0)];
                    M->M[mind_h(1, i, j, k, 1)] = M->M[mind_h(0, i, j, k, 1)];
                    M->M[mind_h(1, i, j, k, 2)] = M->M[mind_h(0, i, j, k, 2)];

                    M->Mat[ind_h(i, j, k)] = 1;
                    M->NUMCELLS[0]++;
                }
            }
        }
    }

    return;
}

__host__ void BlockMaterial(MAG M, OutputFormat Out,MaterialHandle handle)
{
    double Magnitude = 0.0;
    for (int i = Out.xrange[0]; i <= Out.xrange[1]; i++)
    {
        for (int j = Out.yrange[0]; j <= Out.yrange[1]; j++)
        {
            for (int k = Out.zrange[0]; k <= Out.zrange[1]; k++)
            {
                Magnitude = sqrt(pow(M->M[mind_h(0, i, j, k, 0)], 2)
                    + pow(M->M[mind_h(0, i, j, k, 1)], 2)
                    + pow(M->M[mind_h(0, i, j, k, 2)], 2));

                M->M[mind_h(0, i, j, k, 0)] /= Magnitude;
                M->M[mind_h(0, i, j, k, 1)] /= Magnitude;
                M->M[mind_h(0, i, j, k, 2)] /= Magnitude;

                if (M->Mat[ind_h(i, j, k)] == 0)
                {
                    M->Mat[ind_h(i, j, k)] = handle;
                    M->NUMCELLS[0] += 1;
                }
                else {
                    M->Mat[ind_h(i, j, k)] = handle;
                }
            }
        }
    }
    return;
}
__host__ void Cuboid(MAG M, double L, double H, double D, MaterialHandle handle)
{
    double Magnitude = 0.0;
    for (int i = 0; i < NUM_h; i++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {
            for (int k = 0; k < NUMZ_h; k++)
            {
                if ((fabs((i - floor((double)NUM_h / 2))) * CELL_h) <= (L) / 2
                    && (fabs((j - floor((double)NUMY_h / 2))) * CELLY_h) <= (H) / 2
                    && (k * CELLZ_h <= D))
                {

                        M->M[mind_h(0, i, j, k, 0)] = 1.0;
                        M->M[mind_h(0, i, j, k, 1)] = 1.0;
                        M->M[mind_h(0, i, j, k, 2)] = 1.0;
                 
                    Magnitude = sqrt(pow(M->M[mind_h(0, i, j, k, 0)], 2)
                        + pow(M->M[mind_h(0, i, j, k, 1)], 2)
                        + pow(M->M[mind_h(0, i, j, k, 2)], 2));

                    M->M[mind_h(0, i, j, k, 0)] /= Magnitude;
                    M->M[mind_h(0, i, j, k, 1)] /= Magnitude;
                    M->M[mind_h(0, i, j, k, 2)] /= Magnitude;

                    if (M->Mat[ind_h(i, j, k)] == 0)
                    {
                        M->Mat[ind_h(i, j, k)] = handle;
                        M->NUMCELLS[0] += 1;
                    }
                    else {
                        M->Mat[ind_h(i, j, k)] = handle;
                    }
                }
            }
        }
    }
    return;
}
__host__ void Disk(MAG M, double RADIUS, double HEIGHT, MaterialHandle handle)
{
    double Magnitude = 0.0;
    for (int i = 0; i < NUM_h; i++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {
            for (int k = 0; k < NUMZ_h; k++)
            {
                if ((pow(((double)i * CELL_h - ((double)NUM_h / 2.) * CELL_h), 2)
                    + pow(((double)j * CELLY_h - ((double)NUMY_h / 2.) * CELLY_h), 2))
                    <= pow(RADIUS, 2) && k * CELLZ_h <= HEIGHT)
                {
                        M->M[mind_h(0, i, j, k, 0)] = 1.0;
                        M->M[mind_h(0, i, j, k, 1)] = 1.0;
                        M->M[mind_h(0, i, j, k, 2)] = 1.0;

                    Magnitude = sqrt(pow(M->M[mind_h(0, i, j, k, 0)], 2) + pow(M->M[mind_h(0, i, j, k, 1)], 2)
                        + pow(M->M[mind_h(0, i, j, k, 2)], 2));

                    M->M[mind_h(0, i, j, k, 0)] /= Magnitude;
                    M->M[mind_h(0, i, j, k, 1)] /= Magnitude;
                    M->M[mind_h(0, i, j, k, 2)] /= Magnitude;

                    M->M[mind_h(1, i, j, k, 0)] = M->M[mind_h(0, i, j, k, 0)];
                    M->M[mind_h(1, i, j, k, 1)] = M->M[mind_h(0, i, j, k, 1)];
                    M->M[mind_h(1, i, j, k, 2)] = M->M[mind_h(0, i, j, k, 2)];

                    if (M->Mat[ind_h(i, j, k)] == 0)
                    {
                        M->Mat[ind_h(i, j, k)] = handle;
                        M->NUMCELLS[0] += 1;
                    }
                    else {
                        M->Mat[ind_h(i, j, k)] = handle;
                    }
                }
            }
        }
    }

    return;
}
__host__ void Sphere(MAG M, double RADIUS, MaterialHandle handle)
{
    double Magnitude = 0.0;
    for (int i = 0; i < NUM_h; i++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {
            for (int k = 0; k < NUMZ_h; k++)
            {
                if ((pow(((double)i * CELL_h - ((double)NUM_h / 2.) * CELL_h), 2)
                    + pow(((double)j * CELLY_h - ((double)NUMY_h / 2.) * CELLY_h), 2)
                    + pow(((double)k * CELLZ_h - ((double)NUMZ_h / 2.) * CELLZ_h), 2))
                    <= pow(RADIUS, 2))
                {

                        M->M[mind_h(0, i, j, k, 0)] = 1.0;
                        M->M[mind_h(0, i, j, k, 1)] = 1.0;
                        M->M[mind_h(0, i, j, k, 2)] = 1.0;

                    Magnitude = sqrt(pow(M->M[mind_h(0, i, j, k, 0)], 2) + pow(M->M[mind_h(0, i, j, k, 1)], 2)
                        + pow(M->M[mind_h(0, i, j, k, 2)], 2));

                    M->M[mind_h(0, i, j, k, 0)] /= Magnitude;
                    M->M[mind_h(0, i, j, k, 1)] /= Magnitude;
                    M->M[mind_h(0, i, j, k, 2)] /= Magnitude;

                    M->M[mind_h(1, i, j, k, 0)] = M->M[mind_h(0, i, j, k, 0)];
                    M->M[mind_h(1, i, j, k, 1)] = M->M[mind_h(0, i, j, k, 1)];
                    M->M[mind_h(1, i, j, k, 2)] = M->M[mind_h(0, i, j, k, 2)];

                    if (M->Mat[ind_h(i, j, k)] == 0)
                    {
                        M->Mat[ind_h(i, j, k)] = handle;
                        M->NUMCELLS[0] += 1;
                    }
                    else {
                        M->Mat[ind_h(i, j, k)] = handle;
                    }
                }
            }
        }
    }

    return;
}

__host__ void Cuboid_InRegion(MAG M,Region R,MaterialHandle handle)
{
    double Magnitude = 0.0;
    for (int i = R.x[0]; i <= (R.x[1] % NUM_h); i++)
    {
        for (int j = R.y[0]; j <= (R.y[1] % NUMY_h); j++)
        {
            for (int k = R.z[0]; k <= (R.z[1] % NUMZ_h); k++)
            {

                {
                    M->M[mind_h(0, i, j, k, 0)] = 1.0;
                    M->M[mind_h(0, i, j, k, 1)] = 1.0;
                    M->M[mind_h(0, i, j, k, 2)] = 1.0;
                    Magnitude = sqrt(pow(M->M[mind_h(0, i, j, k, 0)], 2)
                        + pow(M->M[mind_h(0, i, j, k, 1)], 2)
                        + pow(M->M[mind_h(0, i, j, k, 2)], 2));

                    M->M[mind_h(0, i, j, k, 0)] /= Magnitude;
                    M->M[mind_h(0, i, j, k, 1)] /= Magnitude;
                    M->M[mind_h(0, i, j, k, 2)] /= Magnitude;

                    if (M->Mat[ind_h(i, j, k)] == 0)
                    {
                        M->Mat[ind_h(i, j, k)] = handle;
                        M->NUMCELLS[0] += 1;
                    }
                    else {
                     M->Mat[ind_h(i, j, k)] = handle;
                    }
                }
            }
        }
    }

    return;
}

__host__ void UniformState_InMaterial(MAG M, int x, int y, int z,MaterialHandle handle)
{
    double norm = sqrt((double)x * x + (double)y * y + (double)z * z);
    for (int i = 0; i < NUM_h; i++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {
            for (int k = 0; k < NUMZ_h; k++)
            {
                if (M->Mat[ind_h(i, j, k)] == 0)
                {
                    continue;
                }

                if (M->Mat[ind_h(i, j, k)] == handle)
                {
                    M->M[mind_h(0, i, j, k, 0)] = (double)x;
                    M->M[mind_h(0, i, j, k, 1)] = (double)y;
                    M->M[mind_h(0, i, j, k, 2)] = (double)z;

                    M->M[mind_h(0, i, j, k, 0)] /= norm;
                    M->M[mind_h(0, i, j, k, 1)] /= norm;
                    M->M[mind_h(0, i, j, k, 2)] /= norm;

                    M->M[mind_h(1, i, j, k, 0)] = M->M[mind_h(0, i, j, k, 0)];
                    M->M[mind_h(1, i, j, k, 1)] = M->M[mind_h(0, i, j, k, 1)];
                    M->M[mind_h(1, i, j, k, 2)] = M->M[mind_h(0, i, j, k, 2)];

                    M->M[mind_h(2, i, j, k, 0)] = M->M[mind_h(0, i, j, k, 0)];
                    M->M[mind_h(2, i, j, k, 1)] = M->M[mind_h(0, i, j, k, 1)];
                    M->M[mind_h(2, i, j, k, 2)] = M->M[mind_h(0, i, j, k, 2)];
                }
            }
        }
    }
    return;
}
__host__ void VortexState_InMaterial(MAG M, int x, int y, int z,MaterialHandle handle)
{
    double Magnitude;
    for (int i = 0; i < NUM_h; i++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {
            for (int k = 0; k < NUMZ_h; k++)
            {
                if (M->Mat[ind_h(i, j, k)] == 0)
                {
                    continue;
                }

                if (M->Mat[ind_h(i, j, k)] == handle)
                {

                    if (z != 0 && y != 0) // Vortex in yz plane
                    {
                        M->M[mind_h(0, i, j, k, 0)] = -0.1;
                        M->M[mind_h(0, i, j, k, 1)] = -cos((M_PI / NUMY_h) * k);
                        M->M[mind_h(0, i, j, k, 2)] = cos((M_PI / NUM_h) * j);
                    }

                    if (z != 0 && x != 0) // Vortex in xz plane
                    {
                        M->M[mind_h(0, i, j, k, 1)] = -0.1;
                        M->M[mind_h(0, i, j, k, 0)] = -cos((M_PI / NUMY_h) * k);
                        M->M[mind_h(0, i, j, k, 2)] = cos((M_PI / NUM_h) * i);
                    }

                    if (x != 0 && y != 0) // Vortex in xy plane
                    {
                        M->M[mind_h(0, i, j, k, 2)] = exp(-pow((j - NUMY_h / 2), 2)) * exp(-pow((i - NUM_h / 2), 2));
                        M->M[mind_h(0, i, j, k, 0)] = cos((M_PI / NUMY_h) * j);
                        M->M[mind_h(0, i, j, k, 1)] = -cos((M_PI / NUM_h) * i);
                    }

                    if (x == 0 && y == 0 && z == 0)
                    {
                        M->M[mind_h(0, i, j, k, 2)] = -0.1;
                        M->M[mind_h(0, i, j, k, 0)] = -cos((M_PI / NUMY_h) * j);
                        M->M[mind_h(0, i, j, k, 1)] = cos((M_PI / NUM_h) * i);
                    }

                    if (x != 0 && y != 0 && z != 0)
                    {
                        M->M[mind_h(0, i, j, k, 2)] = -0.1;
                        M->M[mind_h(0, i, j, k, 0)] = -cos((M_PI / NUMY_h) * j);
                        M->M[mind_h(0, i, j, k, 1)] = cos((M_PI / NUM_h) * i);
                    }

                    Magnitude = sqrt(pow(M->M[mind_h(0, i, j, k, 0)], 2)
                        + pow(M->M[mind_h(0, i, j, k, 1)], 2)
                        + pow(M->M[mind_h(0, i, j, k, 2)], 2));

                    M->M[mind_h(0, i, j, k, 0)] /= Magnitude;
                    M->M[mind_h(0, i, j, k, 1)] /= Magnitude;
                    M->M[mind_h(0, i, j, k, 2)] /= Magnitude;

                    M->M[mind_h(1, i, j, k, 0)] = M->M[mind_h(0, i, j, k, 0)];
                    M->M[mind_h(1, i, j, k, 1)] = M->M[mind_h(0, i, j, k, 1)];
                    M->M[mind_h(1, i, j, k, 2)] = M->M[mind_h(0, i, j, k, 2)];
                    M->M[mind_h(2, i, j, k, 0)] = M->M[mind_h(0, i, j, k, 0)];
                    M->M[mind_h(2, i, j, k, 1)] = M->M[mind_h(0, i, j, k, 1)];
                    M->M[mind_h(2, i, j, k, 2)] = M->M[mind_h(0, i, j, k, 2)];
                }

            }
        }
    }
}
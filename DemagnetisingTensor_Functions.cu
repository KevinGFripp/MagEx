#include "DemagnetisingTensor_Functions.cuh"

__host__  void DemagFieldInitialise(FIELD H)
{

    for (int i = 0; i < NUM_h; i++)
    {
        for (int j = 0; j < NUMY_h; j++)
        {
            for (int k = 0; k < NUMZ_h; k++)
            {
                H->H_m[dmind_h(i, j, k, 0)] = 0.0;
                H->H_m[dmind_h(i, j, k, 1)] = 0.0;
                H->H_m[dmind_h(i, j, k, 2)] = 0.0;

                H->H_ex[find_h(i, j, k, 0)] = 0.0;
                H->H_ex[find_h(i, j, k, 1)] = 0.0;
                H->H_ex[find_h(i, j, k, 2)] = 0.0;
            }
        }
    }
    return;
}
__host__ int DemagTensorPBCWrapNx(int I)
{
    if (I < 0)
    {
        return I + NUM_h;
    }
    else
    {
        return I;
    }
}
__host__ int DemagTensorPBCWrapNy(int J)
{
    if (J < 0)
    {
        return J + NUMY_h;
    }
    else
    {
        return J;
    }
}
__host__ int DemagTensorWrapNx(int I)
{
    if (I < 0)
    {
        return (I + 2 * NUM_h);
    }
    else {
        return I;
    }

}
__host__ int DemagTensorWrapNy(int J)
{
    if (J < 0)
    {
        return (J + 2 * NUMY_h);
    }
    else {
        return J;
    }
}
__host__ int DemagTensorWrapNz(int K)
{
    if (K < 0)
    {
        return (K + 2 * NUMZ_h);
    }
    else {
        return K;
    }
}
__host__ void InitialiseDemagTensor3D(MEMDATA DATA)
{
    for (int I = 0; I < (2 - PBC_x_h) * NUM_h; I++) //Zeroing of Tensor
    {
        for (int J = 0; J < (2 - PBC_y_h) * NUMY_h; J++)
        {
            for (int K = 0; K < 2 * NUMZ_h; K++)
            {
                DATA->kxx[FFTind_h(I, J, K)][0] = 0.0;
                DATA->kxx[FFTind_h(I, J, K)][1] = 0.0;

                DATA->kxy[FFTind_h(I, J, K)][0] = 0.0;
                DATA->kxy[FFTind_h(I, J, K)][1] = 0.0;

                DATA->kxz[FFTind_h(I, J, K)][0] = 0.0;
                DATA->kxz[FFTind_h(I, J, K)][1] = 0.0;

                DATA->kyz[FFTind_h(I, J, K)][0] = 0.0;
                DATA->kyz[FFTind_h(I, J, K)][1] = 0.0;

                DATA->kyy[FFTind_h(I, J, K)][0] = 0.0;
                DATA->kyy[FFTind_h(I, J, K)][1] = 0.0;

                DATA->kzz[FFTind_h(I, J, K)][0] = 0.0;
                DATA->kzz[FFTind_h(I, J, K)][1] = 0.0;

            }
        }
    }
    return;
}
__host__ void ComputeDemagTensorNewell_GQ(MAG M, MEMDATA DATA)
{
    double Dx = CELL_h, Dy = CELLY_h, Dz = CELLZ_h;

    //find minimum cell size
    double temp = Dx;
    if (temp > Dy)
    {
        temp = Dy;
    }
    if (temp > Dz)
    {
        temp = Dz;
    }
    Dx /= temp, Dy /= temp, Dz /= temp;
    double R_CRITICAL = 12.0, R_ASYM = 65.0;

    InitialiseDemagTensor3D(DATA);
    printf("Computing Demagnetising Tensor\n");

#pragma omp parallel for schedule(dynamic)
    for (int I = 0; I < NUM_h; I++)
    {
        for (int J = 0; J < NUMY_h; J++)
        {
            for (int K = 0; K < NUMZ_h; K++)
            {

                if (I == 0 && J == 0 && K == 0)
                {
                    DATA->kxx[0][0] += SelfDemagNx(Dx, Dy, Dz); // Nxx                 
                    DATA->kyy[0][0] += SelfDemagNx(Dy, Dx, Dz); // Nyy                  
                    DATA->kzz[0][0] += SelfDemagNx(Dz, Dy, Dx); // Nzz

                    continue;
                }
                double X = (double)(I * Dx),
                       Y = (double)(J * Dy),
                       Z = (double)(K * Dz);
                double R = sqrt(X * X + Y * Y + Z * Z);

                int index = FFTind_h(I, J, K);

                if (R <= R_CRITICAL)
                {
                    DATA->kxx[index][0] += NxxInt(X, Y, Z, Dx, Dy, Dz); // Nxx
                    DATA->kxy[index][0] += NxyInt(X, Y, Z, Dx, Dy, Dz); // Nxy
                    DATA->kyy[index][0] += NxxInt(Y, X, Z, Dy, Dx, Dz); // Nyy
                    DATA->kxz[index][0] += NxyInt(X, Z, Y, Dx, Dz, Dy); // Nxz
                    DATA->kyz[index][0] += NxyInt(Y, Z, X, Dy, Dz, Dx); // Nyz
                    DATA->kzz[index][0] += NxxInt(Z, Y, X, Dz, Dy, Dx); // Nzz
                }
                if (R > R_CRITICAL && R < R_ASYM)
                {
                    DATA->kxx[index][0] += Nxx_GQ_7(X, Y, Z, Dx, Dy, Dz); // Nxx
                    DATA->kxy[index][0] += Nxy_GQ_7(X, Y, Z, Dx, Dy, Dz); // Nxy
                    DATA->kyy[index][0] += Nxx_GQ_7(Y, X, Z, Dy, Dx, Dz); // Nyy
                    DATA->kxz[index][0] += Nxy_GQ_7(X, Z, Y, Dx, Dz, Dy); // Nxz
                    DATA->kyz[index][0] += Nxy_GQ_7(Y, Z, X, Dy, Dz, Dx); // Nyz
                    DATA->kzz[index][0] += Nxx_GQ_7(Z, Y, X, Dz, Dy, Dx); // Nzz  
                }
                if (R >= R_ASYM)
                {
                    DATA->kxx[index][0] += DemagAsymptoticDiag(X, Y, Z, Dx, Dy, Dz); // Nxx
                    DATA->kxy[index][0] += DemagAsymptoticOffDiag(X, Y, Z, Dx, Dy, Dz); // Nxy
                    DATA->kyy[index][0] += DemagAsymptoticDiag(Y, X, Z, Dy, Dx, Dz); // Nyy
                    DATA->kxz[index][0] += DemagAsymptoticOffDiag(X, Z, Y, Dx, Dz, Dy); // Nxz
                    DATA->kyz[index][0] += DemagAsymptoticOffDiag(Y, Z, X, Dy, Dz, Dx); // Nyz
                    DATA->kzz[index][0] += DemagAsymptoticDiag(Z, Y, X, Dz, Dy, Dx); // Nzz  
                }

            }
        }
    }
    //Reconstruct missing terms from symmetries
#pragma omp parallel for schedule(static)
    for (int i = 0; i < 2 * NUM_h; i++)
    {
        for (int j = 0; j < 2 * NUMY_h; j++)
        {
            for (int k = 0; k < 2 * NUMZ_h; k++)
            {
                int dx = i, dy = j, dz = k;

                if (i > (NUM_h)) { dx = 2 * NUM_h - i; }

                if (j > (NUMY_h)) { dy = 2 * NUMY_h - j; }

                if (k > (NUMZ_h)) { dz = 2 * NUMZ_h - k; }

                int N_index = FFTind_h(dx, dy, dz);
                int index = FFTind_h(i, j, k);

                DATA->kxx[index][0] = DATA->kxx[N_index][0];
                DATA->kyy[index][0] = DATA->kyy[N_index][0];
                DATA->kzz[index][0] = DATA->kzz[N_index][0];

                int G;
                G = SignD((NUM_h - i)) * SignD(NUMY_h - j);
                DATA->kxy[index][0] = G * DATA->kxy[N_index][0];

                G = SignD((NUM_h - i)) * SignD(NUMZ_h - k);
                DATA->kxz[index][0] = G * DATA->kxz[N_index][0];

                G = SignD((NUMY_h - j)) * SignD(NUMZ_h - k);
                DATA->kyz[index][0] = G * DATA->kyz[N_index][0];
            }
        }
    }

    return;
}
__host__ void ComputeDemagTensorNewell3D_PBC(MAG M, MEMDATA DATA)
{

    //Normalised cell distances
    double Dx = CELL_h, Dy = CELLY_h, Dz = CELLZ_h;

    //find maximum cell size
    double temp = Dx;
    if (temp < Dy)
    {
        temp = Dy;
    }
    if (temp < Dz)
    {
        temp = Dz;
    }
    Dx /= temp, Dy /= temp, Dz /= temp;

    double R_CRITICAL = 15.0, R_ASYM = 50.0;

    int Ilower, Iupper, Jlower, Jupper;

    if (PBC_x_h == 1)
    {
        Ilower = -NUM_h / 2;
        Iupper = NUM_h / 2;
    }
    else {
        Ilower = 0;
        Iupper = NUM_h;
    }

    if (PBC_y_h == 1)
    {
        Jlower = -NUMY_h / 2;
        Jupper = NUMY_h / 2;
    }
    else {
        Jlower = 0;
        Jupper = NUMY_h;
    }

    InitialiseDemagTensor3D(DATA);
    if (IsPBCEnabled == true)
    {
        printf("Computing Demagnetising Tensor, With PBCs(%d,%d)\n", PBC_x_images_h * PBC_x_h,
            PBC_y_images_h * PBC_y_h);
    }
    else {
        printf("Computing demagnetising tensor...\n");
    }

    int PROGRESSCOUNT = 0;

    for (int I = Ilower; I < Iupper; I++)
    {
#pragma omp parallel for schedule(dynamic) 

        for (int J = Jlower; J < Jupper; J++)
        {
            for (int K = 0; K < NUMZ_h; K++)
            {
                for (int h_x = -PBC_x_h * (PBC_x_images_h); h_x <= PBC_x_h * PBC_x_images_h; h_x++)
                {
                    for (int h_y = -PBC_y_h * (PBC_y_images_h); h_y <= PBC_y_h * PBC_y_images_h; h_y++)
                    {

                        int l = 0, m = 0, n = 0;
                        double X = (double)((I + (h_x * (double)NUM_h)) * Dx);
                        double Y = (double)((J + (h_y * (double)NUMY_h)) * Dy);
                        double Z = (double)(K * Dz);
                        double R = sqrt(X * X + Y * Y + Z * Z);

                        PBC_x_h == 1? l = DemagTensorPBCWrapNx(I): l = DemagTensorWrapNx(I); 

                        PBC_y_h == 1? m = DemagTensorPBCWrapNy(J) : m = DemagTensorWrapNy(J); 

                        n = DemagTensorWrapNz(K);

                        int index = FFTind_h(l, m, n);


                        if (I == 0 && J == 0 && K == 0)
                        {
                            if (h_x == 0 && h_y == 0)
                            {
                                DATA->kxx[0][0] += SelfDemagNx(Dx, Dy, Dz); // Nxx
                                DATA->kyy[0][0] += SelfDemagNx(Dy, Dx, Dz); // Nyy
                                DATA->kzz[0][0] += SelfDemagNx(Dz, Dy, Dx); // Nzz
                                continue;
                            }
                        }

                        if (R <= R_CRITICAL)
                        {
                            DATA->kxx[index][0] += NxxInt(X, Y, Z, Dx, Dy, Dz); // Nxx
                            DATA->kxy[index][0] += NxyInt(X, Y, Z, Dx, Dy, Dz); // Nxy
                            DATA->kyy[index][0] += NxxInt(Y, X, Z, Dy, Dx, Dz); // Nyy
                            DATA->kxz[index][0] += NxyInt(X, Z, Y, Dx, Dz, Dy); // Nxz
                            DATA->kyz[index][0] += NxyInt(Y, Z, X, Dy, Dz, Dx); // Nyz
                            DATA->kzz[index][0] += NxxInt(Z, Y, X, Dz, Dy, Dx); // Nzz                                                     
                        }

                        if (R > R_CRITICAL && R < R_ASYM)
                        {
                            DATA->kxx[index][0] += Nxx_GQ_5(X, Y, Z, Dx, Dy, Dz); // Nxx
                            DATA->kxy[index][0] += Nxy_GQ_5(X, Y, Z, Dx, Dy, Dz); // Nxy
                            DATA->kyy[index][0] += Nxx_GQ_5(Y, X, Z, Dy, Dx, Dz); // Nyy
                            DATA->kxz[index][0] += Nxy_GQ_5(X, Z, Y, Dx, Dz, Dy); // Nxz
                            DATA->kyz[index][0] += Nxy_GQ_5(Y, Z, X, Dy, Dz, Dx); // Nyz
                            DATA->kzz[index][0] += Nxx_GQ_5(Z, Y, X, Dz, Dy, Dx); // Nzz           
                        }

                        if (R >= R_ASYM)
                        {
                            DATA->kxx[index][0] += DemagAsymptoticDiag(X, Y, Z, Dx, Dy, Dz); // Nxx
                            DATA->kxy[index][0] += DemagAsymptoticOffDiag(X, Y, Z, Dx, Dy, Dz); // Nxy
                            DATA->kyy[index][0] += DemagAsymptoticDiag(Y, X, Z, Dy, Dx, Dz); // Nyy
                            DATA->kxz[index][0] += DemagAsymptoticOffDiag(X, Z, Y, Dx, Dz, Dy); // Nxz
                            DATA->kyz[index][0] += DemagAsymptoticOffDiag(Y, Z, X, Dy, Dz, Dx); // Nyz
                            DATA->kzz[index][0] += DemagAsymptoticDiag(Z, Y, X, Dz, Dy, Dx); // Nzz  
                        }
                    }
                }
            }
        }
        PROGRESSCOUNT++;
        if (I % 8 == 0)
        {
            printf(" %f%% Complete\r", (double)((PROGRESSCOUNT * 100. / (double)(NUM_h))));
        }
    }

    //Reconstruct missing terms from symmetries
#pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < (2 - PBC_x_h) * NUM_h; i++)
    {
        for (int j = 0; j < (2 - PBC_y_h) * NUMY_h; j++)
        {
            for (int k = 0; k < 2 * NUMZ_h; k++)
            {
                int dx = i, dy = j, dz = k;

                if (i > (NUM_h)) { dx = 2 * NUM_h - i; }

                if (j > (NUMY_h)) { dy = 2* NUMY_h - j; }          

                if (k > (NUMZ_h)) { dz = 2 * NUMZ_h - k; }

                int N_index = FFTind_h(dx, dy, dz);
                int index = FFTind_h(i, j, k);

                DATA->kxx[index][0] = DATA->kxx[N_index][0];
                DATA->kyy[index][0] = DATA->kyy[N_index][0];
                DATA->kzz[index][0] = DATA->kzz[N_index][0];

                int G;
                G = SignD((NUM_h - i)) * SignD(NUMY_h - j);
                DATA->kxy[index][0] = G * DATA->kxy[N_index][0];

                G = SignD((NUM_h - i)) * SignD(NUMZ_h - k);
                DATA->kxz[index][0] = G * DATA->kxz[N_index][0];

                G = SignD((NUMY_h - j)) * SignD(NUMZ_h - k);
                DATA->kyz[index][0] = G * DATA->kyz[N_index][0];
            }
        }
    }
    return;
}


__host__ double NewellF_GQ(double X, double Y, double Z, double Dx, double Dy, double Dz,
    double y, double z, double yprime, double zprime)
{
    double Prefactor = (Dy * Dz / (4 * M_PI * Dx));
    double Denominator = sqrt(X * X + pow((Dy * y + Y - Dy * yprime), 2) + pow((Dz * z + Z - Dz * zprime), 2));
    return (Prefactor / Denominator);
}
__host__ double NewellG_GQ(double X, double Y, double Z, double Dx, double Dy, double Dz,
    double y, double z, double zprime, double xprime)
{
    double Prefactor = Dz / (4 * M_PI);
    double Denominator = sqrt(pow((X + Dx * xprime), 2) + pow((Y + Dy * (y - 1)), 2) + pow((Z + Dz * (z + zprime - 1)), 2));
    return (Prefactor / Denominator);
}
__host__ double NxxInt(double X, double Y, double Z, double Dx, double Dy, double Dz)
{
    double C = 1 / (4 * M_PI * Dx * Dy * Dz);

    double arr[27];
    arr[0] = -1 * NewellF(X + Dx, Y + Dy, Z + Dz);
    arr[1] = -1 * NewellF(X + Dx, Y - Dy, Z + Dz);
    arr[2] = -1 * NewellF(X + Dx, Y - Dy, Z - Dz);
    arr[3] = -1 * NewellF(X + Dx, Y + Dy, Z - Dz);
    arr[4] = -1 * NewellF(X - Dx, Y + Dy, Z - Dz);
    arr[5] = -1 * NewellF(X - Dx, Y + Dy, Z + Dz);
    arr[6] = -1 * NewellF(X - Dx, Y - Dy, Z + Dz);
    arr[7] = -1 * NewellF(X - Dx, Y - Dy, Z - Dz);

    arr[8] = 2 * NewellF(X, Y - Dy, Z - Dz);
    arr[9] = 2 * NewellF(X, Y - Dy, Z + Dz);
    arr[10] = 2 * NewellF(X, Y + Dy, Z + Dz);
    arr[11] = 2 * NewellF(X, Y + Dy, Z - Dz);
    arr[12] = 2 * NewellF(X + Dx, Y + Dy, Z);
    arr[13] = 2 * NewellF(X + Dx, Y, Z + Dz);
    arr[14] = 2 * NewellF(X + Dx, Y, Z - Dz);
    arr[15] = 2 * NewellF(X + Dx, Y - Dy, Z);
    arr[16] = 2 * NewellF(X - Dx, Y - Dy, Z);
    arr[17] = 2 * NewellF(X - Dx, Y, Z + Dz);
    arr[18] = 2 * NewellF(X - Dx, Y, Z - Dz);
    arr[19] = 2 * NewellF(X - Dx, Y + Dy, Z);

    arr[20] = -4 * NewellF(X, Y - Dy, Z);
    arr[21] = -4 * NewellF(X, Y + Dy, Z);
    arr[22] = -4 * NewellF(X, Y, Z - Dz);
    arr[23] = -4 * NewellF(X, Y, Z + Dz);
    arr[24] = -4 * NewellF(X + Dx, Y, Z);
    arr[25] = -4 * NewellF(X - Dx, Y, Z);

    arr[26] = 8 * NewellF(X, Y, Z);

    double result = C * AccurateSum(27, arr);

    return result;

}
__host__ double NxyInt(double X, double Y, double Z, double Dx, double Dy, double Dz)
{

    double C = 1 / (4 * M_PI * Dx * Dy * Dz);
    double arr[27];

    arr[0] = -1 * NewellG(X - Dx, Y - Dy, Z - Dz);
    arr[1] = -1 * NewellG(X - Dx, Y - Dy, Z + Dz);
    arr[2] = -1 * NewellG(X + Dx, Y - Dy, Z + Dz);
    arr[3] = -1 * NewellG(X + Dx, Y - Dy, Z - Dz);
    arr[4] = -1 * NewellG(X + Dx, Y + Dy, Z - Dz);
    arr[5] = -1 * NewellG(X + Dx, Y + Dy, Z + Dz);
    arr[6] = -1 * NewellG(X - Dx, Y + Dy, Z + Dz);
    arr[7] = -1 * NewellG(X - Dx, Y + Dy, Z - Dz);

    arr[8] = 2 * NewellG(X, Y + Dy, Z - Dz);
    arr[9] = 2 * NewellG(X, Y + Dy, Z + Dz);
    arr[10] = 2 * NewellG(X, Y - Dy, Z + Dz);
    arr[11] = 2 * NewellG(X, Y - Dy, Z - Dz);
    arr[12] = 2 * NewellG(X - Dx, Y - Dy, Z);
    arr[13] = 2 * NewellG(X - Dx, Y + Dy, Z);
    arr[14] = 2 * NewellG(X - Dx, Y, Z - Dz);
    arr[15] = 2 * NewellG(X - Dx, Y, Z + Dz);
    arr[16] = 2 * NewellG(X + Dx, Y, Z + Dz);
    arr[17] = 2 * NewellG(X + Dx, Y, Z - Dz);
    arr[18] = 2 * NewellG(X + Dx, Y - Dy, Z);
    arr[19] = 2 * NewellG(X + Dx, Y + Dy, Z);

    arr[20] = -4 * NewellG(X - Dx, Y, Z);
    arr[21] = -4 * NewellG(X + Dx, Y, Z);
    arr[22] = -4 * NewellG(X, Y, Z + Dz);
    arr[23] = -4 * NewellG(X, Y, Z - Dz);
    arr[24] = -4 * NewellG(X, Y - Dy, Z);
    arr[25] = -4 * NewellG(X, Y + Dy, Z);

    arr[26] = 8 * NewellG(X, Y, Z);

    return C * AccurateSum(27, arr);
}
__host__ double NewellF(double x, double y, double z)
{
    double R = x * x + y * y + z * z;
    return (+y / 2.0 * (z * z - x * x) * asinh(y / (sqrt(x * x + z * z) + eps)) + z / 2.0 * (y * y - x * x) * asinh(z / (sqrt(x * x + y * y) + eps)) - x * y * z * atan(y * z / (x * sqrt(R) + eps)) + 1.0 / 6.0 * (2 * x * x - y * y - z * z) * sqrt(R + eps));
}
__host__ double NewellG(double x, double y, double z)
{
    double R = x * x + y * y + z * z;
    return (+x * y * z * asinh(z / (sqrt(x * x + y * y) + eps)) + y / 6.0 * (3.0 * z * z - y * y) * asinh(x / (sqrt(y * y + z * z) + eps)) + x / 6.0 * (3.0 * z * z - x * x) * asinh(y / (sqrt(x * x + z * z) + eps)) - z * z * z / 6.0 * atan(x * y / (z * sqrt(R) + eps)) - z * y * y / 2.0 * atan(x * z / (y * sqrt(R) + eps)) - z * x * x / 2.0 * atan(y * z / (x * sqrt(R) + eps)) - x * y * sqrt(R + eps) / 3.0);
}
__host__ double SelfDemagNx(double x, double y, double z)
{
    if (x <= 0.0 || y <= 0.0 || z <= 0.0) return 0.0;
    if (x == y && y == z) {  // Special case: cube
        return 1.0 / 3.0;
    }

    double xsq = x * x, ysq = y * y, zsq = z * z;
    double diag = sqrt(xsq + ysq + zsq);
    double arr[15];

    double mpxy = (x - y) * (x + y);
    double mpxz = (x - z) * (x + z);

    arr[0] = -4 * (2 * xsq * x - ysq * y - zsq * z);
    arr[1] = 4 * (xsq + mpxy) * sqrt(xsq + ysq);
    arr[2] = 4 * (xsq + mpxz) * sqrt(xsq + zsq);
    arr[3] = -4 * (ysq + zsq) * sqrt(ysq + zsq);
    arr[4] = -4 * diag * (mpxy + mpxz);

    arr[5] = 24 * x * y * z * atan(y * z / (x * diag));
    arr[6] = 12 * (z + y) * xsq * log(x);

    arr[7] = 12 * z * ysq * log((sqrt(ysq + zsq) + z) / y);
    arr[8] = -12 * z * xsq * log(sqrt(xsq + zsq) + z);
    arr[9] = 12 * z * mpxy * log(diag + z);
    arr[10] = -6 * z * mpxy * log(xsq + ysq);

    arr[11] = 12 * y * zsq * log((sqrt(ysq + zsq) + y) / z);
    arr[12] = -12 * y * xsq * log(sqrt(xsq + ysq) + y);
    arr[13] = 12 * y * mpxz * log(diag + y);
    arr[14] = -6 * y * mpxz * log(xsq + zsq);

    double Nxx = AccurateSum(15, arr) / (12 * M_PI * x * y * z);
    return Nxx;
}
__host__ double AccurateSum(int n, double* arr)
{

    double sum, corr, y, u, t, v, z, x, tmp;

    sum = arr[0]; corr = 0;
    for (int i = 1; i < n; i++) {
        x = arr[i];
        y = corr + x;
        tmp = y - corr;
        u = x - tmp;
        t = y + sum;
        tmp = t - sum;
        v = y - tmp;
        z = u + v;
        sum = t + z;
        tmp = sum - t;
        corr = z - tmp;
    }
    return sum;
}
__host__ double DemagAsymptoticDiag(double x, double y, double z, double hx, double hy, double hz)
{
    double hx2 = hx * hx;
    double hy2 = hy * hy;
    double hz2 = hz * hz;

    double hx4 = hx2 * hx2;
    double hy4 = hy2 * hy2;
    double hz4 = hz2 * hz2;

    double hx6 = hx4 * hx2;
    double hy6 = hy4 * hy2;
    double hz6 = hz4 * hz2;

    double a1, a2, a3, a4, a5, a6;
    double b1, b2, b3, b4, b5, b6, b7, b8, b9, b10;
    double c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15;

    double lead_weight = (-hx * hy * hz / (4. * M_PI));

    bool cubic_cell;

    //Initialize coefficients for 1/R^5 term
    if (hx2 != hy2 || hx2 != hz2 || hy2 != hz2) {
        cubic_cell = false;

        a1 = a2 = a3 = a4 = a5 = a6 = lead_weight / 4.0;
        a1 *= 8 * hx2 - 4 * hy2 - 4 * hz2;
        a2 *= -24 * hx2 + 27 * hy2 - 3 * hz2;
        a3 *= -24 * hx2 - 3 * hy2 + 27 * hz2;
        a4 *= 3 * hx2 - 4 * hy2 + 1 * hz2;
        a5 *= 6 * hx2 - 3 * hy2 - 3 * hz2;
        a6 *= 3 * hx2 + 1 * hy2 - 4 * hz2;
    }
    else {

        //cubic cell
        cubic_cell = true;
        a1 = a2 = a3 = a4 = a5 = a6 = 0.0;
    }

    //Initialize coefficients for 1/R^7 term
    b1 = b2 = b3 = b4 = b5 = b6 = b7 = b8 = b9 = b10 = lead_weight / 16.0;

    if (cubic_cell) {

        b1 *= -14 * hx4;
        b2 *= 105 * hx4;
        b3 *= 105 * hx4;
        b4 *= -105 * hx4;
        b6 *= -105 * hx4;
        b7 *= 7 * hx4;
        b10 *= 7 * hx4;
        b5 = b8 = b9 = 0;
    }
    else {

        b1 *= 32 * hx4 - 40 * hx2 * hy2 - 40 * hx2 * hz2 + 12 * hy4 + 10 * hy2 * hz2 + 12 * hz4;
        b2 *= -240 * hx4 + 580 * hx2 * hy2 + 20 * hx2 * hz2 - 202 * hy4 - 75 * hy2 * hz2 + 22 * hz4;
        b3 *= -240 * hx4 + 20 * hx2 * hy2 + 580 * hx2 * hz2 + 22 * hy4 - 75 * hy2 * hz2 - 202 * hz4;
        b4 *= 180 * hx4 - 505 * hx2 * hy2 + 55 * hx2 * hz2 + 232 * hy4 - 75 * hy2 * hz2 + 8 * hz4;
        b5 *= 360 * hx4 - 450 * hx2 * hy2 - 450 * hx2 * hz2 - 180 * hy4 + 900 * hy2 * hz2 - 180 * hz4;
        b6 *= 180 * hx4 + 55 * hx2 * hy2 - 505 * hx2 * hz2 + 8 * hy4 - 75 * hy2 * hz2 + 232 * hz4;
        b7 *= -10 * hx4 + 30 * hx2 * hy2 - 5 * hx2 * hz2 - 16 * hy4 + 10 * hy2 * hz2 - 2 * hz4;
        b8 *= -30 * hx4 + 55 * hx2 * hy2 + 20 * hx2 * hz2 + 8 * hy4 - 75 * hy2 * hz2 + 22 * hz4;
        b9 *= -30 * hx4 + 20 * hx2 * hy2 + 55 * hx2 * hz2 + 22 * hy4 - 75 * hy2 * hz2 + 8 * hz4;
        b10 *= -10 * hx4 - 5 * hx2 * hy2 + 30 * hx2 * hz2 - 2 * hy4 + 10 * hy2 * hz2 - 16 * hz4;
    }

    //Initialize coefficients for 1/R^9 term
    c1 = c2 = c3 = c4 = c5 = c6 = c7 = c8 = c9 = c10 = c11 = c12 = c13 = c14 = c15 = lead_weight / 192.0;

    if (cubic_cell) {

        c1 *= 32 * hx6;
        c2 *= -448 * hx6;
        c3 *= -448 * hx6;
        c4 *= -150 * hx6;
        c5 *= 7620 * hx6;
        c6 *= -150 * hx6;
        c7 *= 314 * hx6;
        c8 *= -3810 * hx6;
        c9 *= -3810 * hx6;
        c10 *= 314 * hx6;
        c11 *= -16 * hx6;
        c12 *= 134 * hx6;
        c13 *= 300 * hx6;
        c14 *= 134 * hx6;
        c15 *= -16 * hx6;
    }
    else {

        c1 *= 384 * hx6 + -896 * hx4 * hy2 + -896 * hx4 * hz2 + 672 * hx2 * hy4 + 560 * hx2 * hy2 * hz2 + 672 * hx2 * hz4 + -120 * hy6 + -112 * hy4 * hz2 + -112 * hy2 * hz4 + -120 * hz6;
        c2 *= -5376 * hx6 + 22624 * hx4 * hy2 + 2464 * hx4 * hz2 + -19488 * hx2 * hy4 + -7840 * hx2 * hy2 * hz2 + 672 * hx2 * hz4 + 3705 * hy6 + 2198 * hy4 * hz2 + 938 * hy2 * hz4 + -345 * hz6;
        c3 *= -5376 * hx6 + 2464 * hx4 * hy2 + 22624 * hx4 * hz2 + 672 * hx2 * hy4 + -7840 * hx2 * hy2 * hz2 + -19488 * hx2 * hz4 + -345 * hy6 + 938 * hy4 * hz2 + 2198 * hy2 * hz4 + 3705 * hz6;
        c4 *= 10080 * hx6 + -48720 * hx4 * hy2 + 1680 * hx4 * hz2 + 49770 * hx2 * hy4 + -2625 * hx2 * hy2 * hz2 + -630 * hx2 * hz4 + -10440 * hy6 + -1050 * hy4 * hz2 + 2100 * hy2 * hz4 + -315 * hz6;
        c5 *= 20160 * hx6 + -47040 * hx4 * hy2 + -47040 * hx4 * hz2 + -6300 * hx2 * hy4 + 133350 * hx2 * hy2 * hz2 + -6300 * hx2 * hz4 + 7065 * hy6 + -26670 * hy4 * hz2 + -26670 * hy2 * hz4 + 7065 * hz6;
        c6 *= 10080 * hx6 + 1680 * hx4 * hy2 + -48720 * hx4 * hz2 + -630 * hx2 * hy4 + -2625 * hx2 * hy2 * hz2 + 49770 * hx2 * hz4 + -315 * hy6 + 2100 * hy4 * hz2 + -1050 * hy2 * hz4 + -10440 * hz6;
        c7 *= -3360 * hx6 + 17290 * hx4 * hy2 + -1610 * hx4 * hz2 + -19488 * hx2 * hy4 + 5495 * hx2 * hy2 * hz2 + -588 * hx2 * hz4 + 4848 * hy6 + -3136 * hy4 * hz2 + 938 * hy2 * hz4 + -75 * hz6;
        c8 *= -10080 * hx6 + 32970 * hx4 * hy2 + 14070 * hx4 * hz2 + -6300 * hx2 * hy4 + -66675 * hx2 * hy2 * hz2 + 12600 * hx2 * hz4 + -10080 * hy6 + 53340 * hy4 * hz2 + -26670 * hy2 * hz4 + 3015 * hz6;
        c9 *= -10080 * hx6 + 14070 * hx4 * hy2 + 32970 * hx4 * hz2 + 12600 * hx2 * hy4 + -66675 * hx2 * hy2 * hz2 + -6300 * hx2 * hz4 + 3015 * hy6 + -26670 * hy4 * hz2 + 53340 * hy2 * hz4 + -10080 * hz6;
        c10 *= -3360 * hx6 + -1610 * hx4 * hy2 + 17290 * hx4 * hz2 + -588 * hx2 * hy4 + 5495 * hx2 * hy2 * hz2 + -19488 * hx2 * hz4 + -75 * hy6 + 938 * hy4 * hz2 + -3136 * hy2 * hz4 + 4848 * hz6;
        c11 *= 105 * hx6 + -560 * hx4 * hy2 + 70 * hx4 * hz2 + 672 * hx2 * hy4 + -280 * hx2 * hy2 * hz2 + 42 * hx2 * hz4 + -192 * hy6 + 224 * hy4 * hz2 + -112 * hy2 * hz4 + 15 * hz6;
        c12 *= 420 * hx6 + -1610 * hx4 * hy2 + -350 * hx4 * hz2 + 672 * hx2 * hy4 + 2345 * hx2 * hy2 * hz2 + -588 * hx2 * hz4 + 528 * hy6 + -3136 * hy4 * hz2 + 2198 * hy2 * hz4 + -345 * hz6;
        c13 *= 630 * hx6 + -1470 * hx4 * hy2 + -1470 * hx4 * hz2 + -630 * hx2 * hy4 + 5250 * hx2 * hy2 * hz2 + -630 * hx2 * hz4 + 360 * hy6 + -1050 * hy4 * hz2 + -1050 * hy2 * hz4 + 360 * hz6;
        c14 *= 420 * hx6 + -350 * hx4 * hy2 + -1610 * hx4 * hz2 + -588 * hx2 * hy4 + 2345 * hx2 * hy2 * hz2 + 672 * hx2 * hz4 + -345 * hy6 + 2198 * hy4 * hz2 + -3136 * hy2 * hz4 + 528 * hz6;
        c15 *= 105 * hx6 + 70 * hx4 * hy2 + -560 * hx4 * hz2 + 42 * hx2 * hy4 + -280 * hx2 * hy2 * hz2 + 672 * hx2 * hz4 + 15 * hy6 + -112 * hy4 * hz2 + 224 * hy2 * hz4 + -192 * hz6;
    }

    double tx2 = 0, ty2 = 0, tz2 = 0;
    double R = 0, iR = 0;
    double R2 = 0, iR2 = 0;

    R2 = x * x + y * y + z * z;
    R = sqrt(R2);

    if (R) {

        tx2 = x * x / (R2 * R2);
        ty2 = y * y / (R2 * R2);
        tz2 = z * z / (R2 * R2);

        iR = 1 / R;
        iR2 = 1 / R2;
    }

    if (iR2 <= 0.0) {

        //Asymptotic expansion doesn't apply for R==0. Don't use!
        return 0.0;
    }

    double tz4 = tz2 * tz2;
    double tz6 = tz4 * tz2;
    double term3 = (2 * tx2 - ty2 - tz2) * lead_weight;

    double term5 = 0.0;
    double term7 = 0.0;

    if (cubic_cell) {

        double ty4 = ty2 * ty2;

        term7 = ((b1 * tx2
            + (b2 * ty2 + b3 * tz2)) * tx2
            + (b4 * ty4 + b6 * tz4)) * tx2
            + b7 * ty4 * ty2 + b10 * tz6;
    }
    else {

        term5 = (a1 * tx2 + (a2 * ty2 + a3 * tz2)) * tx2
            + (a4 * ty2 + a5 * tz2) * ty2 + a6 * tz4;

        term7 = ((b1 * tx2
            + (b2 * ty2 + b3 * tz2)) * tx2
            + ((b4 * ty2 + b5 * tz2) * ty2 + b6 * tz4)) * tx2
            + ((b7 * ty2 + b8 * tz2) * ty2 + b9 * tz4) * ty2
            + b10 * tz6;
    }

    double term9 = (((c1 * tx2
        + (c2 * ty2 + c3 * tz2)) * tx2
        + ((c4 * ty2 + c5 * tz2) * ty2 + c6 * tz4)) * tx2
        + (((c7 * ty2 + c8 * tz2) * ty2 + c9 * tz4) * ty2 + c10 * tz6)) * tx2
        + (((c11 * ty2 + c12 * tz2) * ty2 + c13 * tz4) * ty2 + c14 * tz6) * ty2
        + c15 * tz4 * tz4;

    //Error should be of order 1/R^11
    return (term9 + term7 + term5 + term3) * iR;
}
__host__ double DemagAsymptoticOffDiag(double x, double y, double z, double hx, double hy, double hz)
{
    double hx2 = hx * hx;
    double hy2 = hy * hy;
    double hz2 = hz * hz;

    double hx4 = hx2 * hx2;
    double hy4 = hy2 * hy2;
    double hz4 = hz2 * hz2;

    double hx6 = hx4 * hx2;
    double hy6 = hy4 * hy2;
    double hz6 = hz4 * hz2;

    double a1, a2, a3, a4, a5, a6;
    double b1, b2, b3, b4, b5, b6, b7, b8, b9, b10;
    double c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15;

    double lead_weight = (-hx * hy * hz / (4. * M_PI));

    bool cubic_cell;

    // Initialize coefficients for 1/R^5 term
    if (hx2 != hy2 || hx2 != hz2 || hy2 != hz2) {

        //Non-cubic cell
        cubic_cell = false;

        a1 = a2 = a3 = (lead_weight * 5.0) / 4.0;
        a1 *= 4 * hx2 - 3 * hy2 - 1 * hz2;
        a2 *= -3 * hx2 + 4 * hy2 - 1 * hz2;
        a3 *= -3 * hx2 - 3 * hy2 + 6 * hz2;
    }
    else {

        //Cubic cell
        cubic_cell = true;

        a1 = a2 = a3 = 0.0;
    }

    // Initialize coefficients for 1/R^7 term
    b1 = b2 = b3 = b4 = b5 = b6 = (lead_weight * 7.0) / 16.0;

    if (cubic_cell) {

        b1 *= -7 * hx4;
        b2 *= 19 * hx4;
        b3 *= 13 * hx4;
        b4 *= -7 * hx4;
        b5 *= 13 * hx4;
        b6 *= -13 * hx4;
    }
    else {

        b1 *= 16 * hx4 - 30 * hx2 * hy2 - 10 * hx2 * hz2 + 10 * hy4 + 5 * hy2 * hz2 + 2 * hz4;
        b2 *= -40 * hx4 + 105 * hx2 * hy2 - 5 * hx2 * hz2 - 40 * hy4 - 5 * hy2 * hz2 + 4 * hz4;
        b3 *= -40 * hx4 - 15 * hx2 * hy2 + 115 * hx2 * hz2 + 20 * hy4 - 35 * hy2 * hz2 - 32 * hz4;
        b4 *= 10 * hx4 - 30 * hx2 * hy2 + 5 * hx2 * hz2 + 16 * hy4 - 10 * hy2 * hz2 + 2 * hz4;
        b5 *= 20 * hx4 - 15 * hx2 * hy2 - 35 * hx2 * hz2 - 40 * hy4 + 115 * hy2 * hz2 - 32 * hz4;
        b6 *= 10 * hx4 + 15 * hx2 * hy2 - 40 * hx2 * hz2 + 10 * hy4 - 40 * hy2 * hz2 + 32 * hz4;
    }

    // Initialize coefficients for 1/R^9 term
    c1 = c2 = c3 = c4 = c5 = c6 = c7 = c8 = c9 = c10 = lead_weight / 64.0;

    if (cubic_cell) {

        c1 *= 48 * hx6;
        c2 *= -142 * hx6;
        c3 *= -582 * hx6;
        c4 *= -142 * hx6;
        c5 *= 2840 * hx6;
        c6 *= -450 * hx6;
        c7 *= 48 * hx6;
        c8 *= -582 * hx6;
        c9 *= -450 * hx6;
        c10 *= 180 * hx6;
    }
    else {

        c1 *= 576 * hx6 + -2016 * hx4 * hy2 + -672 * hx4 * hz2 + 1680 * hx2 * hy4 + 840 * hx2 * hy2 * hz2 + 336 * hx2 * hz4 + -315 * hy6 + -210 * hy4 * hz2 + -126 * hy2 * hz4 + -45 * hz6;
        c2 *= -3024 * hx6 + 13664 * hx4 * hy2 + 448 * hx4 * hz2 + -12670 * hx2 * hy4 + -2485 * hx2 * hy2 * hz2 + 546 * hx2 * hz4 + 2520 * hy6 + 910 * hy4 * hz2 + 84 * hy2 * hz4 + -135 * hz6;
        c3 *= -3024 * hx6 + 1344 * hx4 * hy2 + 12768 * hx4 * hz2 + 2730 * hx2 * hy4 + -10185 * hx2 * hy2 * hz2 + -8694 * hx2 * hz4 + -945 * hy6 + 1680 * hy4 * hz2 + 2394 * hy2 * hz4 + 1350 * hz6;
        c4 *= 2520 * hx6 + -12670 * hx4 * hy2 + 910 * hx4 * hz2 + 13664 * hx2 * hy4 + -2485 * hx2 * hy2 * hz2 + 84 * hx2 * hz4 + -3024 * hy6 + 448 * hy4 * hz2 + 546 * hy2 * hz4 + -135 * hz6;
        c5 *= 5040 * hx6 + -9940 * hx4 * hy2 + -13580 * hx4 * hz2 + -9940 * hx2 * hy4 + 49700 * hx2 * hy2 * hz2 + -6300 * hx2 * hz4 + 5040 * hy6 + -13580 * hy4 * hz2 + -6300 * hy2 * hz4 + 2700 * hz6;
        c6 *= 2520 * hx6 + 2730 * hx4 * hy2 + -14490 * hx4 * hz2 + 420 * hx2 * hy4 + -7875 * hx2 * hy2 * hz2 + 17640 * hx2 * hz4 + -945 * hy6 + 3990 * hy4 * hz2 + -840 * hy2 * hz4 + -3600 * hz6;
        c7 *= -315 * hx6 + 1680 * hx4 * hy2 + -210 * hx4 * hz2 + -2016 * hx2 * hy4 + 840 * hx2 * hy2 * hz2 + -126 * hx2 * hz4 + 576 * hy6 + -672 * hy4 * hz2 + 336 * hy2 * hz4 + -45 * hz6;
        c8 *= -945 * hx6 + 2730 * hx4 * hy2 + 1680 * hx4 * hz2 + 1344 * hx2 * hy4 + -10185 * hx2 * hy2 * hz2 + 2394 * hx2 * hz4 + -3024 * hy6 + 12768 * hy4 * hz2 + -8694 * hy2 * hz4 + 1350 * hz6;
        c9 *= -945 * hx6 + 420 * hx4 * hy2 + 3990 * hx4 * hz2 + 2730 * hx2 * hy4 + -7875 * hx2 * hy2 * hz2 + -840 * hx2 * hz4 + 2520 * hy6 + -14490 * hy4 * hz2 + 17640 * hy2 * hz4 + -3600 * hz6;
        c10 *= -315 * hx6 + -630 * hx4 * hy2 + 2100 * hx4 * hz2 + -630 * hx2 * hy4 + 3150 * hx2 * hy2 * hz2 + -3360 * hx2 * hz4 + -315 * hy6 + 2100 * hy4 * hz2 + -3360 * hy2 * hz4 + 1440 * hz6;
    }

    double tx2 = 0, ty2 = 0, tz2 = 0;
    double R = 0, iR = 0;
    double R2 = 0, iR2 = 0;

    R2 = x * x + y * y + z * z;
    R = sqrt(R2);

    if (R) {

        tx2 = x * x / (R2 * R2);
        ty2 = y * y / (R2 * R2);
        tz2 = z * z / (R2 * R2);

        iR = 1 / R;
        iR2 = 1 / R2;
    }

    if (R2 <= 0.0) {

        // Asymptotic expansion doesn't apply for R==0. Don't use!
        return 0.0;
    }

    double term3 = 3 * lead_weight;

    double term5 = 0.0;

    if (!cubic_cell) {

        term5 = a1 * tx2 + a2 * ty2 + a3 * tz2;
    }

    double tz4 = tz2 * tz2;

    double term7 = (b1 * tx2 + (b2 * ty2 + b3 * tz2)) * tx2 + (b4 * ty2 + b5 * tz2) * ty2 + b6 * tz4;

    double term9 = ((c1 * tx2
        + (c2 * ty2 + c3 * tz2)) * tx2
        + ((c4 * ty2 + c5 * tz2) * ty2 + c6 * tz4)) * tx2
        + ((c7 * ty2 + c8 * tz2) * ty2 + c9 * tz4) * ty2
        + c10 * tz4 * tz2;

    double iR5 = iR2 * iR2 * iR;

    // Error should be of order 1/R^11
    return (term9 + term7 + term5 + term3) * iR5 * x * y;
}
__host__ double PointDipole_Nxx(double x, double y, double z)
{
    double r2 = x * x + y * y + z * z;
    double r = sqrt(r2);

    return(((3 * x * x) - r2) / (r * r * r * r * r));

}
__host__ double PointDipole_Nxy(double x, double y, double z)
{
    double r2 = x * x + y * y + z * z;
    double r = sqrt(r2);

    return ((3 * x * y) / (r * r * r * r * r));
}

//Kernels
__global__ void DemagTensorStoreFirstOctant_OffDiagonals(MEMDATA DATA)
{
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    int Index = (i * NUMY + j) * NUMZ + k;

    if (i <= NUM && j <= NUMY && k <= NUMZ)
    {
        DATA->Nxy[dind(i, j, k)] = DATA->xFFT[FFTind(i, j, k)][0];
        DATA->Nxz[dind(i, j, k)] = DATA->yFFT[FFTind(i, j, k)][0];
        DATA->Nyz[dind(i, j, k)] = DATA->zFFT[FFTind(i, j, k)][0];
    }
    return;
}
__global__ void DemagTensorStoreFirstOctant_Diagonals(MEMDATA DATA)
{
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    int Index = (i * NUMY + j) * NUMZ + k;

    if (i <= NUM && j <= NUMY && k <= NUMZ)
    {
        DATA->Nxx[dind(i, j, k)] = DATA->xFFT[FFTind(i, j, k)][0];
        DATA->Nyy[dind(i, j, k)] = DATA->yFFT[FFTind(i, j, k)][0];
        DATA->Nzz[dind(i, j, k)] = DATA->zFFT[FFTind(i, j, k)][0];
    }

    return;
}
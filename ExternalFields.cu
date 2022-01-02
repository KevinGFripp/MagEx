#include "ExternalFields.cuh"
#include "Device_Globals_Constants.cuh"
#include "GlobalDefines.cuh"
#include "Device_Misc_Math.cuh"

__device__ Vector ExcitationFunc_CW(double t, double x, double y, double z)
{
    //sinc pulse
    Vector result;
    result.X[0] = 0.0;
    result.X[1] = 0.0;
    result.X[2] = 0.0;

   //-------------Parabolic External Magnetic Field-----------------//
    /*result.X[0] += (1.8972860e-06)*(x - (NUM / 2.) * CELL) * (x - (NUM / 2.) * CELL);
    result.X[0] += 100.0 / mu;*/
   //---------------------------------------------------------------//

   //-------------------Sinc(x)Sinc(t) Profile----------------------//
    /*double hmax = 5.0/mu;
    double kcut = (0.5 * M_PI) / CELL;
    double fcut = 30.0;
    double t_0 = 1.0;
    
    result.X[2] += hmax * sinc(kcut * (x - ((double)NUM / 2.0) * CELL)) 
                       * sinc(2 * M_PI * fcut * (t_d - t_0));*/
   //---------------------------------------------------------------//
   
  //-----------------------Uniform Sinc-----------------------------//
    double hmax = 2.0/mu;
    double fcut = 30.0;
    double t_0 = 0.75;
   
    result.X[1] += hmax * sinc(2 * M_PI * fcut * (t - t_0));
    
  //----------------------------------------------------------------//


  //--------------------- CW wave packet----------------------------//
    /* double hmax = 0.1/mu;
     double f = 16.1;
     double k = -32.41808644/ 1000.0;
     double t_0 = 1.0;
     double source = -350.0;
     double A = 1e5;
     double B = 0.4;
   
     double gaussianX = exp(-pow((x - ((double)NUM / 2. - source)*CELL),2) / (A));
     double erfT =erf(2.*t/t_0);

    result.X[2] += hmax *erfT*gaussianX*sin(2 * M_PI * f *(t))*sin(k*((x - ((double)NUM / 2.) * CELL)));
    result.X[2] += hmax * erfT*gaussianX * cos(2 * M_PI * f * (t)) *cos(k * ((x - ((double)NUM / 2.) * CELL)));*/
  //-----------------------------------------------------------------//

  //--------------------- Gaussian wave packet-----------------------//
  //   double hmax = 0.1/mu;
  //   double f = 10.5;
  //   double k = 82.0/ 1000.0;
  //   double t_0 = 15.0;
  //   double source = 0.0;
  //   double A = 1e5;
  //   double B = 0.4;
  // 
  //   double gaussianX = exp(-pow((x - ((double)NUM / 2. - source)*CELL),2) / (A));
  //   double gaussianT = exp(-pow((t_d - t_0), 2) / B);
     
  //  result.X[2] += hmax *gaussianT*gaussianX*sin(2 * M_PI * f *(t))*sin(k*((x - ((double)NUM / 2.) * CELL)));
  //  result.X[2] += hmax * gaussianT*gaussianX * cos(2 * M_PI * f * (t)) *cos(k * ((x - ((double)NUM / 2.) * CELL)));
  //---------------------------------------------------------------//

    //--------------------- Conditional Gaussian wave packet-----------------------//
    /* double hmax = 0.1/mu;
     double f = 10.5;
     double k = 82.0/ 1000.0;
     double t_0 = 12.0;
     double source = 0.0;
     double A = 1e5;
     double B = 0.4;
   
     double gaussianX = exp(-pow((x - ((double)NUM / 2. - source)*CELL),2) / (A));
     double gaussianT = exp(-pow((t_d - t_0), 2.) / B);

     if (alpha < 0.01)
     {
         result.X[2] += hmax * gaussianT * gaussianX * sin(2 * M_PI * f * (t)) * sin(k * ((x - ((double)NUM / 2.) * CELL)));
         result.X[2] += hmax * gaussianT * gaussianX * cos(2 * M_PI * f * (t)) * cos(k * ((x - ((double)NUM / 2.) * CELL)));
     }*/
    //---------------------------------------------------------------//

    return result;
}
__device__ Vector Excitation_TemporalSinc(double t, double x, double y, double z)
{
    //sinc pulse
    Vector result;
    result.X[0] = 0.0;
    result.X[1] = 0.0;
    result.X[2] = 0.0;


    double hmax = 2.0;
    double fcut = 20.0;
    double t_0 = 0.75;
  
    //Uniform Sinc
    // result.X[1] = hmax * sinc(2 * M_PI * fcut * (t_d - t_0));
    if (z <= 5 * CELLZ)
    {
        result.X[2] = hmax * sinc(2 * M_PI * fcut * (t_d - t_0));
    }

    return result;
}
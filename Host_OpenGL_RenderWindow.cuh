#include <cuda_runtime.h>
#include "DataTypes.cuh"
#include <helper_gl.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#ifndef HOST_OPENGL_RENDERWINDOW_CUH
#define HOST_OPENGL_RENDERWINDOW_CUH

void DrawTexture(void);
int initGlutDisplay(int* argc, char** argv);
void keyboardcontrol(unsigned char key, int x, int y);
void initPixelBuffer();
void initPixelBuffer_XZCrossSection();

__host__ void CreateTexture(MAG M_d, FIELD H_d);
__device__ uchar4 ColourMap_RWB(unsigned char c, bool isnegative);
__device__ uchar4 ColourMap_Jet(unsigned char c, bool isnegative);

__global__ void ProcessTextureKernel(uchar4* texturedata, MAG M_d);
__global__ void ProcessTextureKernel_Demag(uchar4* texturedata, FIELD H_d);
__global__ void ProcessTextureKernel_Exchange(uchar4* texturedata, FIELD H_d);
__global__ void ProcessTextureKernel_Torque(uchar4* texturedata, MAG M_d);

#endif // !HOST_OPENGL_RENDERWINDOW_CUH

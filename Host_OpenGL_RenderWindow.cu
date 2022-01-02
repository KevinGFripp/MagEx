#include "Host_OpenGL_RenderWindow.cuh"
#include "Device_Globals_Constants.cuh"
#include "Array_Indexing_Functions.cuh"
#include "GlobalDefines.cuh"
#include "Host_Globals.cuh"
#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include "Host_OpenGL_Globals.cuh"

void DrawTexture(void)
{
    float tex_cord_x[2];
    float tex_cord_y[2];
    //Create a quad the size of the window
    //Render texture to the quad
    if (NUM_h == NUMY_h) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, NUM_h, NUMY_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glEnable(GL_TEXTURE_2D);

        tex_cord_x[0] = 0.0;
        tex_cord_x[1] = 1.0;
        tex_cord_y[0] = 0.0;
        tex_cord_y[1] = 1.0;
    }
    else {
        glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA8, NUM_h, NUMY_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glEnable(GL_TEXTURE_RECTANGLE);

        tex_cord_x[0] = 0.0;
        tex_cord_x[1] = NUM_h;
        tex_cord_y[0] = 0.0;
        tex_cord_y[1] = NUMY_h;
    }

    glBegin(GL_QUADS);
    glTexCoord2f(tex_cord_x[0], tex_cord_y[0]);
    glVertex2f(0.0, 0.0);

    glTexCoord2f(tex_cord_x[0], tex_cord_y[1]);
    glVertex2f(0.0, NUMY_h);

    glTexCoord2f(tex_cord_x[1], tex_cord_y[1]);
    glVertex2f(NUM_h, NUMY_h);

    glTexCoord2f(tex_cord_x[1], tex_cord_y[0]);
    glVertex2f(NUM_h, 0.0);
    glEnd();
    if (NUM_h == NUMY_h) {
        glDisable(GL_TEXTURE_2D);
    }
    else {
        glDisable(GL_TEXTURE_RECTANGLE);
    }
    glFlush();
    glutSwapBuffers();
}
int initGlutDisplay(int* argc, char** argv)
{
    const int MIN_X_SIZE = 128;
    const int MAX_X_SIZE = 1536;

    const int MIN_Y_SIZE = 128;
    const int MAX_Y_SIZE = 1536;

    int SIZE_X = NUM_h;
    int SIZE_Y = NUMY_h;

    int ratio;

    if (SIZE_X >= SIZE_Y) 
    {
       ratio = SIZE_X / SIZE_Y;

       if (SIZE_X > MAX_X_SIZE)
       {
           SIZE_X = MAX_X_SIZE;
           SIZE_Y = SIZE_X / ratio;
       }
       else if (SIZE_X < MIN_X_SIZE)
       {
           SIZE_X = MIN_X_SIZE;
           SIZE_Y = SIZE_X / ratio;
       }

       if (SIZE_Y < MIN_Y_SIZE)
       {
           SIZE_Y = MIN_Y_SIZE;
       }

    }else if (SIZE_X < SIZE_Y)
    {
        ratio = SIZE_Y / SIZE_X;

        if (SIZE_Y > MAX_Y_SIZE)
        {
            SIZE_Y = MAX_Y_SIZE;
            SIZE_X = SIZE_Y / ratio;
        }
        else if (SIZE_Y < MIN_Y_SIZE)
        {
            SIZE_Y = MIN_Y_SIZE;
            SIZE_X = SIZE_Y / ratio;
        }


        if (SIZE_X < MIN_X_SIZE)
        {
            SIZE_X = MIN_X_SIZE;
        }
    }  

    
    if (OpenGlInitialised == true)
    {
        return glutGetWindow();
    }

    printf("Rendering...\n");
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(SIZE_X,SIZE_Y);
    glutInitWindowPosition(1000, 500);
    char name[100];
    sprintf(name, "MagEx:: Grid[%d x %d] Cell[%f x %f]",NUM_h,NUMY_h,CELL_h,CELLY_h);
    int wnd = glutCreateWindow(name);
    glutDisplayFunc(DrawTexture);
    glutKeyboardFunc(keyboardcontrol);
    gluOrtho2D(0, NUM_h, NUMY_h, 0);
    glewInit();
    
    OpenGlInitialised = true;
    return wnd;
}
void keyboardcontrol(unsigned char key, int x, int y)
{
    int n = 0;
    switch (key) {
    case '1':
        Viewer_contrast_host = 1;
        checkCudaErrors(cudaMemcpyToSymbol(Viewer_contrast, &Viewer_contrast_host, sizeof(int)));

        checkCudaErrors(cudaMemcpyToSymbol(Viewer_component, &n, sizeof(int)));
        break;
    case '2':
        n = 1;
        Viewer_contrast_host = 1;
        checkCudaErrors(cudaMemcpyToSymbol(Viewer_contrast, &Viewer_contrast_host, sizeof(int)));

        checkCudaErrors(cudaMemcpyToSymbol(Viewer_component, &n, sizeof(int)));
        break;
    case '3':
        n = 2;
        Viewer_contrast_host = 1;
        checkCudaErrors(cudaMemcpyToSymbol(Viewer_contrast, &Viewer_contrast_host, sizeof(int)));

        checkCudaErrors(cudaMemcpyToSymbol(Viewer_component, &n, sizeof(int)));
        break;
    case 'z':
        Viewer_zslice_host++;
        if (Viewer_zslice_host >= NUMZ_h)
        {
            Viewer_zslice_host = 0;
        }
        checkCudaErrors(cudaMemcpyToSymbol(Viewer_zslice, &Viewer_zslice_host, sizeof(int)));
        break;
    case 'p':
        Viewer_contrast_host += 3;
        checkCudaErrors(cudaMemcpyToSymbol(Viewer_contrast, &Viewer_contrast_host, sizeof(int)));
        break;
    case 'd':
        ViewDemag = true;
        ViewMag = false;
        ViewExch = false;
        ViewTorque = false;
        break;
    case 'm':
        ViewDemag = false;
        ViewMag = true;
        ViewExch = false;
        ViewTorque = false;
        break;
    case 'e':
        ViewDemag = false;
        ViewMag = false;
        ViewExch = true;
        ViewTorque = false;
        break;
    case 't':
        ViewDemag = false;
        ViewMag = false;
        ViewExch = false;
        ViewTorque = true;
        break;
    }
}
void initPixelBuffer()
{
    if (BufferCreated)
    {
        return;
    }

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * NUM_h * NUMY_h * sizeof(GLubyte), 0, GL_STREAM_DRAW);
    glGenTextures(1, &tex);
    if (NUM_h == NUMY_h) //square texture
    {
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    else {//Rectangle texture
        glBindTexture(GL_TEXTURE_RECTANGLE, tex);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }

    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
    BufferCreated = true;
}
void initPixelBuffer_XZCrossSection()
{
    if (BufferCreated_xz)
    {
        return;
    }

    glGenBuffers(1, &pbo_xz);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_xz);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * NUM_h * NUMZ_h * sizeof(GLubyte), 0, GL_STREAM_DRAW);
    glGenTextures(1, &tex_xz);
    if (NUM_h == NUMZ_h) //square texture
    {
        glBindTexture(GL_TEXTURE_2D, tex_xz);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    else {//Rectangle texture
        glBindTexture(GL_TEXTURE_RECTANGLE, tex_xz);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }

    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo_xz, cudaGraphicsMapFlagsWriteDiscard);
    BufferCreated_xz = true;
}
__host__ void CreateTexture(MAG M_d, FIELD H_d)
{
    uchar4* d_out = 0;
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_out, NULL, cuda_pbo_resource);

    dim3 numblocks;
    dim3 numthreads;
    numblocks.x = 1;
    numblocks.y = NumberofBlocksIntegrator.y;
    numblocks.z = NumberofBlocksIntegrator.z;

    numthreads.x = 1;
    numthreads.y = NumberofThreadsIntegrator.y;
    numthreads.z = NumberofThreadsIntegrator.z;

    if (ViewMag) {
        ProcessTextureKernel << <numblocks, numthreads >> > (d_out, M_d);
    }
    if (ViewDemag) {
        ProcessTextureKernel_Demag << <numblocks, numthreads >> > (d_out, H_d);
    }
    if (ViewExch) {
        ProcessTextureKernel_Exchange << <numblocks, numthreads >> > (d_out, H_d);
    }
    if (ViewTorque) {
        ProcessTextureKernel_Torque << <numblocks, numthreads >> > (d_out, M_d);
    }
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    return;
}

__device__ uchar4 ColourMap_RWB(unsigned char c, bool isnegative)
{
    if (c == 0)
    {
        return make_uchar4(255, 255, 255, 1);
    }

    if (isnegative) { return make_uchar4(256 - c, 256 - c, 255, 1); }
    else { return make_uchar4(255, 256 - c, 256 - c, 1); }
}
__device__ uchar4 ColourMap_Jet(unsigned char c, bool isnegative)
{
    //for -ve:
    //(R,G,B)
    //(0,0,128) -> (0,0,255) 255 -> 191
    //Maps 255 to 0, dark blue ->blue->cyan->light green for -ve
    //increment is 128
    uchar4 colour;
    /* if (c == 0)
     {
         return colour = make_uchar4(128, 255, 128, 1);
     }*/

    if (isnegative) {

        if (c > (256 - 64)) //dark blue to blue
        {
            colour = make_uchar4(0, 0, (128 + (256 - c) * 2), 1);
        }
        if (c > (256 - 192) && c <= (256 - 64)) //blue to cyan (0,0,255) -> (0,255,255) increment 128
        {
            colour = make_uchar4(0, 256 - 2 * (c - (256 - 192)), 255, 1);
        }
        if (c >= (0) && c <= (256 - 192)) //cyan to light green (0,255,255) -> (128,255,128) increment 128
        {
            colour = make_uchar4(128 - 2 * c, 255, 255 + 2 * (c - 64), 1);
        }
    }
    else {

        if (c > (256 - 64)) //dark red to red
        {
            colour = make_uchar4((128 + (256 - c) * 2), 0, 0, 1);
        }
        if (c > (256 - 192) && c <= (256 - 64)) //red to yellow (255,0,0) -> (255,255,0) increment 128
        {
            colour = make_uchar4(255, 256 - 2 * (c - (256 - 192)), 0, 1);
        }
        if (c >= (0) && c <= (256 - 192)) //yellow to light green (255,255,0) -> (128,255,128) increment 128
        {
            colour = make_uchar4(255 + 2 * (c - 64), 255, 128 - 2 * c, 1);
        }
    }

    return colour;
}

__global__ void ProcessTextureKernel(uchar4* texturedata, MAG M_d)
{
    //Texture format uchar4 RGBA
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < NUM && j < NUMY)
    {
        if (M_d->Mat[ind(i, j, Viewer_zslice)] == 0)
        {
            texturedata[abs(j - (NUMY - 1)) * NUM + i] = make_uchar4(0, 0, 0, 1);
        }
        else {
            double mval = Viewer_contrast * 255.0 * M_d->M[mind(0, i, j, Viewer_zslice, Viewer_component)];


            if (mval > 255.0) { mval = 255.; }
            if (mval < -255.0) { mval = -255.; }
            UCHAR m = (UCHAR)abs(mval);

            if (mval >= 0)
            {
                texturedata[abs(j - (NUMY - 1)) * NUM + i] = ColourMap_Jet(m, false);
            }
            else
            {
                texturedata[abs(j - (NUMY - 1)) * NUM + i] = ColourMap_Jet(m, true);
            }
        }
    }
    return;
}
__global__ void ProcessTextureKernel_Demag(uchar4* texturedata, FIELD H_d)
{
    //Texture format uchar4 RGBA
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < NUM && j < NUMY)
    {
        if (H_d->H_ex[find(i, j, Viewer_zslice, 0)] == 0)
        {
            texturedata[abs(j - (NUMY - 1)) * NUM + i] = make_uchar4(0, 0, 0, 1);
        }
        else {
            double Hval = Viewer_contrast * 2 * H_d->H_m[find(i, j, Viewer_zslice, Viewer_component)];

            if (Hval > 255.0) { Hval = 255.; }
            if (Hval < -255.0) { Hval = -255.; }
            UCHAR H = (UCHAR)abs(Hval);
            if (Hval >= 0)
            {
                texturedata[abs(j - (NUMY - 1)) * NUM + i] = ColourMap_Jet(H, false);
            }
            else
            {
                texturedata[abs(j - (NUMY - 1)) * NUM + i] = ColourMap_Jet(H, true);
            }
        }
    }
    return;
}
__global__ void ProcessTextureKernel_Exchange(uchar4* texturedata, FIELD H_d)
{
    //Texture format uchar4 RGBA
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < NUM && j < NUMY)
    {
        if (H_d->H_ex[find(i, j, Viewer_zslice, 0)] == 0)
        {
            texturedata[abs(j - (NUMY - 1)) * NUM + i] = make_uchar4(0, 0, 0, 1);
        }
        else {
            double Hval = Viewer_contrast * 2 * H_d->H_ex[find(i, j, Viewer_zslice, Viewer_component)];
            if (Hval > 255.0) { Hval = 255.; }
            if (Hval < -255.0) { Hval = -255.; }
            UCHAR H = (UCHAR)abs(Hval);
            if (Hval >= 0)
            {
                texturedata[abs(j - (NUMY - 1)) * NUM + i] = ColourMap_Jet(H, false);
            }
            else
            {
                texturedata[abs(j - (NUMY - 1)) * NUM + i] = ColourMap_Jet(H, true);
            }
        }
    }
    return;
}
__global__ void ProcessTextureKernel_Torque(uchar4* texturedata, MAG M_d)
{
    //Texture format uchar4 RGBA
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < NUM && j < NUMY)
    {
        if (M_d->Mat[ind(i, j, Viewer_zslice)] == 0)
        {
            texturedata[abs(j - (NUMY - 1)) * NUM + i] = make_uchar4(0, 0, 0, 1);
        }
        else {
            double mval = Viewer_contrast * 10.0 * M_d->M[mind(3, i, j, Viewer_zslice, Viewer_component)];


            if (mval > 255.0) { mval = 255.; }
            if (mval < -255.0) { mval = -255.; }
            UCHAR m = (UCHAR)abs(mval);
            if (mval >= 0)
            {
                texturedata[abs(j - (NUMY - 1)) * NUM + i] = ColourMap_Jet(m, false);
            }
            else
            {
                texturedata[abs(j - (NUMY - 1)) * NUM + i] = ColourMap_Jet(m, true);
            }
        }
    }
    return;
}
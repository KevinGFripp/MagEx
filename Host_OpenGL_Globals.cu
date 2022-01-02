#include "Host_OpenGL_GLobals.cuh"

//OpenGL GLUT Globals
int* ARG_c;
char** ARG_v;

//texture pixel objects
GLuint pbo = 0;
//void* pbo_buffer = NULL;
GLuint tex = 0;
struct cudaGraphicsResource* cuda_pbo_resource;

//texture pixel objects XZ cross-section
GLuint pbo_xz = 0;
//void* pbo_buffer = NULL;
GLuint tex_xz = 0;
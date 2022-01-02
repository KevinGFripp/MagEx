#include <helper_gl.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#ifndef HOST_OPENGL_GLOBALS_CUH
#define HOST_OPENGL_GLOBALS_CUH

//OpenGL GLUT Globals
extern int* ARG_c;
extern char** ARG_v;

//texture pixel objects
extern GLuint pbo;
//void* pbo_buffer = NULL;
extern GLuint tex;
extern struct cudaGraphicsResource* cuda_pbo_resource;

//texture pixel objects XZ cross-section
extern GLuint pbo_xz;
//void* pbo_buffer = NULL;
extern GLuint tex_xz;

#endif // !HOST_OPENGL_GLOBALS_CUH

#include "DataTypes.cuh"
#include "Host_Globals.cuh"
#include "Device_Globals_Constants.cuh"
#include "GlobalDefines.cuh"
#include "Array_Indexing_Functions.cuh"
#include <stdlib.h>
#include <stdio.h>

#ifndef MAGNETISATION_FUNCTIONS_CUH
#define MAGNETISATION_FUNCTIONS_CUH

__host__ void MeshBoundaries(MAG M);
__host__ void MagnetiseFilm3D(MAG M, double L, double H, double D, int Flag);
__host__ void MagnetiseDisk3D(MAG M, double RADIUS, double HEIGHT, int Flag);
__host__ void MagnetiseFilm3D_bool(MAG M, double L, double H, double D,
    int xpos, int ypos, int zpos, bool add);
__host__ void MagnetiseDisk3D_bool(MAG M, double RADIUS, double HEIGHT,
    int xpos, int ypos, int zpos, bool add);
__host__ void UniformState(MAG M, int x, int y, int z);
__host__ void UniformState_InRegion(MAG M, int x, int y, int z, Region R);
__host__ void VortexState(MAG M, int x, int y, int z);
__host__ void MagnetisationInitialise(MAG M);
__host__ void MagnetiseFilm3D_InRegion(MAG M, Region R);
__host__ void MagnetiseFilm3D_InRegion_bool(MAG M, Region R, int A);
__host__ void VortexState_InRegion(MAG M, int x, int y, int z, Region R);
__host__ void BlockGeometry(MAG M, OutputFormat Out);
__host__ void MagnetisationInitialise(MAG M);
__host__ void MagnetiseSphere3D(MAG M, double RADIUS, int Flag);

__host__ void BlockMaterial(MAG M, OutputFormat Out, MaterialHandle handle);
__host__ void Cuboid(MAG M, double L, double H, double D, MaterialHandle handle);
__host__ void Disk(MAG M, double RADIUS, double HEIGHT, MaterialHandle handle);
__host__ void Sphere(MAG M, double RADIUS, MaterialHandle handle);

__host__ void Cuboid_InRegion(MAG M, Region R, MaterialHandle handle);

__host__ void UniformState_InMaterial(MAG M, int x, int y, int z, MaterialHandle handle);
__host__ void VortexState_InMaterial(MAG M, int x, int y, int z, MaterialHandle handle);
#endif // !MAGNETISATION_FUNCTIONS_CUH

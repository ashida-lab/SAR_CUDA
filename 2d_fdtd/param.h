//#pragma once

#define _USE_MATH_DEFINES
#include <math.h>


#define Pi M_PI
#define mu0 (1.2566370614*1e-6)
#define e0 (8.854187817*1e-12)

#define C (2.99792458*1e8)

#define I 4.0E3 //coil current x turn [A T]     
#define alpha 0. //////////// pi/2=atack 0,0=atack pi/2

#define dx 30. //60.//grid width [m]  if two particle or boltzman dx < debye length
#define dtf 0.6E-7
#define Grid_Nx 512 //grid number x,y-axis
#define Grid_Nz 512 //grid number z-axis

#define Step 1001

#define eps 1.0E-10

#define Absorb_grid 32  //field

#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr
#define ABSORB(x) x

/*******************************************************************
type holding grid infomation
******************************************************************/
typedef struct {
	double bx, by, bz;//total magnetic field   b0-b=induced magnetic field
	double ex, ey, ez;//electric field
	double jx0, jy0, jz0;//ion current density
}Grid;

typedef struct garray {
	Grid grid[Grid_Nx + 4][Grid_Nz + 4];
	double sigma[2 * (Grid_Nx + 4)][2 * (Grid_Nz + 4)];
} garray_t;

static void HandleError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR(err) (HandleError(err,__FILE__,__LINE__))
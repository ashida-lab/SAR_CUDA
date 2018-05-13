#include "cuda_runtime.h"
#include <stdio.h>

#include "param.h"

#define DEVICE 1
//0 1070
//1 1050

#define N_block 16

__global__ void renew_grid(garray_t *g);
__global__ void renew_b4(garray_t *g);
__global__ void renew_e4(garray_t *g);

int external_current_perp(garray_t *g, int c);

extern int output(Grid grid[][Grid_Nz + 4], char *filename, int h, int w, int h2, int w2, int flag);

int fdtd(garray_t *g)
{
	int c = 0;
	int c_output=0;
	char fname[256];

	cudaEvent_t t_start, t_stop;
	float elapsedTime;

	HANDLE_ERROR(cudaSetDevice(DEVICE));//

	HANDLE_ERROR(cudaEventCreate(&t_start));
	HANDLE_ERROR(cudaEventCreate(&t_stop));

	dim3 grid4((Grid_Nx + 4) / 4, (Grid_Nz + 4) / 4);
	dim3 block4(4, 4);

	dim3 grid(Grid_Nx / N_block, Grid_Nz / N_block);
	dim3 block(N_block, N_block);

	printf("FDTD start\n");

	renew_grid << <grid4, block4 >> > (g);

	cudaDeviceSynchronize();

	for (c = 0; c < Step; c++) {
		HANDLE_ERROR(cudaEventRecord(t_start, 0));

		renew_b4 << <grid, block >> > (g);

		//cudaDeviceSynchronize();

		//external_current_perp(g, c);

		renew_b4 << <grid, block >> > (g);

		cudaDeviceSynchronize();

		external_current_perp(g, c);

		renew_e4 << <grid, block >> > (g);

		cudaDeviceSynchronize();

		HANDLE_ERROR(cudaEventRecord(t_stop, 0));
		HANDLE_ERROR(cudaEventSynchronize(t_stop));
		HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, t_start, t_stop));
		
		if (c_output > 2) {//999
			c_output = 0;

			printf("Step : %d\n", c);
			printf("Time :  %3.1f ms\n", elapsedTime);

			sprintf(fname,"img\\e-%05d.bmp", c);
			output(g->grid, fname, Grid_Nx+4, Grid_Nz+4, Grid_Nx + 4, Grid_Nz + 4, 1);
		}

		c_output++;
	}

	return(0);
}

__global__ void renew_grid(garray_t *g)
{
	unsigned int k = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int m = threadIdx.y + blockIdx.y*blockDim.y;

	g->grid[k][m].jx0 = 0.;
	g->grid[k][m].jy0 = 0.;
	g->grid[k][m].jz0 = 0.;
}

__global__ void renew_b4(garray_t *g)
{
	unsigned int k = threadIdx.x + blockIdx.x*blockDim.x + 2;
	unsigned int m = threadIdx.y + blockIdx.y*blockDim.y + 2;

	g->grid[k][m].bx = (1. - g->sigma[2 * k + 1][2 * m] / e0*dtf*0.5*0.5) / (1. + g->sigma[2 * k + 1][2 * m] / e0*dtf*0.5*0.5)*(g->grid[k][m].bx)
		+ 0.5*dtf / (1. + g->sigma[2 * k + 1][2 * m] / e0*dtf*0.5*0.5)
		*1. / 24.*(27.*(g->grid[k][m].ey - g->grid[k][m - 1].ey) - (g->grid[k][m + 1].ey - g->grid[k][m - 2].ey)) / dx;

	/*g->grid[k][m].by = (1. - g->sigma[2 * k][2 * m] / e0*dtf*0.5*0.5) / (1. + g->sigma[2 * k][2 * m] / e0*dtf*0.5*0.5)*(g->grid[k][m].by)
		- dtf*0.5 / (1. + g->sigma[2 * k][2 * m] / e0*dtf*0.5*0.5)
		*(1. / 24.*(27.*(g->grid[k][m].ex - g->grid[k][m - 1].ex) - (g->grid[k][m + 1].ex - g->grid[k][m - 2].ex)) / dx
			- 1. / 24.*(27.*(g->grid[k][m].ez - g->grid[k - 1][m].ez) - (g->grid[k + 1][m].ez - g->grid[k - 2][m].ez)) / dx);*/

	g->grid[k][m].bz = (1. - g->sigma[2 * k][2 * m + 1] / e0*dtf*0.5*0.5) / (1. + g->sigma[2 * k][2 * m + 1] / e0*dtf*0.5*0.5)*(g->grid[k][m].bz)
		- dtf*0.5 / (1. + g->sigma[2 * k][2 * m + 1] / e0*dtf*0.5*0.5)
		*1. / 24.*(27.*(g->grid[k][m].ey - g->grid[k - 1][m].ey) - (g->grid[k + 1][m].ey - g->grid[k - 2][m].ey)) / dx;
}

__global__ void renew_e4(garray_t *g)
{
	unsigned int k = threadIdx.x + blockIdx.x*blockDim.x + 2;
	unsigned int m = threadIdx.y + blockIdx.y*blockDim.y + 2;

	/*g->grid[k][m].ex = (1. - 0.5*mu0*g->sigma[2 * k][2 * m + 1] * C*C*dtf) / (1. + 0.5*mu0*g->sigma[2 * k][2 * m + 1] * C*C*dtf)*g->grid[k][m].ex
		+ C*C*dtf / (1. + 0.5*mu0*g->sigma[2 * k][2 * m + 1] * C*C*dtf)
		*(-1. / 24.*(27.*(g->grid[k][m + 1].by - g->grid[k][m].by)
			- (g->grid[k][m + 2].by - g->grid[k][m - 1].by)) / dx
			- mu0*(g->grid[k][m].jx0));//

	g->grid[k][m].ez = (1. - 0.5*mu0*g->sigma[2 * k + 1][2 * m] * C*C*dtf) / (1. + 0.5*mu0*g->sigma[2 * k + 1][2 * m] * C*C*dtf)*g->grid[k][m].ez
		+ C*C*dtf / (1. + 0.5*mu0*g->sigma[2 * k + 1][2 * m] * C*C*dtf)
		*(1. / 24.*(27.*(g->grid[k + 1][m].by - g->grid[k][m].by)
			- (g->grid[k + 2][m].by - g->grid[k - 1][m].by)) / dx
			- mu0*(g->grid[k][m].jz0));///*/

	g->grid[k][m].ey = (1. - 0.5*mu0*g->sigma[2 * k + 1][2 * m + 1] * C*C*dtf) / (1. + 0.5*mu0*g->sigma[2 * k + 1][2 * m + 1] * C*C*dtf)*g->grid[k][m].ey
		+ C*C*dtf / (1. + 0.5*mu0*g->sigma[2 * k + 1][2 * m + 1] * C*C*dtf)
		*(1. / 24.*(27.*(g->grid[k][m + 1].bx - g->grid[k][m].bx)
			- (g->grid[k][m + 2].bx - g->grid[k][m - 1].bx)) / dx
			- 1. / 24.*(27.*(g->grid[k + 1][m].bz - g->grid[k][m].bz)
				- (g->grid[k + 2][m].bz - g->grid[k - 1][m].bz)) / dx
			- mu0*(g->grid[k][m].jy0));//
}

int external_current_perp(garray_t *g, int c)
{
	int k, m;
	const double w = Pi / 20.;

	if (c < 40) {
		for (k = 0; k < Grid_Nx + 4; k++) {
			m = 48;
			g->grid[k][m].jy0 = I *sin(w*c);
		}
	}
	else {
		for (k = 0; k < Grid_Nx + 4; k++) {
			m = 48;
			g->grid[k][m].jy0 = 0;
		}
	}

	return(0);
}


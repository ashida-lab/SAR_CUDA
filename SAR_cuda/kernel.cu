#include "param.h"
#include <stdio.h>

#include "cuda_runtime.h"
#include <cuComplex.h>

#include <curand.h>
#include <curand_kernel.h>


#define DEVICE 0
#define N_block 8

#define PI   3.14159265358979323846
#define PI_2 1.57079632679489661923

extern int output_kernel(double2 *mobj, char *filename, int h, int w, int h2, int w2, int flag);

__global__ void zero_init(double2 *dst, int h, int w);
__global__ void rand_init(double2 *dst, int h, int w);
__global__ void Et_cuda(double2 *dst, unsigned char *src, int h, int w, int sh, int sw, int l, int k, float th0);
__global__ void Et_cuda_xyz(double2 *dst, int h, int w, double x, double y, double z, double s, float th0);
__global__ void transpose(double2 *dst, double2 *src, int h, int w);
__global__ void shift(double2 *dst, double2 *src, int h, int w, int sh, int sw);
__global__ void dotProduct(double2 *x, double2 *y, int h, int w);
__global__ void stoltMapping(double2 *dst, double2 *src, int h, int w);

__global__ void norm(double2 *x, int h, int w)
{
	unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int nid = threadIdx.y + blockIdx.y*blockDim.y;

	x[nid*w + gid].x = x[nid*w + gid].x / (double)w;
	x[nid*w + gid].y = x[nid*w + gid].y / (double)w;
}

int mul(double2 *rmobj, double2 *xmobj, int h, int w)
{
	cudaEvent_t t_start, t_stop;
	float elapsedTime;

	HANDLE_ERROR(cudaSetDevice(DEVICE));//

	HANDLE_ERROR(cudaEventCreate(&t_start));
	HANDLE_ERROR(cudaEventCreate(&t_stop));

	HANDLE_ERROR(cudaEventRecord(t_start, 0));

	dim3 grid(w / N_block, h / N_block);
	dim3 block(N_block, N_block);

	dotProduct << <grid, block >> > (rmobj, xmobj, h, w);


	HANDLE_ERROR(cudaEventRecord(t_stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(t_stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, t_start, t_stop));
	printf("Time :  %3.1f ms\n", elapsedTime);

	DB(output_kernel(rmobj, "06_mul.bmp", h, w, h, w, 1));

	return(0);
}

__global__ void dotProduct(double2 *x, double2 *y, int h, int w)
{
	unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int nid = threadIdx.y + blockIdx.y*blockDim.y;

	double2 temp;

	temp.x = x[nid*w + gid].x*y[nid*w + gid].x - x[nid*w + gid].y*y[nid*w + gid].y;
	temp.y = x[nid*w + gid].x*y[nid*w + gid].y + x[nid*w + gid].y*y[nid*w + gid].x;

	x[nid*w + gid] = temp;
}

__global__ void idotProduct(double2 *x, double2 *y, int h, int w)
{
	unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int nid = threadIdx.y + blockIdx.y*blockDim.y;

	double2 temp;

	temp.x = x[nid*w + gid].x*y[nid*w + gid].x + x[nid*w + gid].y*y[nid*w + gid].y;
	temp.y = x[nid*w + gid].x*y[nid*w + gid].y - x[nid*w + gid].y*y[nid*w + gid].x;

	x[nid*w + gid] = temp;
}

int trans_target(double2 *rmobj, double2 *xmobj, int h, int w)
{
	cudaEvent_t t_start, t_stop;
	float elapsedTime;

	HANDLE_ERROR(cudaSetDevice(DEVICE));//

	HANDLE_ERROR(cudaEventCreate(&t_start));
	HANDLE_ERROR(cudaEventCreate(&t_stop));

	HANDLE_ERROR(cudaEventRecord(t_start, 0));

	dim3 grid(w / N_block, h / N_block);
	dim3 block(N_block, N_block);

	transpose << <grid, block >> > (rmobj, xmobj, h, w);


	HANDLE_ERROR(cudaEventRecord(t_stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(t_stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, t_start, t_stop));
	printf("Time :  %3.1f ms\n", elapsedTime);

	DB(output_kernel(rmobj, "03_trans_target.bmp", w, h, w, h, 1));

	return(0);
}

__global__ void transpose(double2 *dst, double2 *src, int h, int w)
{
	unsigned int xgid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int ygid = threadIdx.y + blockIdx.y*blockDim.y;

	unsigned int iid = ygid*w + xgid;
	unsigned int oid = xgid*h + ygid;

	dst[oid] = src[iid];
}

int stolt_mapping(double2 *rmobj, double2 *xmobj, int h, int w)
{
	cudaEvent_t t_start, t_stop;
	float elapsedTime;

	HANDLE_ERROR(cudaSetDevice(DEVICE));//

	HANDLE_ERROR(cudaEventCreate(&t_start));
	HANDLE_ERROR(cudaEventCreate(&t_stop));

	HANDLE_ERROR(cudaEventRecord(t_start, 0));

	dim3 grid(w / N_block, h / N_block);
	dim3 block(N_block, N_block);

	stoltMapping << <grid, block >> > (rmobj, xmobj, h, w);


	HANDLE_ERROR(cudaEventRecord(t_stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(t_stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, t_start, t_stop));
	printf("Time :  %3.1f ms\n", elapsedTime);

	DB(output_kernel(rmobj, "07_stolt_mapping.bmp", h, w, h, w, 1));


	return(0);
}

__global__ void stoltMapping(double2 *dst, double2 *src, int h, int w)
{
	int gid = threadIdx.x + blockIdx.x*blockDim.x;
	int nid = threadIdx.y + blockIdx.y*blockDim.y;

	double fr = (nid - h / 2) / ((double)h*dt);
	double fa = (gid - w / 2) / ((double)w*Tp);

	double f = -fc + sqrt((fc + fr)*(fc + fr) + C*C*fa*fa / (4 * Vs*Vs));

	int x = ((int)((f)*h*dt + h / 2.));

	x = x%h;

	if (fc + fr > 0.) {
		dst[nid*w + gid] = src[x*w + gid];
	}
	else {
		dst[nid*w + gid].x = 0.;
		dst[nid*w + gid].y = 0.;
	}
}

int range_shift(double2 *rmobj, double2 *xmobj, int h, int w)
{
	cudaEvent_t t_start, t_stop;
	float elapsedTime;

	HANDLE_ERROR(cudaSetDevice(DEVICE));//

	HANDLE_ERROR(cudaEventCreate(&t_start));
	HANDLE_ERROR(cudaEventCreate(&t_stop));

	HANDLE_ERROR(cudaEventRecord(t_start, 0));

	dim3 grid(w / N_block, h / N_block);
	dim3 block(N_block, N_block);

	shift << <grid, block >> > (rmobj, xmobj, h, w, 0, w / 2);


	HANDLE_ERROR(cudaEventRecord(t_stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(t_stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, t_start, t_stop));
	printf("Time :  %3.1f ms\n", elapsedTime);

	DB(output_kernel(rmobj, "range_shift(result).bmp", h, w, h, w, 1));

	return(0);
}

int azimuth_shift(double2 *rmobj, double2 *xmobj, int h, int w)
{
	cudaEvent_t t_start, t_stop;
	float elapsedTime;

	HANDLE_ERROR(cudaSetDevice(DEVICE));//

	HANDLE_ERROR(cudaEventCreate(&t_start));
	HANDLE_ERROR(cudaEventCreate(&t_stop));

	HANDLE_ERROR(cudaEventRecord(t_start, 0));

	dim3 grid(w / N_block, h / N_block);
	dim3 block(N_block, N_block);

	shift << <grid, block >> > (rmobj, xmobj, h, w, h / 2, 0);


	HANDLE_ERROR(cudaEventRecord(t_stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(t_stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, t_start, t_stop));
	printf("Time :  %3.1f ms\n", elapsedTime);

	DB(output_kernel(rmobj, "azimuth_shift.bmp", h, w, h, w, 1));

	return(0);
}

__global__ void shift(double2 *dst, double2 *src, int h, int w, int sh, int sw)
{
	unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int nid = threadIdx.y + blockIdx.y*blockDim.y;

	dst[nid*w + gid] = src[((nid - sh) % h)*w + (gid - sw) % w];
}

int init_target(double2* xmobj, unsigned char* imobj, int h, int w, int sh, int sw, float OFFNAD)
{
	cudaEvent_t t_start, t_stop;
	float elapsedTime;

	HANDLE_ERROR(cudaSetDevice(DEVICE));//

	HANDLE_ERROR(cudaEventCreate(&t_start));
	HANDLE_ERROR(cudaEventCreate(&t_stop));

	HANDLE_ERROR(cudaEventRecord(t_start, 0));

	dim3 grid(w / N_block, h / N_block);
	dim3 block(N_block, N_block);

	rand_init << <grid, block >> > (xmobj, h, w);

	int k, l;
	for (k = 0; k < sw; k++) {
		for (l = 0; l < sh; l++) {

			Et_cuda << <grid, block >> > (xmobj, imobj, h, w, sh, sw, l, k, OFFNAD);
		}
	}


	HANDLE_ERROR(cudaEventRecord(t_stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(t_stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, t_start, t_stop));
	printf("Time :  %3.1f ms\n", elapsedTime);

	DB(output_kernel(xmobj, "01_target.bmp", h, w, h, w, 1));


	return(0);
}

__global__ void zero_init(double2 *dst, int h, int w)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	dst[j*w + i].x = 0;
	dst[j*w + i].y = 0;
}

__global__ void rand_init(double2 *dst, int h, int w)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	curandState state;

	curand_init(j, i, 0, &state);

	dst[j*w + i].x = 128 * (curand_uniform(&state) - 0.5);
	dst[j*w + i].y = 128 * (curand_uniform(&state) - 0.5);
}

__global__ void Et_cuda(double2 *dst, unsigned char *src, int h, int w, int sh, int sw, int l, int k, float th0)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	double ta = (j - h / 2)*Tp;
	double tr = 2 * R0 / C + (i - w / 2)*dt;
	double x;
	double y;
	double H = src[l*sw + k];
	double rp0, ip0;

	double tmp;

	double R;
	double r;


	x = SCALE * AZ * (k - sw / 2);
	y = SCALE * RG * (l - sh / 2);
	R = sqrt((R0*sin(th0) + y)*(R0*sin(th0) + y) + (R0*cos(th0) - H)*(R0*cos(th0) - H));
	r = sqrt((R + ve*ta)*(R + ve*ta) + (x - Vs*ta)*(x - Vs*ta));

	if (fabs(tr - 2 * r / C) < pulw / 2.&&fabs(ta - x / Vs) < ta0 / 2.) {
		tmp = -4 * PI*fc*r / C + alpha0*(tr - 2 * r / C)*(tr - 2 * r / C);
		rp0 = cos(tmp);
		ip0 = sin(tmp);
	}
	else {
		rp0 = 0;
		ip0 = 0;
	}

	dst[j*w + i].x += src[l*sw + k] * rp0;
	dst[j*w + i].y += src[l*sw + k] * ip0;
}

int init_target_xyz(double2* xmobj, char *fname, int h, int w, float OFFNAD)
{
	FILE *fp = fopen(fname, "r");
	int ret;

	cudaEvent_t t_start, t_stop;
	float elapsedTime;

	HANDLE_ERROR(cudaSetDevice(DEVICE));//

	HANDLE_ERROR(cudaEventCreate(&t_start));
	HANDLE_ERROR(cudaEventCreate(&t_stop));

	HANDLE_ERROR(cudaEventRecord(t_start, 0));

	dim3 grid(w / N_block, h / N_block);
	dim3 block(N_block, N_block);

	zero_init << <grid, block >> > (xmobj, h, w);

	double data[4];
	while ((ret = fscanf(fp, "%lf,%lf,%lf,%lf", &data[0], &data[1], &data[2], &data[3])) != EOF) {
		printf("%lf %lf %lf %lf\n", data[0], data[1], data[2], data[3]);

		/*data[1] += 125;
		Et_cuda_xyz << <grid, block >> > (xmobj, h, w, SCALE*data[0], SCALE*data[1], SCALE*data[2], SCALE*data[3]);

		data[1] -= 250;*/

		Et_cuda_xyz << <grid, block >> > (xmobj, h, w, SCALE*data[0], SCALE*data[1], SCALE*data[2], SCALE*data[3], OFFNAD);
	}


	HANDLE_ERROR(cudaEventRecord(t_stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(t_stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, t_start, t_stop));
	printf("Time :  %3.1f ms\n", elapsedTime);

	DB(output_kernel(xmobj, "01_target.bmp", h, w, h, w, 1));


	return(0);
}

__global__ void Et_cuda_xyz(double2 *dst, int h, int w, double x, double y, double z, double s, float th0)
{
	// x azimuth
	// y range

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	double ta = (j - h / 2)*Tp;
	double tr = 2 * R0 / C + (i - w / 2)*dt;
	double H = z;
	double rp0, ip0;

	double tmp;

	double R;
	double r;

	if (y < 0) {
		x = x + Vt_x*ta;//アジマス方向速度付加
		y = y + Vt_y*ta;//レンジ方向速度付加
	}

	R = sqrt((R0*sin(th0) + y)*(R0*sin(th0) + y) + (R0*cos(th0) + ANT_z - H)*(R0*cos(th0) + ANT_z - H));//アンテナ位置考慮
	r = sqrt((R + ve*ta)*(R + ve*ta) + (x - Vs*ta - ANT_x)*(x - Vs*ta - ANT_x));//アンテナ位置考慮

	if (fabs(tr - 2 * r / C) < pulw / 2.&&fabs(ta - x / Vs) < ta0 / 2.) {
		tmp = -4 * PI*fc*r / C + alpha0*(tr - 2 * r / C)*(tr - 2 * r / C);
		rp0 = cos(tmp);
		ip0 = sin(tmp);
	}
	else {
		rp0 = 0;
		ip0 = 0;
	}

	dst[j*w + i].x += s * rp0;
	dst[j*w + i].y += s * ip0;
}
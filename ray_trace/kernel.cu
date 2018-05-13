#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ObjLoader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <Windows.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#pragma comment(lib,"opencv_core320.lib")
#pragma comment(lib,"opencv_highgui320.lib")
#pragma comment(lib,"opencv_imgcodecs320.lib")
#pragma comment(lib,"opencv_imgproc320.lib") 

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		Sleep(2000);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define DIM 512
#define GND 256

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF 2e10f

#define MAX_NUM 5000

//#define ANGLE (60*M_PI/180.)

//#define SCALE 15

//c130 80
//ratel 30
//building 150

struct Sphere {
	float   r, b, g;
	float   radius;
	float   x, y, z;
};

__device__ float sph_hit(Sphere s, float ox, float oy, float *n)
{
	float dx = ox - s.x;
	float dy = oy - s.y;
	if (dx*dx + dy*dy < s.radius*s.radius) {
		float dz = sqrtf(s.radius*s.radius - dx*dx - dy*dy);
		*n = dz / s.radius;

		return dz + s.z;
	}
	return -INF;
}

struct Triangle {
	float   r, b, g;
	float   x[3], y[3], z[3];
};

__device__ float tri_hit(Triangle t, float ox, float oy)
{
	if (fabs((t.x[1] - t.x[0])*(t.y[2] - t.y[0]) - (t.x[2] - t.x[0])*(t.y[1] - t.y[0])) < 1e-3) {
		return -INF;
	}

	float d0 = (t.x[1] - t.x[0])*(oy - t.y[0]) - (t.y[1] - t.y[0])*(ox - t.x[0]);
	float d1 = (t.x[2] - t.x[1])*(oy - t.y[1]) - (t.y[2] - t.y[1])*(ox - t.x[1]);
	float d2 = (t.x[0] - t.x[2])*(oy - t.y[2]) - (t.y[0] - t.y[2])*(ox - t.x[2]);

	if (((d0 < 0 && d1 < 0 && d2 < 0) || (d0 > 0 && d1 > 0 && d2 > 0))) {
		float oz = t.z[0] + 1 / ((t.x[1] - t.x[0])*(t.y[2] - t.y[0]) - (t.x[2] - t.x[0])*(t.y[1] - t.y[0]))
			*(((t.y[2] - t.y[0])*(ox - t.x[0]) - (t.x[2] - t.x[0])*(oy - t.y[0]))*(t.z[1] - t.z[0]) + (-(t.y[1] - t.y[0])*(ox - t.x[0]) + (t.x[1] - t.x[0])*(oy - t.y[0]))*(t.z[2] - t.z[0]));

		return oz;
	}
	return -INF;
}

__global__ void kernel_tri(Triangle *tri, int s_num_triangles, int e_num_triangles, unsigned char *ptr, float *maxz) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	int i;
	float ox = (x - DIM / 2);
	float oy = (y - DIM / 2);

	float r = ptr[offset * 4 + 0] / 255., g = ptr[offset * 4 + 1] / 255., b = ptr[offset * 4 + 2] / 255.;

	float t;

	for (i = s_num_triangles; i < e_num_triangles; i++) {
		t = tri_hit(tri[i], ox, oy);
		if (t > maxz[offset]) {
			r = tri[i].r;
			g = tri[i].g;
			b = tri[i].b;
			maxz[offset] = t;
		}
	}

	ptr[offset * 4 + 0] = (unsigned char)(r * 255);
	ptr[offset * 4 + 1] = (unsigned char)(g * 255);
	ptr[offset * 4 + 2] = (unsigned char)(b * 255);
	ptr[offset * 4 + 3] = 255;
}

__global__ void kernel_sph(Sphere *s, int rep, int s_num_vertices, int e_num_vertices, unsigned char *ptr, int *index, float *maxz) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	int i;
	float ox = (x - DIM / 2);
	float oy = (y - DIM / 2);

	float r = ptr[offset * 4 + 0] / 255., g = ptr[offset * 4 + 1] / 255., b = ptr[offset * 4 + 2] / 255.;

	float t;
	float n;

	for (i = s_num_vertices; i < e_num_vertices; i++) {
		t = sph_hit(s[i], ox, oy, &n);
		if (t > maxz[offset]) {
			r = s[i].r*n;
			g = s[i].g*n;
			b = s[i].b*n;
			maxz[offset] = t;
			index[offset] = rep*MAX_NUM + i;
		}
	}

	ptr[offset * 4 + 0] = (unsigned char)(r * 255);
	ptr[offset * 4 + 1] = (unsigned char)(g * 255);
	ptr[offset * 4 + 2] = (unsigned char)(b * 255);
	ptr[offset * 4 + 3] = 255;
}

void rotation(float out[], float in[], float theta, float phi)
{
	float tmp[3];
	//y-axis

	tmp[0] = cos(theta)*in[0] + sin(theta)*in[2];
	tmp[1] = in[1];
	tmp[2] = -sin(theta)*in[0] + cos(theta)*in[2];

	//x-axis

	out[0] = tmp[0];
	out[1] = cos(phi)*tmp[1] - sin(phi)*tmp[2];
	out[2] = sin(phi)*tmp[1] + cos(phi)*tmp[2];

}

int main(int argc, char *argv[])
{
	model m;

	Sphere *s;
	Triangle *tri;

	if (argc < 5) {
		printf("ray_trace ObjFile offnadir rotation scale");
	}

	//load_obj(&m, "cube3.obj");
	//load_obj(&m, "chair.obj");
	//load_obj(&m, "Seahawk.obj");
	//load_obj(&m, "ratel_new.obj");
	load_obj(&m, argv[1]);

	float OFFNAD = M_PI / 2. - M_PI / 180.*atof(argv[2]);
	float ROTATE = M_PI / 180.*atof(argv[3]);
	float SCALE = atof(argv[4]);

	printf("param %f %f %f\n", OFFNAD, ROTATE, SCALE);

	int num_triangles = m.num_vertices / 9;
	int num_vertices = m.num_vertices / 3;

	printf("%d\n", num_triangles);
	printf("%d\n", num_vertices + GND*GND);

	HANDLE_ERROR(cudaSetDevice(0));
	HANDLE_ERROR(cudaDeviceReset());

	size_t available1, available2, total_mem;
	HANDLE_ERROR(cudaMemGetInfo(&available1, &total_mem));

	printf("available %d MB, total %d MB\n", available1 / 1000000, total_mem / 1000000);

	// capture the start time
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	unsigned char *bitmap;
	unsigned char *dev_bitmap;

	int *index;
	int *dev_index;

	float *dev_maxz;

	// allocate memory on the GPU for the output bitmap
	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, sizeof(unsigned char)*DIM*DIM * 4));
	HANDLE_ERROR(cudaMalloc((void**)&dev_index, sizeof(int)*DIM*DIM));
	HANDLE_ERROR(cudaMalloc((void**)&tri, sizeof(Triangle) * MAX_NUM));
	HANDLE_ERROR(cudaMalloc((void**)&s, sizeof(Sphere) * MAX_NUM));
	HANDLE_ERROR(cudaMalloc((void**)&dev_maxz, sizeof(float) * DIM*DIM));

	HANDLE_ERROR(cudaMemGetInfo(&available2, &total_mem));

	printf("available %d MB, total %d MB, used %d MB\n", available2 / 1000000, total_mem / 1000000, (available1 - available2) / 1000000);

	bitmap = (unsigned char*)malloc(4 * sizeof(unsigned char) * DIM*DIM);
	if (bitmap == NULL) {
		printf("mem alloc error\n");
	}

	index = (int*)malloc(sizeof(int) * DIM*DIM);
	if (index == NULL) {
		printf("mem alloc error\n");
	}

	Triangle *temp_tri = (Triangle*)malloc(sizeof(Triangle) * num_triangles);
	Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere) * (num_vertices + GND*GND));

	float *grand_p = (float*)malloc(sizeof(float) * (GND*GND) * 3);
	float *maxz = (float*)malloc(sizeof(float) * DIM*DIM);

	printf("memory size: %f MB, %f MB\n", (num_vertices + GND*GND) * sizeof(Sphere) / 1000000., num_triangles * sizeof(Triangle) / 1000000.);

	for (int i = 0; i < num_triangles; i++) {
		temp_tri[i].r = rnd(1.0f);
		temp_tri[i].g = rnd(1.0f);
		temp_tri[i].b = rnd(1.0f);

		for (int j = 0; j < 3; j++) {
			float pos[3];
			float pos_out[3];

			pos[0] = SCALE * m.vertices[i * 9 + j * 3 + 0];
			pos[1] = SCALE * m.vertices[i * 9 + j * 3 + 1];
			pos[2] = SCALE * m.vertices[i * 9 + j * 3 + 2];

			rotation(pos_out, pos, ROTATE, OFFNAD);

			temp_tri[i].x[j] = pos_out[0];
			temp_tri[i].y[j] = pos_out[1];
			temp_tri[i].z[j] = pos_out[2];
		}
	}

	for (int i = 0; i < GND*GND; i++) {
		grand_p[i * 3 + 0] = 350 * (rnd(1.0f) - 0.5);
		grand_p[i * 3 + 1] = 1 * (rnd(1.0f) - 0.5);
		grand_p[i * 3 + 2] = 350 * (rnd(1.0f) - 0.5);
	}

	for (int i = 0; i < num_vertices + GND*GND; i++) {
		temp_s[i].r = 1;
		temp_s[i].g = 1;
		temp_s[i].b = 1;

		float pos[3];
		float pos_out[3];

		if (i < num_vertices) {
			pos[0] = SCALE * m.vertices[i * 3 + 0];
			pos[1] = SCALE * m.vertices[i * 3 + 1];
			pos[2] = SCALE * m.vertices[i * 3 + 2];
		}
		else {
			pos[0] = grand_p[(i - num_vertices) * 3 + 0];
			pos[1] = grand_p[(i - num_vertices) * 3 + 1];
			pos[2] = grand_p[(i - num_vertices) * 3 + 2];
		}

		rotation(pos_out, pos, ROTATE, OFFNAD);

		temp_s[i].x = pos_out[0];
		temp_s[i].y = pos_out[1];
		temp_s[i].z = pos_out[2];

		temp_s[i].radius = rnd(1.0f) + 0.5;
	}

	for (int i = 0; i < DIM*DIM; i++) {
		index[i] = num_vertices + GND*GND;
	}

	for (int i = 0; i < DIM*DIM; i++) {
		maxz[i] = -INF;
	}

	HANDLE_ERROR(cudaMemcpy(dev_index, index, sizeof(int) * DIM*DIM, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_maxz, maxz, sizeof(float) * DIM*DIM, cudaMemcpyHostToDevice));

	// generate a bitmap from our sphere data
	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);

	for (int i = 0; i < num_triangles / MAX_NUM; i++) {
		HANDLE_ERROR(cudaMemcpy(tri, &temp_tri[i*MAX_NUM], sizeof(Triangle) * MAX_NUM, cudaMemcpyHostToDevice));

		printf("%d/%d\n", i + 1, num_triangles / MAX_NUM);
		kernel_tri << <grids, threads >> > (tri, 0, MAX_NUM, dev_bitmap, dev_maxz);
	}
	HANDLE_ERROR(cudaMemcpy(tri, &temp_tri[(int)(num_triangles / MAX_NUM)*MAX_NUM], sizeof(Triangle) * (num_triangles%MAX_NUM), cudaMemcpyHostToDevice));
	kernel_tri << <grids, threads >> > (tri, 0, (num_triangles%MAX_NUM), dev_bitmap, dev_maxz);

	for (int i = 0; i < (num_vertices + GND*GND) / MAX_NUM; i++) {
		HANDLE_ERROR(cudaMemcpy(s, &temp_s[i*MAX_NUM], sizeof(Sphere) * MAX_NUM, cudaMemcpyHostToDevice));

		printf("%d/%d\n", i + 1, (num_vertices + GND*GND) / MAX_NUM);
		kernel_sph << <grids, threads >> > (s, i, 0, MAX_NUM, dev_bitmap, dev_index, dev_maxz);
	}
	HANDLE_ERROR(cudaMemcpy(s, &temp_s[(int)((num_vertices + GND*GND) / MAX_NUM)*MAX_NUM], sizeof(Sphere) * ((num_vertices + GND*GND) % MAX_NUM), cudaMemcpyHostToDevice));
	kernel_sph << <grids, threads >> > (s, (num_vertices + GND*GND) / MAX_NUM, 0, ((num_vertices + GND*GND) % MAX_NUM), dev_bitmap, dev_index, dev_maxz);

	// copy our bitmap back from the GPU for display
	HANDLE_ERROR(cudaMemcpy(index, dev_index, sizeof(int) * DIM*DIM, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaMemcpy(bitmap, dev_bitmap, 4 * sizeof(unsigned char) * DIM*DIM, cudaMemcpyDeviceToHost));

	int *points;
	points = (int*)malloc(sizeof(int) * (num_vertices + GND*GND));
	for (int i = 0; i < num_vertices + GND*GND; i++) {
		points[i] = 0;
	}

	for (int i = 0; i < DIM*DIM; i++) {
		if (index[i] < num_vertices + GND*GND) {
			points[index[i]] = 1;
		}
	}

	int total = 0;
	for (int i = 0; i < num_vertices + GND*GND; i++) {
		total += points[i];
	}

	printf("%d/%d\n", total, num_vertices + GND*GND);

	FILE *fp = fopen("points.csv", "w");

	for (int i = 0; i < num_vertices + GND*GND; i++) {
		if (points[i] > 0) {
			float pos[3];
			float pos_out[3];

			if (i < num_vertices) {
				pos[0] = SCALE * m.vertices[i * 3 + 0];
				pos[1] = SCALE * m.vertices[i * 3 + 1];
				pos[2] = SCALE * m.vertices[i * 3 + 2];
			}
			else {
				pos[0] = grand_p[(i - num_vertices) * 3 + 0];
				pos[1] = grand_p[(i - num_vertices) * 3 + 1];
				pos[2] = grand_p[(i - num_vertices) * 3 + 2];
			}

			rotation(pos_out, pos, ROTATE, 0);

			//SAR_cuda
			//x azimuth
			//y range
			//z height

			if (i < num_vertices) {
				float nor[3];
				float nor_out[3];

				nor[0] = m.normals[i * 3 + 0];
				nor[1] = m.normals[i * 3 + 1];
				nor[2] = m.normals[i * 3 + 2];

				rotation(nor_out, nor, ROTATE, 0);

				fprintf(fp, "%e,%e,%e,%e\n", pos_out[0], -pos_out[2], pos_out[1], 1.0/*2 * fabs(nor_out[2])*/);
			}
			else {
				/*if (pos[2] < -120) {
				fprintf(fp, "%e,%e,%e,%e\n", pos_out[0], -pos_out[2], pos_out[1], 1.5 / 2);
				}
				else {*/
				fprintf(fp, "%e,%e,%e,%e\n", pos_out[0], -pos_out[2], pos_out[1], 0.5 / 2);
				//}
			}
		}
	}
	fclose(fp);


	// get stop time, and display the timing results
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float   elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
		start, stop));
	printf("Time to generate:  %3.1f ms\n", elapsedTime);

	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));

	HANDLE_ERROR(cudaFree(dev_bitmap));
	HANDLE_ERROR(cudaFree(tri));
	HANDLE_ERROR(cudaFree(s));

	free(temp_tri);
	free(temp_s);


	cv::Mat img(DIM, DIM, CV_8UC3);

	for (int k = 0; k < DIM; k++) {
		for (int l = 0; l < DIM; l++) {
			img.at < cv::Vec3b >(DIM - 1 - k, l)[0] = bitmap[4 * (k*DIM + l) + 0];
			img.at < cv::Vec3b >(DIM - 1 - k, l)[1] = bitmap[4 * (k*DIM + l) + 1];
			img.at < cv::Vec3b >(DIM - 1 - k, l)[2] = bitmap[4 * (k*DIM + l) + 2];
		}
	}

	cv::imwrite("bitmap.png", img);

}
#define _CRT_SECURE_NO_WARNINGS

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <signal.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#pragma comment(lib,"opencv_core320.lib")
#pragma comment(lib,"opencv_highgui320.lib")
#pragma comment(lib,"opencv_imgcodecs320.lib")
#pragma comment(lib,"opencv_imgproc320.lib") 

#include "param.h"

#define FILEHEADERSIZE 14
#define INFOHEADERSIZE 40
#define HEADERSIZE (FILEHEADERSIZE+INFOHEADERSIZE)

typedef struct {
	unsigned char b;
	unsigned char g;
	unsigned char r;
}Rgb;

typedef struct {
	unsigned int height;
	unsigned int width;
	unsigned int clip_height;
	unsigned int clip_width;
	Rgb *data;
}Image;

extern int fdtd(garray_t *g);

int init_grid(Grid grid[][Grid_Nz + 4], double sigma[][2 * (Grid_Nz + 4)], cv::Mat img);

Image *create_image(int width, int height, int clip_width, int clip_height);
int output(Grid grid[][Grid_Nz + 4], char *filename, int h, int w, int h2, int w2, int flag);
int write_bmp(char *filename, Image *img);
int get_color(Image *img, double *data);
int free_image(Image *img);

int main(int argc, char *argv[])
{

	if (1.414*C*dtf > dx) {
		printf("Courant error\n");
		//exit(-1);
	}
	else {
		printf("dx/dtf/C/1.414 = %f >1\n", dx / dtf / C / 1.414);
	}

	cv::Mat img = cv::imread("index.png");

	cv::Point2f center = cv::Point2f(
		static_cast<float>(img.cols / 2),
		static_cast<float>(img.rows / 2));

	double degree = 0.0;  // ‰ñ“]Šp“x
	double scale = 1.0;    // Šg‘å—¦

						   // ƒAƒtƒBƒ“•ÏŠ·s—ñ
	cv::Mat affine;
	cv::getRotationMatrix2D(center, degree, scale).copyTo(affine);

	cv::warpAffine(img, img, affine, img.size(), cv::INTER_CUBIC);

	garray_t *G;
	size_t size = sizeof(garray_t);
	cudaMallocManaged(&G, size);

	init_grid(G->grid, G->sigma, img);

	fdtd(G);

	return(0);
}

int init_grid(Grid grid[][Grid_Nz + 4], double sigma[][2 * (Grid_Nz + 4)], cv::Mat img)
{
	int k, m;
	const int absorb_n = 4;
	const double a = 1E-5;
	double x, z;
	double bx, bz;

	for (k = 0; k < Grid_Nx + 4; k++) {
		for (m = 0; m < Grid_Nz + 4; m++) {
			grid[k][m].bx = 0.;
			grid[k][m].by = 0.;
			grid[k][m].bz = 0.;

			grid[k][m].ex = 0.;
			grid[k][m].ey = 0.;
			grid[k][m].ez = 0.;

			grid[k][m].jx0 = 0.;
			grid[k][m].jy0 = 0.;
			grid[k][m].jz0 = 0.;


		}
	}

#pragma omp parallel for private(m,x,z)
	for (k = 0; k < 2 * (Grid_Nx + 4); k++) {
		for (m = 0; m < 2 * (Grid_Nz + 4); m++) {
			sigma[k][m] = 0.;


			if (k > 128 + 2 && k < 128 + 2 + 256 && m >256 - 48 && m < 256 - 48 + 256) {
				sigma[k][m] = 1e3*(255 - img.at < cv::Vec3b >((int)(127-(k - 128 - 2) / 2), (int)((m - 256 + 48) / 2))[0]);
			}


			if (k - 4 <= 2 * Absorb_grid) {
				x = (2 * Absorb_grid - (k - 4)) / 4.;
			}
			else if (k - 4 >= 2 * Grid_Nx - 2 * Absorb_grid) {
				x = ((k - 4) - (2 * Grid_Nx - 2 * Absorb_grid)) / 4.;
			}
			else {
				x = 0.;
			}

			if (m - 4 <= 2 * Absorb_grid) {
				z = (2 * Absorb_grid - (m - 4)) / 4.;
			}
			else if (m - 4 >= 2 * Grid_Nz - 2 * Absorb_grid) {
				z = ((m - 4) - (2 * Grid_Nz - 2 * Absorb_grid)) / 4.;
			}
			else {
				z = 0.;
			}

			sigma[k][m] += a*(0.1*sqrt(pow(x, 4) + pow(z, 4)) + 0.01*sqrt(pow(x, 8) + pow(z, 8)));


		}
	}

	return(0);
}

int output(Grid grid[][Grid_Nz + 4], char *filename, int h, int w, int h2, int w2, int flag)
{
	int i, j;
	Image *img;
	double *ampd;

	ampd = (double*)malloc(h*w * sizeof(double));

	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			ampd[w*i + j] = grid[i][j].ey;
		}
	}

	if ((img = (Image *)create_image(w, h, w2, h2)) == NULL) {
		fprintf(stderr, "Can't create image\n");
		exit(0);
	}

	get_color(img, ampd);

	write_bmp(filename, img);

	free_image(img);
	free(ampd);

	return(0);
}

int write_bmp(char *filename, Image *img)
{
	int i, j;
	FILE *fp;
	int real_width;
	unsigned char *bmp_line_data;
	unsigned char header_buf[HEADERSIZE];
	unsigned int file_size;
	unsigned int offset_to_data;
	unsigned long info_header_size;
	unsigned int planes;
	unsigned int color;
	unsigned long compress;
	unsigned long data_size;
	long xppm;
	long yppm;

	if ((fp = fopen(filename, "wb")) == NULL) {
		fprintf(stderr, "Can't open %s.", filename);
		return(1);
	}

	real_width = img->clip_width * 3 + img->clip_width % 4;

	file_size = img->clip_height*real_width + HEADERSIZE;
	offset_to_data = HEADERSIZE;
	info_header_size = INFOHEADERSIZE;
	planes = 1;
	color = 24;
	compress = 0;
	data_size = img->clip_height*real_width;
	xppm = 1;
	yppm = 1;

	header_buf[0] = 'B';
	header_buf[1] = 'M';
	memcpy(header_buf + 2, &file_size, sizeof(file_size));
	header_buf[6] = 0;
	header_buf[7] = 0;
	header_buf[8] = 0;
	header_buf[9] = 0;
	memcpy(header_buf + 10, &offset_to_data, sizeof(file_size));
	header_buf[11] = 0;
	header_buf[12] = 0;
	header_buf[13] = 0;

	memcpy(header_buf + 14, &info_header_size, sizeof(info_header_size));
	header_buf[15] = 0;
	header_buf[16] = 0;
	header_buf[17] = 0;
	memcpy(header_buf + 18, &img->clip_width, sizeof(img->clip_width));
	memcpy(header_buf + 22, &img->clip_height, sizeof(img->clip_height));
	memcpy(header_buf + 26, &planes, sizeof(planes));
	memcpy(header_buf + 28, &color, sizeof(color));
	memcpy(header_buf + 30, &compress, sizeof(compress));
	memcpy(header_buf + 34, &data_size, sizeof(data_size));
	memcpy(header_buf + 38, &xppm, sizeof(xppm));
	memcpy(header_buf + 42, &yppm, sizeof(yppm));
	header_buf[46] = 0;
	header_buf[47] = 0;
	header_buf[48] = 0;
	header_buf[49] = 0;
	header_buf[50] = 0;
	header_buf[51] = 0;
	header_buf[52] = 0;
	header_buf[53] = 0;

	fwrite(header_buf, sizeof(unsigned char), HEADERSIZE, fp);

	if ((bmp_line_data = (unsigned char *)malloc(sizeof(unsigned char)*real_width)) == NULL) {
		fprintf(stderr, "Can't get memory\n");
		fclose(fp);
		return(1);
	}

	for (i = 0; i < img->clip_height; i++) {
		for (j = 0; j < img->clip_width; j++) {
			bmp_line_data[j * 3] = img->data[(img->clip_height - i - 1)*img->clip_width + j].b;
			bmp_line_data[j * 3 + 1] = img->data[(img->clip_height - i - 1)*img->clip_width + j].g;
			bmp_line_data[j * 3 + 2] = img->data[(img->clip_height - i - 1)*img->clip_width + j].r;
		}

		for (j = img->clip_width * 3; j < real_width; j++) {
			bmp_line_data[j] = 0;
		}
		fwrite(bmp_line_data, sizeof(unsigned char), real_width, fp);
	}

	free(bmp_line_data);

	fclose(fp);

	return(0);
}

Image *create_image(int width, int height, int clip_width, int clip_height)
{
	Image *img;

	if ((img = (Image *)malloc(sizeof(Image))) == NULL) {
		fprintf(stderr, "Can't get memory\n");
		return(NULL);
	}

	if ((img->data = (Rgb*)malloc(sizeof(Rgb)*clip_width*clip_height)) == NULL) {
		fprintf(stderr, "Can't get memory\n");
		free(img);
		return(NULL);
	}

	img->width = width;
	img->height = height;
	img->clip_width = clip_width;
	img->clip_height = clip_height;

	return(img);
}

int free_image(Image *img)
{
	free(img->data);
	free(img);

	return(0);
}

int get_color(Image *img, double *data)
{
	int i, j;
	double min_value, max_value;
	double range;
	double r, g, b;

	for (i = 0; i < img->clip_height; i++) {
		for (j = 0; j < img->clip_width; j++) {
			img->data[(img->clip_height - i - 1)*img->clip_width + j].b = 0;
			img->data[(img->clip_height - i - 1)*img->clip_width + j].g = 0;
			img->data[(img->clip_height - i - 1)*img->clip_width + j].r = 0;
		}
	}

	min_value = max_value = 0;

	for (i = 0; i < img->clip_height; i++) {
		for (j = 0; j < img->clip_width; j++) {
			if (min_value > data[i*img->width + j]) {
				min_value = data[i*img->width + j];
			}

			if (max_value < data[i*img->width + j]) {
				max_value = data[i*img->width + j];
			}
		}
	}
	fprintf(stdout, "%E %E\n", min_value, max_value);

	range = max_value - min_value;

	for (i = 0; i < img->clip_height; i++) {
		for (j = 0; j < img->clip_width; j++) {
			r = 255 * (data[i*img->width + j] - min_value) / range;
			g = 255 * (data[i*img->width + j] - min_value) / range;
			b = 255 * (data[i*img->width + j] - min_value) / range;

			img->data[(img->clip_height - i - 1)*img->clip_width + j].r = (char)r;
			img->data[(img->clip_height - i - 1)*img->clip_width + j].g = (char)g;
			img->data[(img->clip_height - i - 1)*img->clip_width + j].b = (char)b;
		}
	}

	return(0);
}

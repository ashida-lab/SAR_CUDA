#define _CRT_SECURE_NO_WARNINGS

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#pragma comment(lib,"opencv_core320.lib")
#pragma comment(lib,"opencv_highgui320.lib")
#pragma comment(lib,"opencv_imgcodecs320.lib")
#pragma comment(lib,"opencv_imgproc320.lib") 

#define _USE_MATH_DEFINES
#include <math.h>

#include "param.h"

#define TEST 0

#define PI M_PI

#define AMP(a,b) (sqrt((a)*(a)+(b)*(b)))

#define DB(x) x

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

extern int init_target(double2 *xmobj, unsigned char *imobj, int h, int w, int sh, int sw, float OFFNAD);
extern int init_target_xyz(double2* xmobj, char *fname, int h, int w, float OFFNAD);
extern int trans_target(double2 *rmobj, double2 *xmobj, int h, int w);
extern int range_shift(double2 *rmobj, double2 *xmobj, int h, int w);
extern int azimuth_shift(double2 *rmobj, double2 *xmobj, int h, int w);
extern int mul(double2 *rmobj, double2 *xmobj, int h, int w);
extern int stolt_mapping(double2 *rmobj, double2 *xmobj, int h, int w);


int fft_range_target(double2 *rmobj, double2 *xmobj, int h, int w);
int fft_azimuth_target(double2 *rmobj, double2 *xmobj, int h, int w);
int ifft_range_target(double2 *rmobj, double2 *xmobj, int h, int w);
int ifft_azimuth_target(double2 *rmobj, double2 *xmobj, int h, int w);

int init_filter(double2 *filter, int h, int w);
int Er(double fr, double fa, double *rp, double *ip);

int fft_core(double2 *dst, double2 *src, int h, int w, int direction);
Image *create_image(int width, int height, int clip_width, int clip_height);
int output(double2 *rm, char *filename, int h, int w, int h2, int w2, int flag);
int output_kernel(double2 *mobj, char *filename, int h, int w, int h2, int w2, int flag);
int output_kernel_complex(double2 *mobj, char *filename, int h, int w, int h2, int w2);
int write_bmp(char *filename, Image *img);
int get_color(Image *img, double *data);
int free_image(Image *img);

using namespace std;

int main(int argc, char *argv[])
{
	float OFFNAD = M_PI / 180. * 45;

	if (argc < 2) {
		printf("SAR_cuda offnadir");
	}
	else {
		OFFNAD = M_PI / 180.*atof(argv[1]);
	}

	HANDLE_ERROR(cudaSetDevice(0));//
	HANDLE_ERROR(cudaDeviceReset());

	int h, w;

	h = (int)(TA / Tp);
	w = (int)(TR / dt);

	cv::Mat img = cv::imread("index.png");
	//cv::Mat img = cv::imread("C:\\Users\\yasu\\Desktop\\lena.png");

	cv::Point2f center = cv::Point2f(
		static_cast<float>(img.cols / 2),
		static_cast<float>(img.rows / 2));

	double degree = 0.0;  // 回転角度
	double scale = 1.0;    // 拡大率

						   // アフィン変換行列
	cv::Mat affine;
	cv::getRotationMatrix2D(center, degree, scale).copyTo(affine);

	cv::warpAffine(img, img, affine, img.size(), cv::INTER_CUBIC);

	FILE* fp_p = fopen("param.txt", "w");

	fprintf(fp_p, "sampling freq %e\n", 1 / dt);
	fprintf(fp_p, "band width %e\n", fabs(alpha0)*pulw / M_PI);
	fprintf(fp_p, "range resolution %e\n", M_PI*C / 2 / fabs(alpha0) / pulw);
	fprintf(fp_p, "wave length %e\n", C / fc);
	fprintf(fp_p, "doppler band width %e\n", 2 * Vs*Vs*fc / C / R0*ta0);
	fprintf(fp_p, "azimuth resolution %e\n", C / fc*R0 / 2 / Vs / ta0);
	fprintf(fp_p, "range bin %f\n", TR / dt);
	fprintf(fp_p, "azimuth hit %f\n", TA / Tp);

	fclose(fp_p);
	//while (1);

	int sw = img.cols;
	int sh = img.rows;

	if (TEST == 1) {
		sw = 1;
		sh = 1;
	}

	unsigned char *im;
	double2 *xm;
	double2 *filter;

	im = (unsigned char*)malloc(sw*sh * sizeof(unsigned char));
	xm = (double2*)malloc(h*w * sizeof(double2));
	filter = (double2*)malloc(h*w * sizeof(double2));

	unsigned char *imobj;
	double2 *xmobj;
	double2 *rmobj;
	double2 *tmobj;
	double2 *wmobj;
	double2 *filter_mobj;
	double2 *rfilter_mobj;

	HANDLE_ERROR(cudaMalloc((void**)&imobj, sw*sh * sizeof(unsigned char)));
	HANDLE_ERROR(cudaMalloc((void**)&xmobj, w*h * sizeof(double2)));
	HANDLE_ERROR(cudaMalloc((void**)&rmobj, w*h * sizeof(double2)));
	HANDLE_ERROR(cudaMalloc((void**)&tmobj, w*h * sizeof(double2)));
	HANDLE_ERROR(cudaMalloc((void**)&wmobj, w / 2 * sizeof(double2)));

	HANDLE_ERROR(cudaMalloc((void**)&filter_mobj, w*h * sizeof(double2)));
	HANDLE_ERROR(cudaMalloc((void**)&rfilter_mobj, w*h * sizeof(double2)));


	for (int k = 0; k < sw; k++) {
		for (int l = 0; l < sh; l++) {
			im[l*sw + k] = img.at < cv::Vec3b >(l, k)[0];
		}
	}

	if (TEST == 1) {
		im[0] = 1;
	}

	HANDLE_ERROR(cudaMemcpy(imobj, im, sw*sh * sizeof(unsigned char), cudaMemcpyHostToDevice));

	//init_target(xmobj, imobj, h, w, sh, sw,OFFNAD);
	init_target_xyz(xmobj, "points.csv", h, w, OFFNAD);

	fft_range_target(rmobj, xmobj, h, w);

	trans_target(xmobj, rmobj, h, w);

	h = (int)(TR / dt);
	w = (int)(TA / Tp);

	fft_azimuth_target(rmobj, xmobj, h, w);

	range_shift(xmobj, rmobj, h, w);
	azimuth_shift(rmobj, xmobj, h, w);

	init_filter(filter, h, w);

	HANDLE_ERROR(cudaMemcpy(filter_mobj, filter, w*h * sizeof(double2), cudaMemcpyHostToDevice));

	mul(rmobj, filter_mobj, h, w);

	stolt_mapping(tmobj, rmobj, h, w);

	ifft_azimuth_target(rmobj, tmobj, h, w);

	trans_target(xmobj, rmobj, h, w);

	h = (int)(TA / Tp);
	w = (int)(TR / dt);

	ifft_range_target(rmobj, xmobj, h, w);

	range_shift(xmobj, rmobj, h, w);

	output_kernel_complex(xmobj, "test.raw", h, w, h, w);


	return(0);
}

int fft_range_target(double2 *rmobj, double2 *xmobj, int h, int w)
{
	fft_core(rmobj, xmobj, h, w, CUFFT_FORWARD);

	DB(output_kernel(rmobj, "02_fft_range_target.bmp", h, w, h, w, 1));

	return(0);
}

int fft_azimuth_target(double2 *rmobj, double2 *xmobj, int h, int w)
{
	fft_core(rmobj, xmobj, h, w, CUFFT_FORWARD);

	DB(output_kernel(rmobj, "04_fft_azimuth_target.bmp", h, w, h, w, 1));

	return(0);
}

int ifft_range_target(double2 *rmobj, double2 *xmobj, int h, int w)
{
	fft_core(rmobj, xmobj, h, w, CUFFT_INVERSE);

	DB(output_kernel(rmobj, "09_ifft_range_target.bmp", h, w, h, w, 1));

	return(0);
}

int ifft_azimuth_target(double2 *rmobj, double2 *xmobj, int h, int w)
{
	fft_core(rmobj, xmobj, h, w, CUFFT_INVERSE);

	DB(output_kernel(rmobj, "08_ifft_azimuth_target.bmp", h, w, h, w, 1));

	return(0);
}

int init_filter(double2 *filter, int h, int w)
{
	int i, j;
	double rp, ip;

	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			Er((i - h / 2.) / (h*dt), (j - w / 2.) / (w*Tp), &rp, &ip);
			((double*)filter)[(2 * w*i) + 2 * j + 0] = rp;
			((double*)filter)[(2 * w*i) + 2 * j + 1] = ip;
		}
	}

	DB(output(filter, "05_filter.bmp", h, w, h, w, 0));

	return(0);
}

int Er(double fr, double fa, double *rp, double *ip)
{
	double f;

	double temp = (fc + fr)*(fc + fr) - C*C*fa*fa / (4 * Vs*Vs);
	if (temp < 0.) {
		*rp = 0.;
		*ip = 0.;
	}
	else {
		f = 4 * PI*(R0 + 1275) / C*sqrt(temp) + PI*PI*fr*fr / alpha0;//適当にシフト
		*rp = cos(f);
		*ip = sin(f);
	}

	return(0);
}

int fft_core(double2 *dst, double2 *src, int h, int w, int direction)
{
	// CUFFT plan simple API
	cufftHandle plan;
	cufftPlan1d(&plan, w, CUFFT_Z2Z, h);

	// Transform signal and kernel
	printf("Transforming signal cufftExecC2C\n");
	cufftExecZ2Z(plan, (cufftDoubleComplex *)src, (cufftDoubleComplex *)dst, direction);

	cufftDestroy(plan);

	return(0);
}

int output(double2 *rm, char *filename, int h, int w, int h2, int w2, int flag)
{
	int i, j;
	Image *img;
	double *ampd;

	ampd = (double*)malloc(h*w * sizeof(double));
	if (flag == 1) {
		for (i = 0; i < h; i++) {
			for (j = 0; j < w; j++) {
				ampd[w*i + j] = AMP(((double*)rm)[2 * w*i + 2 * j], ((double*)rm)[2 * w*i + 2 * j + 1]);
			}
		}
	}
	else {
		for (i = 0; i < h; i++) {
			for (j = 0; j < w; j++) {
				ampd[w*i + j] = ((double*)rm)[2 * w*i + 2 * j];
			}
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

int output_kernel(double2 *mobj, char *filename, int h, int w, int h2, int w2, int flag)
{
	int i, j;
	int ret;
	Image *img;
	double *ampd;
	double2 *rm;

	rm = (double2*)malloc(h*w * sizeof(double2));

	HANDLE_ERROR(cudaMemcpy(rm, mobj, h*w * sizeof(double2), cudaMemcpyDeviceToHost));

	ampd = (double*)malloc(h*w * sizeof(double));
	if (flag == 1) {
		for (i = 0; i < h; i++) {
			for (j = 0; j < w; j++) {
				ampd[w*i + j] = AMP(((double*)rm)[2 * w*i + 2 * j], ((double*)rm)[2 * w*i + 2 * j + 1]);
			}
		}
	}
	else {
		for (i = 0; i < h; i++) {
			for (j = 0; j < w; j++) {
				ampd[w*i + j] = ((double*)rm)[2 * w*i + 2 * j];
			}
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
	free(rm);

	return(0);
}

int output_kernel_complex(double2 *mobj, char *filename, int h, int w, int h2, int w2)
{
	int i, j;
	int ret;
	double2 *rm;

	rm = (double2*)malloc(h*w * sizeof(double2));

	HANDLE_ERROR(cudaMemcpy(rm, mobj, h*w * sizeof(double2), cudaMemcpyDeviceToHost));

	FILE *fp;

	fp = fopen(filename, "wb");

	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			fwrite(&((double*)rm)[2 * w*i + 2 * j], sizeof(double), 1, fp);
		}
	}

	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			fwrite(&((double*)rm)[2 * w*i + 2 * j + 1], sizeof(double), 1, fp);
		}
	}

	fclose(fp);
	free(rm);

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

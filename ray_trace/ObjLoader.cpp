#include <stdio.h>
#include <stdlib.h>
#include "ObjLoader.h"

/* 初期バッファサイズ */
#define DEF_BUF_SIZE 2

/* 浮動小数点数バッファ */
typedef struct {
	int buf_size;
	int current_index;
	float *buf;
} float_buffer;

/* 整数バッファ */
typedef struct {
	int buf_size;
	int current_index;
	int *buf;
} int_buffer;

/* 頂点座標の読み込み */
void read_vertices(const char *line, float_buffer *vs);

/* 法線ベクトルの読み込み */
void read_normals(const char *line, float_buffer *vns);

/* UV座標の読み込み */
void read_uvs(const char *line, float_buffer *vts);

/* インデックスの読み込み */
void read_indices(const char *line, int_buffer *fs);

/* モデルの作成 */
void create_model(model *m,
	float_buffer *vs, float_buffer *vns,
	float_buffer *vts, int_buffer *fs);

/* 浮動小数点数バッファの操作 */
float_buffer *alloc_float_buffer(void);
void free_float_buffer(float_buffer *fbuf);
void add_float(float_buffer *fbuf, float value);
float get_float(float_buffer *fbuf, int index);

/* 整数バッファの操作 */
int_buffer *alloc_int_buffer(void);
void free_int_buffer(int_buffer *ibuf);
void add_int(int_buffer *ibuf, int value);
int get_int(int_buffer *ibuf, int index);

/*==============================*
** OBJファイルの読み込み
**==============================*/
void load_obj(model *m, const char *file_name)
{
	FILE *fp;
	char line[1024];
	float_buffer *vs, *vns, *vts;
	int_buffer *fs;

	fp = fopen(file_name, "r");
	if (!fp) {
		fprintf(stderr, "Cannot open %s.\n", file_name);
		exit(EXIT_FAILURE);
	}

	vs = alloc_float_buffer();
	vns = alloc_float_buffer();
	vts = alloc_float_buffer();
	fs = alloc_int_buffer();

	while (!feof(fp)) {
		fgets(line, sizeof(line), fp);
		if (line[0] == 'v' && line[1] == ' ') {
			read_vertices(line, vs);
		}
		else if (line[0] == 'v' && line[1] == 'n' && line[2] == ' ') {
			read_normals(line, vns);
		}
		else if (line[0] == 'v' && line[1] == 't' && line[2] == ' ') {
			read_uvs(line, vts);
		}
		else if (line[0] == 'f' && line[1] == ' ') {
			read_indices(line, fs);
		}
	}

	create_model(m, vs, vns, vts, fs);

	free_float_buffer(vs);
	free_float_buffer(vns);
	free_float_buffer(vts);
	free_int_buffer(fs);

	fclose(fp);
}

/*------------------------------*
** 頂点座標の読み込み
**------------------------------*/
void read_vertices(const char *line, float_buffer *vs)
{
	float x, y, z;
	int count;

	count = sscanf(line, "%*s%f%f%f", &x, &y, &z);

	if (count == 3) {
		add_float(vs, x);
		add_float(vs, y);
		add_float(vs, z);
	}
}

/*------------------------------*
** 法線ベクトルの読み込み
**------------------------------*/
void read_normals(const char *line, float_buffer *vns)
{
	float x, y, z;
	int count;

	count = sscanf(line, "%*s%f%f%f", &x, &y, &z);

	if (count == 3) {
		add_float(vns, x);
		add_float(vns, y);
		add_float(vns, z);
	}
}

/*------------------------------*
** UV座標の読み込み
**------------------------------*/
void read_uvs(const char *line, float_buffer *vts)
{
	float u, v;
	int count;

	count = sscanf(line, "%*s%f%f", &u, &v);

	if (count == 2) {
		add_float(vts, u);
		add_float(vts, v);
	}
}

/*------------------------------*
** インデックスの読み込み
**------------------------------*/
void read_indices(const char *line, int_buffer *fs)
{
	int v1, v2, v3;
	int t1, t2, t3;
	int n1, n2, n3;
	int count;

	count = sscanf(line,
		"%*s %d%*c%d%*c%d "
		"%d%*c%d%*c%d "
		"%d%*c%d%*c%d",
		&v1, &t1, &n1, &v2, &t2, &n2, &v3, &t3, &n3);

	//printf("%s %d\n",line, count);

	if (count == 9) {
		add_int(fs, v1);
		add_int(fs, t1);
		add_int(fs, n1);
		add_int(fs, v2);
		add_int(fs, t2);
		add_int(fs, n2);
		add_int(fs, v3);
		add_int(fs, t3);
		add_int(fs, n3);
	}

	count = sscanf(line,
		"%*s %d%*c%*c%d "
		"%d%*c%*c%d "
		"%d%*c%*c%d",
		&v1, &n1, &v2, &n2, &v3, &n3);

	if (count == 6) {
		add_int(fs, v1);
		add_int(fs, 0);
		add_int(fs, n1);
		add_int(fs, v2);
		add_int(fs, 0);
		add_int(fs, n2);
		add_int(fs, v3);
		add_int(fs, 0);
		add_int(fs, n3);
	}
}

/*------------------------------*
** モデルの作成
**------------------------------*/
void create_model(model *m,
	float_buffer *vs, float_buffer *vns,
	float_buffer *vts, int_buffer *fs)
{
	int i, j;

	m->num_vertices = fs->current_index / 3 * 3;
	m->vertices = (float*)malloc(sizeof(float) * m->num_vertices);

	if (!m->vertices) {
		fprintf(stderr, "Memory allocation error.\n");
		exit(EXIT_FAILURE);
	}

	m->num_normals = fs->current_index / 3 * 3;
	m->normals = (float*)malloc(sizeof(float) * m->num_normals);

	if (!m->normals) {
		fprintf(stderr, "Memory allocation error.\n");
		exit(EXIT_FAILURE);
	}

	m->num_uvs = fs->current_index / 3 * 2;
	//m->uvs = (float*)malloc(sizeof(float) * m->num_uvs);

	/*if (!m->uvs) {
		fprintf(stderr, "Memory allocation error.\n");
		exit(EXIT_FAILURE);
	}*/

	for (i = 0; i < fs->current_index; i++) {
		int idx = fs->buf[i] - 1;
		if (i % 3 == 0) {
			for (j = 0; j < 3; j++) {
				m->vertices[i / 3 * 3 + j] = vs->buf[idx * 3 + j];
			}
		}
		else if (i % 3 == 1) {
			/*for (j = 0; j < 2; j++) {
				m->uvs[i / 3 * 2 + j] = vts->buf[idx * 2 + j];
			}*/
		}
		else {
			for (j = 0; j < 3; j++) {
				m->normals[i / 3 * 3 + j] = vns->buf[idx * 3 + j];
			}
		}

	}
}

/*------------------------------*
** 浮動小数点数バッファの割り当て
**------------------------------*/
float_buffer *alloc_float_buffer(void)
{
	float_buffer *fbuf;

	fbuf = (float_buffer*)malloc(sizeof(float_buffer));
	if (!fbuf) {
		fprintf(stderr, "Memory allocation error.\n");
		exit(EXIT_FAILURE);
	}

	fbuf->buf_size = DEF_BUF_SIZE;
	fbuf->current_index = 0;
	fbuf->buf = (float*)malloc(sizeof(float) * fbuf->buf_size);

	return fbuf;
}

/*------------------------------*
** 浮動小数点数バッファの解放
**------------------------------*/
void free_float_buffer(float_buffer *fbuf)
{
	free(fbuf->buf);
	free(fbuf);
}

/*------------------------------*
** バッファに浮動小数点数を追加
**------------------------------*/
void add_float(float_buffer *fbuf, float value)
{
	fbuf->buf[fbuf->current_index] = value;
	fbuf->current_index++;
	if (fbuf->current_index >= fbuf->buf_size) {
		fbuf->buf_size *= 2;
		fbuf->buf = (float*)realloc(fbuf->buf, sizeof(float) * fbuf->buf_size);
		if (!fbuf->buf) {
			fprintf(stderr, "Memory allocation error.\n");
			exit(EXIT_FAILURE);
		}
	}
}

/*------------------------------*
** バッファから浮動小数点数を取り出し
**------------------------------*/
float get_float(float_buffer *fbuf, int index)
{
	return fbuf->buf[index];
}

/*------------------------------*
** 整数バッファの割り当て
**------------------------------*/
int_buffer *alloc_int_buffer(void)
{
	int_buffer *ibuf;

	ibuf = (int_buffer*)malloc(sizeof(int_buffer));
	if (!ibuf) {
		fprintf(stderr, "Memory allocation error.\n");
		exit(EXIT_FAILURE);
	}

	ibuf->buf_size = DEF_BUF_SIZE;
	ibuf->current_index = 0;
	ibuf->buf = (int*)malloc(sizeof(int) * ibuf->buf_size);

	return ibuf;
}

/*------------------------------*
** 整数バッファの解放
**------------------------------*/
void free_int_buffer(int_buffer *ibuf)
{
	free(ibuf->buf);
	free(ibuf);
}

/*------------------------------*
** バッファに整数を追加
**------------------------------*/
void add_int(int_buffer *ibuf, int value)
{
	ibuf->buf[ibuf->current_index] = value;
	ibuf->current_index++;
	if (ibuf->current_index >= ibuf->buf_size) {
		ibuf->buf_size *= 2;
		ibuf->buf = (int*)realloc(ibuf->buf, sizeof(int) * ibuf->buf_size);
		if (!ibuf->buf) {
			fprintf(stderr, "Memory allocation error.\n");
			exit(EXIT_FAILURE);
		}
	}
}

/*------------------------------*
** バッファから整数を取り出し
**------------------------------*/
int get_int(int_buffer *ibuf, int index)
{
	return ibuf->buf[index];
}
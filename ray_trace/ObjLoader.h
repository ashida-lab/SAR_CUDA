#ifndef OBJLOADER_H_INCLUDE
#define OBJLOADER_H_INCLUDE

/* モデルデータ */
typedef struct {
	float *vertices;
	int num_vertices;
	float *normals;
	int num_normals;
	float *uvs;
	int num_uvs;
} model;

/* OBJファイルの読み込み */
void load_obj(model *m, const char *file_name);

#endif
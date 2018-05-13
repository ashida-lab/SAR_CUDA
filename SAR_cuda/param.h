#include <stdio.h>

#define dt (0.25e-7) // 1/サンプリング周波数
#define Tp (1e-3) // 1/PRF
#define pulw (35.1e-6) //パルス幅
#define ta0 (1.024/2) //合成開口時間
#define fc (1275e6) //送信周波数
#define alpha0 (-4.3e11*PI)
//#define th0 (45./180*PI) //オフナディア角

#define Vs (7.9e3) //速度
#define ve 0//(30)

#define R0 (4e5) //スラントレンジ
#define C (3e8) //光速

#define TR (2.56e-5) //ゲート時間
#define TA (1.024) //撮像時間

#define RG  (PI*C / 2. / fabs(alpha0) / pulw)
#define AZ  (C / fc*R0 / 2. / Vs / ta0)

#define ANT_x (0.) //Vs*Tp/2) //アジマス
#define ANT_z (0.) //エレベーション

#define Vt_x (0.) //ターゲット速度アジマス
#define Vt_y (0.) //ターゲット速度レンジ

#define SCALE 8.

static void HandleError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR(err) (HandleError(err,__FILE__,__LINE__))

#define DB(x) x


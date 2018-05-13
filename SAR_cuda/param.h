#include <stdio.h>

#define dt (0.25e-7) // 1/�T���v�����O���g��
#define Tp (1e-3) // 1/PRF
#define pulw (35.1e-6) //�p���X��
#define ta0 (1.024/2) //�����J������
#define fc (1275e6) //���M���g��
#define alpha0 (-4.3e11*PI)
//#define th0 (45./180*PI) //�I�t�i�f�B�A�p

#define Vs (7.9e3) //���x
#define ve 0//(30)

#define R0 (4e5) //�X�����g�����W
#define C (3e8) //����

#define TR (2.56e-5) //�Q�[�g����
#define TA (1.024) //�B������

#define RG  (PI*C / 2. / fabs(alpha0) / pulw)
#define AZ  (C / fc*R0 / 2. / Vs / ta0)

#define ANT_x (0.) //Vs*Tp/2) //�A�W�}�X
#define ANT_z (0.) //�G���x�[�V����

#define Vt_x (0.) //�^�[�Q�b�g���x�A�W�}�X
#define Vt_y (0.) //�^�[�Q�b�g���x�����W

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


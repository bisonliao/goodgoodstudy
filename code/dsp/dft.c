#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define Pi (3.141592653589793238462643383280)

int dft( double* input, int N, int i_start, double *ReX, double *ImX, double * MagX, double * PhaseX)
{
	int i, k;
	for (k = 0; k <= N/2;++k)
	{
		ReX[k] = 0;

		for (i = i_start; i < (N+i_start); ++i)
		{
			ReX[k] += input[i-i_start] * cos(2*Pi/N * i * k);
		}

		ImX[k] = 0;

		for (i = i_start; i < (N+i_start); ++i)
		{
			ImX[k] -= input[i-i_start] * sin(2*Pi/N * i * k);
		}
	}
	for (k = 0; k <= N/2;++k)
	{
		MagX[k] = sqrt(ReX[k]*ReX[k]+ImX[k]*ImX[k]);
		PhaseX[k] = atan2(ImX[k], ReX[k]);

		// for cos, phase -pi is pi
		if (PhaseX[k]-(-Pi) < 0.0000000001)
		{
			PhaseX[k] = Pi;
		}
	}
	return 0;
}
void print(const char * msg, double * v, int len)
{
	printf("%s ", msg);
	int i;
	for (i = 0; i < len; ++i)
	{
		printf("%.14f ", v[i]);
	}
	printf("\n");
}

int main()
{
	double input[] = {  1,2,3,4,5,6,7,8,0,0,0,0};
	double input2[] = {  0,1,2,3,4,5,6,7,8,0,0,0};

	#define NUM (12)

	double ReX[NUM/2+1];
	double ImX[NUM/2+1];
	double MagX[NUM/2+1];
	double PhaseX[NUM/2+1], PhaseX2[NUM/2+1];

	dft(input, NUM, 0, ReX, ImX, MagX, PhaseX);
	dft(input2, NUM, 0, ReX, ImX, MagX, PhaseX2);
	print("PhaseX:", PhaseX, NUM/2+1);
	print("PhaseX2:", PhaseX2, NUM/2+1);

	int i;
	for (i = 0; i < NUM/2+1; ++i)
	{
		printf("%f\n", (NUM)* (PhaseX2[i] - PhaseX[i])/(2*Pi));

	}

	return 0;
}



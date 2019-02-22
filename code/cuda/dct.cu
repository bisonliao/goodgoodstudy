

#include <stdio.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <lame.h>
#include <fcntl.h>  
#include <sys/types.h>  
#include <sys/stat.h>  
#include <io.h>  
#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>


#include <helper_cuda.h>

#include <stdlib.h>

#include <string.h>

#include <stdio.h>

#include <stdint.h>

#include <math.h>

#include "common.h"







__device__ int dft(double* input, int N, int i_start, double *ReX, double *ImX)

{

	int i, k;
//	int id = blockDim.x * blockIdx.x + threadIdx.x;
	//printf("thread id#%d begin dft()\n", id);
	//printf("N=%d\n", N);

	for (k = 0; k <= N / 2; ++k)
	{
		ReX[k] = 0;

		//printf("%d!!\n", __LINE__);
		for (i = i_start; i < (N + i_start); ++i)
		{
			ReX[k] += input[i - i_start] * cos(2 * Pi / N * i * k);
		}
		//printf("%d!!\n", __LINE__);
		if (k == 0 || (N%2)==1 && (N/2)==k )
		{
			ReX[k] = ReX[k] / N;
		}
		else
		{
			ReX[k] = ReX[k] / N * 2;
		}
	
		//printf("%d!!\n", __LINE__);

		ImX[k] = 0;
		for (i = i_start; i < (N + i_start); ++i)
		{
			ImX[k] -= input[i - i_start] * sin(2 * Pi / N * i * k);
		}
		ImX[k] = -ImX[k] / N * 2;
		
		//printf("%d, k=%d!!\n", __LINE__, k);
	}
	//printf("dft success\n");
	return 0;

}

__device__ int idft(const double *ReX, const double *ImX, int Num, int i_start, double * output)

{

	int i, k;

	/*
	INPUT[i_] =  Sum[ReX[k] *Cos[2 Pi/NUM*k*i], {k, 0, NUM/2}] +   Sum[ImX[k]*Sin[2 Pi/NUM * i * k], {k, 0, NUM/2}];
	*/
	for (i = i_start; i < Num+i_start; ++i)
	{
		output[i-i_start] = 0;
		for (k = 0; k <= Num / 2; ++k)
		{
			output[i-i_start] += ReX[k] * cos(2 * Pi / Num * i * k) + ImX[k] * sin(2 * Pi / Num * i * k);
		}
	}

	return 0;

}

__device__ int dct(const double *input, int len, double *output)
{
	const int len2 = len + len - 1;
	const int resultLen = 1 + len2 / 2;
	//FRAME_SZ is a const micro, value=256
	double input2[FRAME_SZ * 2 - 1];
	double ReX[FRAME_SZ];
	double ImX[FRAME_SZ];
	/*
	 double *input2 = NULL, *ReX = NULL, *ImX = NULL;

	input2 = (double *)malloc(len2 * sizeof(double));
	ReX = (double*)malloc(resultLen * sizeof(double));
	ImX = (double*)malloc(resultLen * sizeof(double));

	

	if (input2 == NULL || ReX == NULL || ImX == NULL)
	{
		printf("failed to allocate memory!\n");
		return -1;
	}
	*/
	int i;
	for (i = 0; i < len; i++)
	{
		input2[i] = input[i];
	}
	///printf("%d!!\n", __LINE__);
	for (i=0; i < len-1; i++)
	{
		int index = len - 2 - i;
		input2[i+len] = input[index];
	}
	///printf("%d!!\n", __LINE__);
	dft(input2, len2, -(len - 1), ReX, ImX);
	///printf("%d!!\n", __LINE__);
	for (i = 0; i < resultLen; ++i)
	{
		//printf("ImX#%d=%f\n",i, ImX[i]);
		output[i] = ReX[i];
	}
	//printf("%d!!\n", __LINE__);
/*
	free(ReX);
	free(ImX);
	free(input2);
	*/
	

	return 0;

}
__device__ int idct(const double * input, int len, double * output)
{
	int len2 = len + len - 1;
	double ImX[FRAME_SZ];
	double output2[FRAME_SZ * 2 - 1];
	/*
	double * ImX = (double*)malloc(len * sizeof(double));
	double * output2 = (double*)malloc(len2 * sizeof(double));
	if (ImX == NULL||output2 == NULL)
	{
		return -1;
	}
	*/
	int i;
	for (i = 0; i < len; ++i)
	{
		ImX[i] = 0;
	}
	idft(input, ImX, len2, -(len - 1), output2);
	for (i = 0; i < len; ++i)
	{
		output[i] = output2[i];
	}
	/*
	free(ImX);
	free(output2);
	*/
	return 0;
}

__global__ void huge_dct_on_device(const double * input, int frameSz, int frameNr, double * output)
{

	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < frameNr)
	{
		if (dct(input + id*frameSz, frameSz, output + id*frameSz) != 0)
		{
			printf("thread#%d failed!\n", id);
		}
	
	}
}

__global__ void huge_idct_on_device(const double * input, int frameSz, int frameNr, double * output)
{

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < frameNr)
	{
		idct(input + i*frameSz, frameSz, output + i*frameSz);
	}
}

int  huge_dct(const double * input, int frameSz, int frameNr, double * output)
{
	double *d_input = NULL;
	double *d_output = NULL;
	cudaError_t err;
	size_t size = frameSz * frameNr * sizeof(double);
	printf("huge_dct() start\n");

	err = cudaMalloc((void **)&d_input, size);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device input (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}
	err = cudaMalloc((void **)&d_output, size);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device output (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}

	err = cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to memcpy input to device  (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}
	int threadsPerBlock = 256;
	int blocksPerGrid = (frameNr + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	huge_dct_on_device << <blocksPerGrid, threadsPerBlock >> > (d_input, frameSz, frameNr, d_output);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch huge_dct_on_device kernel (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}
	err = cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to memcpy output from device  (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}
	cudaFree(d_input);
	cudaFree(d_output);
	printf("huge_dct() success!\n");
	return 0;

}
int  huge_idct(const double * input, int frameSz, int frameNr, double * output)
{
	double *d_input = NULL;
	double *d_output = NULL;
	cudaError_t err;
	size_t size = frameSz * frameNr * sizeof(double);
	printf("huge_idct() start!\n");

	err = cudaMalloc((void **)&d_input, size);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device input (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}
	err = cudaMalloc((void **)&d_output, size);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device output (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}

	err = cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to memcpy input to device  (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}
	int threadsPerBlock = 256;
	int blocksPerGrid = (frameNr + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	huge_idct_on_device << <blocksPerGrid, threadsPerBlock >> > (d_input, frameSz, frameNr, d_output);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch huge_dct_on_device kernel (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}
	err = cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to memcpy output from device  (error code %s)!\n", cudaGetErrorString(err));
		return -1;
	}
	cudaFree(d_input);
	cudaFree(d_output);
	printf("huge_idct() success!\n");
	return 0;

}

#if 0
int main(void)
{
	

	int k;
	int frameNr = 10000;
	int itera = 10;
	int i;
	double *input = (double*)malloc(FRAME_SZ*frameNr*itera * sizeof(double));
	double *output = (double*)malloc(FRAME_SZ*frameNr*itera * sizeof(double));
	double *output2 = (double*)malloc(FRAME_SZ*frameNr*itera * sizeof(double));
	if (input == NULL || output == NULL || output2 == NULL)
	{
		return -1;
	}
	for (i = 0; i < FRAME_SZ*frameNr*itera; ++i)
	{
		input[i] = i + 0.1;
	}
	for (k = 0; k < itera; ++k)
	{
		printf("\nitera:%d\n", k);
		if (huge_dct(input + k*frameNr, FRAME_SZ, frameNr, output + k*frameNr) != 0)
		{
			return -1;
		}


		if (huge_idct(output + k*frameNr, FRAME_SZ, frameNr, output2 + k*frameNr) != 0)
		{
			return -1;
		}
		for (i = 0; i < FRAME_SZ*frameNr; ++i)
		{
			if (abs(output2[i + k*frameNr] - input[i + k*frameNr]) > 0.1)
			{
				fprintf(stderr, "different! %f <=> %f\n", output2[i + k*frameNr], input[i + k*frameNr]);
				return -1;
			}
		}

		for (i = 56; i < 60; ++i)
		{
			printf("data#%d: %f, %f, %f\n", i, input[i + k*frameNr], output2[i + k*frameNr], output[i + k*frameNr]);
		}


	}
	free(input);
	free(output);
	free(output2);

	printf("Done\n");
	return 0;
}
#endif

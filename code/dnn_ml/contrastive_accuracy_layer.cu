#include "caffe/layers/contrastive_accuracy_layer.hpp"
#include <math.h>

namespace caffe {

	template <typename Dtype>
	__global__ void Calc_distance(int chn_num, int batchsz, 
		                 const Dtype* data1, const Dtype*data2, const Dtype*data3, 
		                 float margin, int*accuracy)
	{
		int id = blockDim.x * blockIdx.x + threadIdx.x;
		if (id >= batchsz)
		{
			return;
		}
		float distance = 0;
		int y;
		for (int j = 0; j < chn_num; ++j)
		{
			int index = id*chn_num + j;
			distance += (data1[index] - data2[index])*(data1[index] - data2[index]);
		}

		distance = sqrt(distance);

		if (distance < margin)
		{
			y = 1;
		}
		else
		{
			y = 0;
		}
		Dtype label = data3[id];
		if (y == label) // right!!
		{
			accuracy[id] = 1;
		}
		else
		{
			accuracy[id] = 0;
		}
	}
	template <typename Dtype>
	void ContrastiveAccuracyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		

		int N = bottom[0]->num();
		int C = bottom[0]->channels();
		
		const Dtype* data1 = bottom[0]->gpu_data();
		const Dtype* data2 = bottom[1]->gpu_data();
		const Dtype* data3 = bottom[2]->gpu_data();

		int * d_accuracy = NULL;
		
		cudaError_t  err = cudaMalloc((void **)&d_accuracy, N*sizeof(int));
		if (err != cudaSuccess)
		{
			LOG(FATAL)<<"Failed to allocate device memory for result!"<< cudaGetErrorString(err);
			return;
		}
		Calc_distance << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> > (C, N,
			data1, data2, data3,
			margin_, d_accuracy);
		
		int * h_accuracy = (int*)malloc(N * sizeof(int));
		if (h_accuracy == NULL)
		{
			cudaFree(d_accuracy);
			LOG(FATAL) << "Failed to allocate host memory for result!";
			return;
		}
		
		err = cudaMemcpy(h_accuracy, d_accuracy, N*sizeof(int), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			cudaFree(d_accuracy);
			free(h_accuracy);
			LOG(FATAL) << "Failed to copy device memory for result!" << cudaGetErrorString(err);
			return;
		}
		
		int i;
		int total = 0;
		for (i = 0; i < N; ++i)
		{
			if (h_accuracy[i]) { total++; }
		}
		cudaFree(d_accuracy);
		free(h_accuracy);
	
		Dtype result = total / (N*1.0);
		Dtype* top_data = top[0]->mutable_gpu_data(); 
		caffe_copy(1, &result, top_data);
		

	}

	INSTANTIATE_LAYER_GPU_FUNCS(ContrastiveAccuracyLayer);  

}

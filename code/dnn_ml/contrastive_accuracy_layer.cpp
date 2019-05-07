#include "caffe/layers/contrastive_accuracy_layer.hpp"
#include <math.h>

namespace caffe {
	
	
	template <typename Dtype>
	void ContrastiveAccuracyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Layer<Dtype>::LayerSetUp(bottom, top);

		margin_ = this->layer_param_.contrastive_accuracy_param().margin();


		CHECK_EQ(bottom[0]->num_axes(), 2);
		CHECK_EQ(bottom[1]->num_axes(), 2);
		CHECK_EQ(bottom[2]->num_axes(), 1);

		// two bottom blob have same shape
		
		CHECK_EQ(bottom[0]->num(), bottom[1]->num());
		/*
		CHECK_EQ(bottom[0]->height(), bottom[1]->height());
		CHECK_EQ(bottom[0]->width(), bottom[1]->width());
		CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());

		CHECK_EQ(bottom[0]->height(), 1);
		CHECK_EQ(bottom[0]->width(), 1);
		CHECK_EQ(bottom[2]->channels(), 1);
		*/


		CHECK_EQ(bottom[2]->num(), bottom[1]->num());

	}
	template <typename Dtype>
	void ContrastiveAccuracyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		//top[0]->Reshape(1, 1, 1, 1); // output a float value

		vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
		top[0]->Reshape(top_shape);
	}
	template <typename Dtype>
	void ContrastiveAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{

		int N = bottom[0]->num();
		int C = bottom[0]->channels();

		int i, j;
		int accuracy = 0;
		for (i = 0; i < N; ++i)
		{

			//两个 blob之间的欧式距离
			float distance = 0;
			Dtype y = 0;

			const Dtype* data1 = bottom[0]->cpu_data();
			const Dtype* data2 = bottom[1]->cpu_data();
			const Dtype* data3 = bottom[2]->cpu_data();

			for (j = 0; j < C; ++j)
			{
				int index = i*C + j;
				distance += (data1[index] - data2[index])*(data1[index] - data2[index]);
			}

			distance = sqrt(distance);
#if 1
			if (distance < margin_)
			{
				y = 1;
			}
			else
			{
				y = 0;
			}

			Dtype label = data3[i];
			if (y == label) // right!!
			{
				accuracy++;
			}
#endif
		}

		top[0]->mutable_cpu_data()[0] = accuracy / (N*1.0);

	}

#ifdef CPU_ONLY
	STUB_GPU(ContrastiveAccuracyLayer);
#endif
	INSTANTIATE_CLASS(ContrastiveAccuracyLayer);
	REGISTER_LAYER_CLASS(ContrastiveAccuracy);

}

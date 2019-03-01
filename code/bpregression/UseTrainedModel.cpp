// UseTrainedModel.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include "layer_reg.h"
#include <caffe/caffe.hpp>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe; 
using std::string;
 
const int label_size = 21;
const int input_size = 33;
const int scale = 3;

unsigned int get_blob_index(boost::shared_ptr< Net<float> > & net, char *query_blob_name)
{
	std::string str_query(query_blob_name);
	vector< string > const & blob_names = net->blob_names();
	for (unsigned int i = 0; i != blob_names.size(); ++i)
	{
		if (str_query == blob_names[i])
		{
			return i;
		}
	}
	LOG(FATAL) << "Unknown blob name: " << str_query;
}


  
void regression(boost::shared_ptr<Net<float> > net, double x)
{
	float data_input[1][1][1]; 

	data_input[0][0][0] = x;

	Blob<float>* input_blobs = net->input_blobs()[0];
	switch (Caffe::mode())
	{
	case Caffe::CPU:
		memcpy(input_blobs->mutable_cpu_data(), data_input,
			sizeof(float) * input_blobs->count());
		break;
	case Caffe::GPU:

		cudaMemcpy(input_blobs->mutable_gpu_data(), data_input,
			sizeof(float) * input_blobs->count(), cudaMemcpyHostToDevice);


		break;
	default:
		LOG(FATAL) << "Unknown Caffe mode.";
	}

	net->Forward();

	int index = get_blob_index(net, "fc8_sbt");
	boost::shared_ptr<Blob<float> > blob = net->blobs()[index];
	unsigned int num_data = blob->count();
	int i;
	double expectValue = 2 * x*x + x + 3;
	for (i = 0; i < num_data; ++i)
	{
		const float *blob_ptr = (const float *)blob->cpu_data();
		printf("%f,%f,%f,\n", *(blob_ptr+i), expectValue, *(blob_ptr + i) / expectValue);
	}
	
	printf("\n");
	


}


#if 1


int main(int argc, char **argv) {

	
	
	


	const char *proto = "E:\\DeepLearning\\structural_regression\\deploy.prototxt";
	const char *model = "E:\\DeepLearning\\structural_regression\\struct_reg_iter_10000.caffemodel";
	//char *mean_file = "H:\\Models\\Caffe\\imagenet_mean.binaryproto";
	Phase phase = TEST;
	Caffe::set_mode(Caffe::GPU); 
	
	
	boost::shared_ptr<Net<float> > net(new caffe::Net<float>(proto, phase));
	net->CopyTrainedLayersFrom(model);
	
	int i;
	srand(time(NULL));
	for (i = 0; i < 100; ++i)
	{
		double x = ((double)rand()) / RAND_MAX * 10;
		regression(net, x);
	}




	return 0;

}
#else

#endif


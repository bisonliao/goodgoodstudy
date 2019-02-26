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

void caffe_forward(boost::shared_ptr< Net<float> > & net, float *data_ptr)
{
	Blob<float>* input_blobs = net->input_blobs()[0];
	switch (Caffe::mode())
	{
	case Caffe::CPU:
		memcpy(input_blobs->mutable_cpu_data(), data_ptr,
			sizeof(float) * input_blobs->count());
		break;
	case Caffe::GPU:
		
		cudaMemcpy(input_blobs->mutable_gpu_data(), data_ptr,
		sizeof(float) * input_blobs->count(), cudaMemcpyHostToDevice);
		
		
		break;
	default:
		LOG(FATAL) << "Unknown Caffe mode.";
	}
	net->Forward();
}
  
void classify(boost::shared_ptr<Net<float> > net,
	cv::Mat & img)
{
	float data_input[1][4]; 

	int sub_i, sub_j;
	for (sub_i = 0; sub_i < 1; ++sub_i)
	{
		for (sub_j = 0; sub_j < 4; ++sub_j)
		{
			data_input[sub_i][sub_j] = (float)(img.at<uchar>( sub_i,  sub_j));
		}
	}

	caffe_forward(net, (float*)data_input);
	int index = get_blob_index(net, "prob");
	boost::shared_ptr<Blob<float> > blob = net->blobs()[index];
	unsigned int num_data = blob->count();
	int i;
	for (i = 0; i < num_data; ++i)
	{
		const float *blob_ptr = (const float *)blob->cpu_data();
		printf("%f\n", *(blob_ptr+i) );
	}
	
	printf("\n");
	


}


#if 1


int main(int argc, char **argv) {

	if (argc < 3)
	{
		printf("usage:%s x y \n", argv[0]);
		return 255;
	}
	uint16_t x = atoi(argv[1]); unsigned char * px = (unsigned char*)&x;
	uint16_t y = atoi(argv[2]); unsigned char * py = (unsigned char*)&y;

	

	cv::Mat img(1,4, CV_8UC1);
	img.at<uchar>(0, 0) = *px;
	img.at<uchar>(0, 1) = *(px+1);
	img.at<uchar>(0, 2) = *py;
	img.at<uchar>(0, 3) = *(py+1);
	/*
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	cv::imshow("Display window", img);
	cv::waitKey(0);
	cv::destroyWindow("Display window");
	*/
	
	


	const char *proto = "E:\\DeepLearning\\structural_classify\\train\\deploy.prototxt";
	const char *model = "E:\\DeepLearning\\structural_classify\\train\\alexNet_range_iter_20000.caffemodel";
	//char *mean_file = "H:\\Models\\Caffe\\imagenet_mean.binaryproto";
	Phase phase = TEST;
	Caffe::set_mode(Caffe::GPU);
	
	
	boost::shared_ptr<Net<float> > net(new caffe::Net<float>(proto, phase));
	net->CopyTrainedLayersFrom(model);
	
	int i;
	for (i = 0; i < 100; ++i)
	{
		classify(net, img);
	}



	return 0;

}
#else

#endif


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
 

#define input_size (227)
#define input_channel  (3)


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
	printf("input blobs size:%d, uchar count:%d\n", net->input_blobs().size(), input_blobs->count());
	
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
	float data_input[input_channel][input_size][input_size];
	
	int width, height, chn;
	
	
	for (height = 0; height < input_size; ++height)
	{
		for (width = 0; width < input_size; ++width)
		{
			cv::Point3_<uchar>* p = img.ptr<cv::Point3_<uchar> >(height, width);
			data_input[0][height][width] = p->x;//B
			data_input[1][height][width] = p->y;//G
			data_input[2][height][width] = p->z;//R

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





int main(int argc, char **argv) {

	if (argc < 2)
	{
		printf("usage:%s filename \n", argv[0]);
		return 255;
	}
	

	cv::Mat img = cv::imread(argv[1] );
	if (img.data == NULL)
	{
		fprintf(stderr, "failed to load image file\n");
		return -1;
	}
	if (img.channels() != input_channel)
	{
		fprintf(stderr, "image channel is not 3,abort!\n");
		return -1;
	}
	cv::Mat img2 = img;
	cv::resize(img, img2, cv::Size(input_size, input_size), 0.0, 0.0, cv::INTER_CUBIC);
	img = img2;

	
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	cv::imshow("Display window", img);
	cv::waitKey(0);
	cv::destroyWindow("Display window");

	
	
	


	const char *proto = "E:\\DeepLearning\\age_gender_classify\\deploy_gender.prototxt";
	const char *model = "E:\\DeepLearning\\age_gender_classify\\gender_net.caffemodel";
	//char *mean_file = "H:\\Models\\Caffe\\imagenet_mean.binaryproto";
	Phase phase = TEST;
	Caffe::set_mode(Caffe::GPU);
	
	
	boost::shared_ptr<Net<float> > net(new caffe::Net<float>(proto, phase));
	net->CopyTrainedLayersFrom(model);
	
	classify(net, img);



	return 0;

}



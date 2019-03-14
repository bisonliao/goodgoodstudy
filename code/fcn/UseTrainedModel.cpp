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
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe; 
using std::string;
using namespace cv;
 

const int input_size = 200;


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


  
void classify(boost::shared_ptr<Net<float> > net, const Mat & img)
{
	float data_input[3][input_size][input_size];

	if (img.channels() != 3)
	{
		return;
	}

	Mat inputMat;
	resize(img, inputMat, Size(input_size, input_size), 0, 0, INTER_CUBIC);

	int width, height, chn;


	for (height = 0; height < input_size; ++height)
	{
		for (width = 0; width < input_size; ++width)
		{
			cv::Point3_<uchar>* p = inputMat.ptr<cv::Point3_<uchar> >(height, width);
			data_input[0][height][width] = p->x;//B
			data_input[1][height][width] = p->y;//G
			data_input[2][height][width] = p->z;//R
			//训练的时候，输入数据减过均值127，那么调用的时候也减一下才对
			data_input[0][height][width] -= 127;
			data_input[1][height][width] -= 127;
			data_input[2][height][width] -= 127;
			

		}
	}


	

	Blob<float>* input_blobs = net->input_blobs()[0];
	printf("inpupt blob x count:%d\n", input_blobs->count());
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

	int index = get_blob_index(net, "score");
	boost::shared_ptr<Blob<float> > blob = net->blobs()[index];
	unsigned int num_data = blob->count();
	printf("output blob index:%d,  y count:%d\n", index, blob->count());

	int class_idx;
	Mat result(input_size, input_size, CV_8UC1, Scalar(0));
	const int CLASS_NUM = 2;
	const float *blob_ptr = (const float *)blob->cpu_data();
	for (height = 0; height < input_size; height++)
	{
		for (width = 0; width < input_size; width++)
		{
			int max = blob_ptr[0 * (input_size*input_size) + height * input_size + width];
			int max_idx = 0;
			for (class_idx = 1; class_idx < CLASS_NUM; ++class_idx)
			{
				int offset = class_idx * (input_size*input_size) + height * input_size + width;
				if (blob_ptr[offset] > max)
				{
					max = blob_ptr[offset];
					max_idx = class_idx;
				}
			}
			if (max_idx == 1)
			{
				result.at<uchar>(height, width) = 255;
			}

		}
	}
	namedWindow("Display window", WINDOW_AUTOSIZE);
	Mat toShow;
	cvtColor(inputMat, toShow, COLOR_BGR2GRAY);
	toShow.push_back(result);

	imshow("Display window", toShow);                // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
/*
	imshow("Display window", result);                // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
*/	
	return;


}


#if 1


int main(int argc, char **argv) {

	
	if (argc < 2)
	{
		fprintf(stderr, "usage: %s imagefilename\n", argv[0]);
		return -1;
	}
	


	const char *proto = "E:\\DeepLearning\\myRPN\\deploy_fcn.prototxt";
	const char *model = "E:\\DeepLearning\\myRPN\\myrpn_iter_20000.caffemodel";
	Phase phase = TEST;
	Caffe::set_mode(Caffe::GPU); 
	
	
	boost::shared_ptr<Net<float> > net(new caffe::Net<float>(proto, phase));
	net->CopyTrainedLayersFrom(model);

	Mat img = imread(argv[1]);
	
	classify(net, img);




	return 0;

}
#else

#endif


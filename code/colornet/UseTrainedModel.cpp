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
 

const int input_size = 224;



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
	static float data_input[1][input_size][input_size];

	if (img.channels() != 3)
	{
		return;
	}

	Mat resizedMat, yuvMat;
	resize(img, resizedMat, Size(input_size, input_size), 0, 0, INTER_CUBIC);
	cvtColor(resizedMat, yuvMat, CV_BGR2YUV);

	int width, height, chn;

	float meanval[3] = { 104, 116, 123 };
	for (height = 0; height < input_size; ++height)
	{
		for (width = 0; width < input_size; ++width)
		{
			const cv::Point3_<uchar>* p = yuvMat.ptr<cv::Point3_<uchar> >(height, width);
			data_input[0][height][width] = p->x;//Y
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

	int index = get_blob_index(net, "softmax");
	boost::shared_ptr<Blob<float> > blob = net->blobs()[index];
	unsigned int num_data = blob->count();
	printf("output blob index:%d,  y count:%d\n", index, blob->count());

	int class_idx;
	
	Mat result(input_size/2, input_size/2, CV_8UC3, Scalar(0));
	const int CLASS_NUM = 256;
	const float *blob_ptr = (const float *)blob->cpu_data();
	for (height = 0; height < input_size/2; height++)
	{
		for (width = 0; width < input_size/2; width++)
		{
			float max = blob_ptr[0 * (input_size/2*input_size/2) + height * input_size/2 + width];
			int max_idx = 0;
			for (class_idx = 1; class_idx < CLASS_NUM; ++class_idx)
			{
				int offset = class_idx * (input_size/2*input_size/2) + height * input_size/2 + width;
				if (blob_ptr[offset] > max)
				{
					max = blob_ptr[offset];
					max_idx = class_idx;
				}
				/*
				if (width == 0 && height == 0)
				{
					printf("%f ", blob_ptr[offset]);
				}
				*/
			}
			printf("%d ", max_idx);
			cv::Point3_<uchar>* p = result.ptr<cv::Point3_<uchar> >(height, width);
			const cv::Point3_<uchar>* p2 = yuvMat.ptr<cv::Point3_<uchar> >(height*2, width*2);
			p->x = p2->x;
			p->y = max_idx & 0xf0;
			p->z = (max_idx & 0x0f) << 4;
		}
		printf("\n");
	}
	printf("\n");
	resize(result, resizedMat, Size(input_size, input_size), 0, 0, INTER_CUBIC);
	cvtColor(resizedMat, result, CV_YUV2BGR);

	namedWindow("Display window", WINDOW_AUTOSIZE);
	Mat toShow;
	resize(img, toShow, Size(input_size, input_size), 0, 0, INTER_CUBIC);
	toShow.push_back(result);

	imshow("Display window", toShow);                // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window

	return;


}

int denoise(Mat & img)
{
	if (img.channels() != 1)
	{
		Mat tmp = img;
		cvtColor(tmp, img, COLOR_BGR2GRAY);
	}
	
	Mat dst;// Mat dst = img ; will result problem!!!
	int kernel_sz = 5;
	//Mat kernel = (Mat_<float>(3, 3) <<0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111);
	Mat kernel = Mat::ones(kernel_sz, kernel_sz, CV_32F) / (float)(kernel_sz *kernel_sz);

	filter2D(img, dst, -1, kernel);
	printf("width:%d,%d\n", img.cols, dst.cols);
	int width, height;
	for (height = 0; height < dst.rows; ++height)
	{
		for (width = 0; width < dst.cols; ++width)
		{
			uchar v = dst.at<uchar>(height, width);
			const uchar thr = 255 * 0.5;
			if (v < thr)
			{
				dst.at<uchar>(height, width) = 0;
			}
			else
			{
				dst.at<uchar>(height, width) = 255;
			}	
		}
	}
	namedWindow("Display window", WINDOW_AUTOSIZE);
	Mat toShow = img;
	toShow.push_back(dst);
	imshow("Display window", toShow);
	waitKey(0);

	dst.copyTo(img);
	return 0;
}
#if 1


int main(int argc, char **argv) {

	const char * imgname = NULL;
	if (argc < 2)
	{
		fprintf(stderr, "usage: %s imagefilename\n", argv[0]);
		imgname = "e:\\DeepLearning\\data\\coco\\train2014\\COCO_train2014_000000000025.jpg";
	}
	
	



	const char *proto = "E:\\DeepLearning\\myColor\\deploy_fcn.prototxt";
	const char *model = "E:\\DeepLearning\\myColor\\snapshot\\colornet_iter_15000.caffemodel";
	

	Phase phase = TEST;
	Caffe::set_mode(Caffe::CPU); 
	
	
	boost::shared_ptr<Net<float> > net(new caffe::Net<float>(proto, phase));
	net->CopyTrainedLayersFrom(model);

	if (imgname != NULL)
	{
		Mat img = imread(imgname);
		if (img.data == NULL)
		{
			return -1;
		}
		classify(net, img);
	}

	int i;
	for (i = 1; i < argc; ++i)
	{
		Mat img = imread(argv[i]);
		if (img.data == NULL)
		{
			continue;
		}

		classify(net, img);
	}




	return 0;

}
#else

#endif


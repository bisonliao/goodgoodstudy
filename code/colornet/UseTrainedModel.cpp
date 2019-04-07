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

#define COLOR_NUM 64

int demulaw(int a /* -128 to 127*/, int u = 255)
{
	int neg = 0;
	if (a < 0)
	{
		a = -a;
		neg = 1;
	}
	float y = a / 128.0 * log((float)u + 1);
	y = (exp((double)y) - 1) * 128 / u;
	if (neg)
	{
		y = -y;
	}
	return y;
}
int from_yuv(const Mat & src, Mat & dst)
{
	if (src.channels() != 2)
	{
		fprintf(stderr, "my yuv image must have 2 channels!\n");
		return -1;
	}
	Mat tmp(src.rows, src.cols, CV_8UC3);
	int height, width;
	for (height = 0; height < src.rows; ++height)
	{
		for (width = 0; width < src.cols; ++width)
		{
			cv::Point3_<uchar>* p2 = tmp.ptr<cv::Point3_<uchar> >(height, width);
			const cv::Point_<uchar> *p = src.ptr<cv::Point_<uchar> >(height, width);
			p2->x = p->x; //L



#if (COLOR_NUM == 256)
			int y = (p->y & 0xf0) >> 4;
			int z = (p->y & 0x0f);
			y -= 8;
			z -= 8;
			y *= 16;
			z *= 16;

#elif (COLOR_NUM == 64)

			int y = (p->y & 0x38) >> 3;
			int z = (p->y & 0x07);
			y -= 4;
			z -= 4;
			y *= 32;
			z *= 32;


#elif (COLOR_NUM == 16)

			int y = (p->y & 0x0c) >> 2;
			int z = (p->y & 0x03);
			y -= 2;
			z -= 2;
			y *= 64;
			z *= 64;


#else
#error invalid color number!
#endif
			y = demulaw(y, 10);
			z = demulaw(z, 10);
			y += 128;
			z += 128;
			p2->y = y;
			p2->z = z;
		}
	}
	//printf("ch:%d\n", tmp.channels());
	cvtColor(tmp, dst, CV_YUV2BGR);
	return 0;
}
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

	const int output_size = input_size / 2; // deploy_color.prototxt
	//const int output_size = input_size; // deploy_fcn.prototxt
	
	Mat result(output_size, output_size, CV_8UC2, Scalar(0));
	const int CLASS_NUM = 64;
	const float *blob_ptr = (const float *)blob->cpu_data();
	for (height = 0; height < output_size; height++)
	{
		for (width = 0; width < output_size; width++)
		{
			float max = blob_ptr[0 * (output_size*output_size) + height * output_size + width];
			int max_idx = 0;
			for (class_idx = 1; class_idx < CLASS_NUM; ++class_idx)
			{
				int offset = class_idx * (output_size*output_size) + height * output_size + width;
				if (blob_ptr[offset] > max)
				{
					max = blob_ptr[offset];
					max_idx = class_idx;
				}
				
				if (width == 0 && height < 10)
				{
					printf("class:%d,prob:%f\n", class_idx, blob_ptr[offset]);
				}
				
			}
			if (width == 0 && height < 10)
			{
				printf("max:%d\n", max_idx);
			}
			cv::Point_<uchar>* p = result.ptr<cv::Point_<uchar> >(height, width);
			const cv::Point3_<uchar>* p2 = yuvMat.ptr<cv::Point3_<uchar> >(height * input_size / output_size, width* input_size / output_size);
			p->x = p2->x;
			p->y = max_idx;
			
			
		}
	
	}

	Mat img2, toShow(0, resizedMat.cols, CV_8UC3);

	toShow.push_back(resizedMat);

	from_yuv(result, img2);
	if (output_size != input_size)
	{
		Mat tmp = img2.clone();
		resize(tmp, img2, Size(input_size, input_size), 0, 0, INTER_CUBIC);
	}
	
	printf("type:%d, %d,%d\n", img.type(), img2.type(), toShow.type());
	printf("cols:%d, %d,%d\n", img.cols, img2.cols, toShow.cols);
	toShow.push_back(img2);

	namedWindow("show");
	cv::imshow("show", toShow);
	cv::waitKey(0);
	
	return;


}


#if 1


int main(int argc, char **argv) {

	const char * imgname = NULL;
	if (argc < 2)
	{
		fprintf(stderr, "usage: %s imagefilename\n", argv[0]);
		imgname = "e:\\DeepLearning\\data\\coco\\train2014\\COCO_train2014_000000000025.jpg";
	}
	
	

#if 0

	const char *proto = "E:\\DeepLearning\\myColor\\deploy_fcn.prototxt";
	const char *model = "E:\\DeepLearning\\myColor\\snapshot\\colorfcn_iter_55000.caffemodel";
#else
	const char *proto = "E:\\DeepLearning\\myColor\\deploy_color.prototxt";
	const char *model = "E:\\DeepLearning\\myColor\\snapshot\\colornet_iter_15000.caffemodel";
#endif

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


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

#include"leveldb/db.h"

using namespace std;
using namespace leveldb;


using namespace caffe; 
using std::string;
using namespace cv;
 

const int input_size = 28;




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


  
int classify(boost::shared_ptr<Net<float> > net, const Mat & img1, const Mat & img2)
{
	static float data_input[2][input_size][input_size];
	Mat image1, image2;
	
	if (img1.channels() != 1)
	{
		cvtColor(img1, image1, CV_BGR2GRAY);
	}
	else
	{
		image1 = img1;
	}
	if (img2.channels() != 1)
	{
		cvtColor(img2, image2, CV_BGR2GRAY);
	}
	else
	{
		image2 = img2;
	}
	Mat resized = image1.clone();
	cv::resize(resized, image1, cv::Size(input_size, input_size));
	resized = image2.clone();
	cv::resize(resized, image2, cv::Size(input_size, input_size));

	


	int width, height, chn;


	for (height = 0; height < input_size; ++height)
	{
		for (width = 0; width < input_size; ++width)
		{
			uchar v = image1.at<uchar>(height, width);
			data_input[0][height][width] = v / 256.0;

			v = image2.at<uchar>(height, width);
			data_input[1][height][width] = v / 256.0;

		}
	}

	Blob<float>* input_blobs = net->input_blobs()[0];
	//printf("inpupt blob x count:%d\n", input_blobs->count());
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

	int index = get_blob_index(net, "loss");
	boost::shared_ptr<Blob<float> > blob = net->blobs()[index];
	//printf("output blob index:%d,  y count:%d\n", index, blob->count());
	const float *blob_ptr = (const float *)blob->cpu_data();
	float y = *blob_ptr;
	
	if (y > 0.2)
	{
		return 0;
	}
	else
	{
		return 1;
	}
	



}

const char * basename(const char * path)
{
	const char * p = strrchr(path, '\\');
	if (p == NULL)
	{
		return path;
	}
	else
	{
		return p + 1;
	}
}

int test(boost::shared_ptr<Net<float> > net, const char * dbpath)
{
	DB *db;
	Options options;
	options.create_if_missing = true;
	
	Status s = DB::Open(options, dbpath, &db);
	if (!s.ok()) 
	{
		printf("%s", s.ToString().c_str());
		return -1;
	}
	int succ = 0, cnt = 0;
	Iterator *it = db->NewIterator(ReadOptions());
	for (it->SeekToFirst(); it->Valid(); it->Next()) {
		Datum d;
		d.ParseFromString(it->value().ToString());
		std::string bytes = d.data();
		if (bytes.length() != (input_size * input_size * 2))
		{
			printf("data number mismatch!\n");
			continue;
		}
		Mat img1(input_size, input_size, CV_8UC1);
		Mat img2(input_size, input_size, CV_8UC1);
		int h, w;
		for (h = 0; h < input_size; ++h)
		{
			for (w = 0; w < input_size; ++w)
			{
				img1.at<uchar>(h, w) = bytes.at(h*input_size + w);
				img2.at<uchar>(h, w) = bytes.at(input_size*input_size + h*input_size + w);
			}
		}
		int label = d.label();
		int y = classify(net, img1, img2);
		if (y == label)
		{
			succ++;
			/*
			printf("ok, %d\n", y);

			Mat toShow = img1.clone();
			toShow.push_back(img2);
			namedWindow("mismatch");
			imshow("mismatch", toShow);
			waitKey(0);
			*/

		}
		else
		{
			/*
			
			printf("%d mismatch %d\n", label, y);
			Mat toShow = img1.clone();
			toShow.push_back(img2);
			namedWindow("mismatch");
			imshow("mismatch", toShow);
			waitKey(0);
			*/
			
			
		}
		cnt++;

		if (cnt > 10000)
		{
			break;
		}

	
		
	}
	printf("accuracy:%f\n", succ / (cnt*1.0));
	return 0;

}


int main(int argc, char **argv) {



	const char *proto = "E:\\DeepLearning\\siamese\\siamese\\mnist_siamese_deploy.prototxt";
	const char *model = "E:\\DeepLearning\\siamese\\siamese\\snapshot\\mnist_siamese_iter_50000.caffemodel";


	Phase phase = TEST;
	Caffe::set_mode(Caffe::GPU); 
	
	
	boost::shared_ptr<Net<float> > net(new caffe::Net<float>(proto, phase));
	net->CopyTrainedLayersFrom(model);

	if (argc < 2)
	{

		test(net, "E:\\DeepLearning\\siamese\\siamese\\examples\\siamese\\mnist_siamese_test_leveldb");
	}
	else
	{
		int i;
		for (i = 1; i+1 < argc; i=i+2)
		{
			Mat img1 = imread(argv[i]);
			Mat img2 = imread(argv[i + 1]);
			if (img1.data == NULL|| img2.data == NULL)
			{
				continue;
			}
			
			printf("%s vs %s:\n", basename(argv[i]), basename(argv[i + 1]));
			printf("y'= %d\n", classify(net, img1, img2));
		}
	}




	return 0;

}



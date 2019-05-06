

#include "stdafx.h"

#include "layer_reg.h"
#include <caffe/caffe.hpp>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iostream>
#include <fstream>
#include <io.h>


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

//
// This script converts the MNIST dataset to the leveldb format used
// by caffe to train siamese network.
// Usage:
//    convert_mnist_data input_image_file input_label_file output_db_file
// The MNIST dataset could be downloaded at
//    http://yann.lecun.com/exdb/mnist/
#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"
#include "caffe/util/math_functions.hpp"


#include "leveldb/db.h"

using namespace std;

const int g_rows = 256;
const int g_cols = 256;



void read_image(const string & filename, unsigned char* pixels) {
	cv::Mat mat = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (mat.data == NULL || mat.channels() != 1 || mat.rows != g_rows || mat.cols != g_cols)
	{
		fprintf(stderr, "failed to read images\n");
		return;
	}

	/*		cv::Mat toShow = mat.clone();

		cv::namedWindow("n");
		cv::imshow("n", toShow);
		cv::waitKey(0);
	*/
	int i, j;
	for (i = 0; i < g_rows; i++)
	{
		for (j = 0; j < g_cols; j++)
		{
			pixels[i*g_cols + j] = mat.at<uchar>(i, j);
		}
	}
}

void convert_dataset(const string &dir, const string & pattern,	const char* db_filename) {


	intptr_t handle;
	_finddata_t findData;

	vector<pair<string, string>> images;



	handle = _findfirst((dir + pattern).c_str(), &findData);
	if (handle == -1)        // 检查是否成功
	{
		fprintf(stderr, "_findfirst() failed!\n");
		return;
	}

	do
	{

		if (findData.attrib & _A_SUBDIR)
		{
			continue;
		}
		char filename[1024];
		_snprintf_s(filename, sizeof(filename), "%s\\%s", dir.c_str(), findData.name);
		char person_id[100];
		const char * pos = strchr(findData.name, '.');
		if (pos == NULL)
		{
			continue;
		}
		strncpy_s(person_id, sizeof(person_id), findData.name, pos - findData.name);
		images.push_back(pair<string, string>(person_id, filename));
	


	} while (_findnext(handle, &findData) == 0);


	_findclose(handle);

	
	int num_items = images.size();
	

	// Open leveldb
	leveldb::DB* db;
	leveldb::Options options;
	options.create_if_missing = true;
	options.error_if_exists = true;
	leveldb::Status status = leveldb::DB::Open(
		options, db_filename, &db);
	CHECK(status.ok()) << "Failed to open leveldb " << db_filename
		<< ". Is it already existing?";


	unsigned char* pixels = new unsigned char[2 * g_rows * g_cols];
	std::string value;

	caffe::Datum datum;
	datum.set_channels(2);  // one channel for each image in the pair
	datum.set_height(g_rows);
	datum.set_width(g_cols);
	cout << "A total of " << num_items << " items."<<endl;
	cout << "Rows: " << g_rows << " Cols: " << g_cols<<endl;
	int positive_num = 0;
	int negtive_num = 0;
	vector<pair<string, string>>::iterator it1, it2;
	
	int cnt = 0;
	for (it1 =  images.begin(); it1 != images.end(); it1++) 
	{

		for (it2 =  images.begin(); it2 != images.end(); it2++)
		{
			read_image(it1->second, pixels);
			read_image(it2->second, pixels+g_rows*g_cols);
			string label_i = it1->first;
			string label_j = it2->first;

           if (label_i == label_j)
			{
				positive_num++;
			}
			else
			{
				/*
				if (positive_num < (negtive_num/3))
				{
					// unbalanced
					continue;
				}
				*/
				negtive_num++;
			}
			cnt++;
			datum.set_data(pixels, 2 * g_rows*g_cols);
			if (label_i == label_j) {
				datum.set_label(1);
			}
			else {
				datum.set_label(0);
			}
			datum.SerializeToString(&value);
			std::string key_str = caffe::format_int(cnt, 8);
			db->Put(leveldb::WriteOptions(), key_str, value);
		}

		
	}
	cout << "positive:" << positive_num << " negtive:" << negtive_num << endl;

	delete db;
	delete[] pixels;
}

int main(int argc, char** argv) {
	
	google::InitGoogleLogging(argv[0]);
	convert_dataset("E:\\DeepLearning\\data\\jaffe\\train\\", "*.tiff", "E:\\DeepLearning\\siamese\\siamese\\faces\\train_leveldb");
	convert_dataset("E:\\DeepLearning\\data\\jaffe\\test\\", "*.tiff", "E:\\DeepLearning\\siamese\\siamese\\faces\\test_leveldb");

	return 0;
}

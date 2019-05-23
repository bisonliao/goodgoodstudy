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


#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include "caffe.pb.h"

using namespace std;
using namespace leveldb;


using namespace caffe; 
using std::string;
using namespace cv;
 

const int input_size = 300;




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


  
int classify(boost::shared_ptr<Net<float> > net, const Mat & img1, Mat & img2)
{
	static float data_input[3][input_size][input_size];
	Mat image1, image2;
	
	if (img1.channels() != 3)
	{
		cv::Mat tmp = img1.clone();
		cv::cvtColor(tmp, img1, cv::COLOR_GRAY2BGR);
	}

	Mat resized = img1.clone();
	cv::resize(resized, image1, cv::Size(input_size, input_size));
	

	int width, height, chn;


	for (height = 0; height < input_size; ++height)
	{
		for (width = 0; width < input_size; ++width)
		{
			const cv::Point3d *p = image1.ptr<cv::Point3d>(height, width);
			
			data_input[0][height][width] = p->x - 104.0;
			data_input[1][height][width] = p->y - 117.0;
			data_input[2][height][width] = p->z - 123.0;

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
	img2 = image1.clone();

	net->Forward();
	
	Blob<float>* result_blob = net->output_blobs()[0];
	const float* result = result_blob->cpu_data(); // shape is [ 1, 1, num_det, 7], 输出每行7个数，分别表示图片id、标签、置信度以及4个坐标
	const int num_det = result_blob->height();
	printf("detect %d bbox\n", num_det);
	for (int k = 0; k < num_det; ++k) {
		if (result[0] == -1) {
			// Skip invalid detection.
			result += 7;
			continue;
         }
		float xmin = *(result + 3);
		float ymin = *(result + 4);
		float xmax = *(result + 5);
		float ymax = *(result + 6);

		int x = xmin * input_size;
		int y = ymin * input_size;
		int w = (xmax - xmin)*input_size;
		int h = (ymax - ymin)*input_size;
		cv::Rect rec(x, y, w, h);
		cv::rectangle(img2, rec, cv::Scalar(0, 0, 255));
		
		result += 7;
    }
	return 0;


	
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
int flag = 0;

int test(boost::shared_ptr<Net<float> > net, const char * dbpath)
{
	MDB_env *env;
	MDB_txn * txn;
	MDB_dbi dbi;
	MDB_cursor * cursor;
	int iRet;
	if (iRet = mdb_env_create(&env))
	{
		fprintf(stderr, "mdb_env_create() failed, iRet=%d\n", iRet);
		return -1;
	}
	if (iRet = mdb_env_open(env, dbpath, MDB_NOSUBDIR | MDB_RDONLY, 0644))
	{
		fprintf(stderr, "mdb_env_open() failed, iRet=%d\n", iRet);
		return -1;
	}

	if (iRet = mdb_txn_begin(env, NULL, MDB_RDONLY, &txn))
	{
		fprintf(stderr, "mdb_txn_begin() failed, iRet=%d\n", iRet);
		return -1;
	}

	if (iRet = mdb_dbi_open(txn, NULL, 0, &dbi))
	{
		fprintf(stderr, "mdb_dbi_open() failed, iRet=%d\n", iRet);
		return -1;
	}
	if (iRet = mdb_cursor_open(txn, dbi, &cursor))
	{
		fprintf(stderr, "mdb_cursor_open() failed, iRet=%d\n", iRet);
		return -1;
	}

	MDB_val key, data;
	MDB_cursor_op op = MDB_FIRST;
	int i;
	cv::namedWindow("check lmdb", cv::WINDOW_AUTOSIZE);


	for (i = 0; i < 10000; i++)
	{
		if (i > 0) { op = MDB_NEXT; }
		if (iRet = mdb_cursor_get(cursor, &key, &data, op))
		{
			fprintf(stderr, "mdb_cursor_get() failed, iRet=%d\n", iRet);
			break;
		}
		else
		{
			string k((const char*)key.mv_data, (size_t)key.mv_size);
			string v((const char*)data.mv_data, (size_t)data.mv_size);
			AnnotatedDatum record;
			record.ParseFromString(v);
			int width, height, channel;

			
			const string & datastr = record.datum().data();
			const uchar * data_ptr = (const uchar*)(datastr.c_str());
			

			std::vector<uchar> vv(datastr.begin(), datastr.end());

			cv::Mat img = imdecode(vv, CV_LOAD_IMAGE_COLOR);
			printf("c:%d,h:%d,w:%d, \n",img.channels(), img.rows, img.cols);
			cv::Mat result;
			classify(net, img, result);
			imshow("check lmdb", result);
			waitKey(0);
		}
	}

}


int main(int argc, char **argv) {



	const char *proto = "E:\\DeepLearning\\ssd\\deploy2.prototxt";
	const char *model = "E:\\DeepLearning\\ssd\\snapshot\\ssd_iter_20000.caffemodel";


	Phase phase = TEST;
	Caffe::set_mode(Caffe::GPU); 
	
	
	boost::shared_ptr<Net<float> > net(new caffe::Net<float>(proto, phase));
	net->CopyTrainedLayersFrom(model);

	if (argc < 2)
	{
		test(net, "E:\\DeepLearning\\ssd\\test_lmdb\\data.mdb");
	}
	else
	{
		int i;
		cv::namedWindow("check lmdb", cv::WINDOW_AUTOSIZE);
		for (i = 1; i < argc; i++)
		{
			Mat img1 = imread(argv[i]);
			
			if (img1.data == NULL)
			{
				continue;
			}
			
			cv::Mat result;
			classify(net, img1, result);
			imshow("check lmdb", result);
			waitKey(0);
		}
	}




	return 0;

}



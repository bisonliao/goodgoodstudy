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


string g_labels[21] = {
	"background",
	"aeroplane",
	"bicycle",
	"bird",
	"boat",
	"bottle",
	"bus",
	"car",
	"cat",
	"chair",
	"cow",
	"diningtable",
	"dog",
	"horse",
	"motorbike",
	"person",
	"pottedplant",
	"sheep",
	"sofa",
	"train",
	"tvmonitor"

};


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

void drawText(Mat & img, const std::string &txt, cv::Point & org)
{
	std::string text = txt;
	int font_face = cv::FONT_HERSHEY_COMPLEX;
	double font_scale = 0.5;
	int thickness = 1;
	int baseline;
	//获取文本框的长宽
	cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

	//将文本框居中绘制
	cv::putText(img, text, org, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, 0);
}
  
int classify(boost::shared_ptr<Net<float> > net, const Mat & img1, Mat & img2)
{
	static float data_input[3][input_size][input_size];
	
	
	
	if (img1.channels() != 3)
	{
		fprintf(stderr, "channel number is not 3!\n");
		return -1;
	}
	Mat resized;
	cv::resize(img1, resized, cv::Size(input_size, input_size));
	
	

	int width, height;


	for (height = 0; height < input_size; ++height)
	{
		for (width = 0; width < input_size; ++width)
		{
			
			//为什么下面这种方式取不到有效值呢？
		/*
			cv::Point3f *p = resized.ptr<cv::Point3f>(height, width);
			if (height == 0 && width == 0)
			{
				printf("%f,%f,%f\n", p->x, p->y, p->z);
			}
		*/	
			/*
			//这样可以
			cv::Point3_<uchar> * p = resized.ptr<cv::Point3_<uchar>>(height, width);
			if (height == 100 && width == 100)
			{
				printf("%d,%d,%d\n", p->x, p->y, p->z);
			}
			*/

			cv::Vec3b *p2 = resized.ptr<cv::Vec3b>(height, width);
			data_input[0][height][width] = (*p2)[0] - 104.0;
			data_input[1][height][width] = (*p2)[1] - 117.0;
			data_input[2][height][width] = (*p2)[2] - 123.0;
			

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
	img2 = resized.clone();

	net->Forward();
	
	Blob<float>* result_blob = net->output_blobs()[0];
	const float* result = result_blob->cpu_data(); // shape is [ 1, 1, num_det, 7], 输出每行7个数，分别表示图片id、标签、置信度以及4个坐标
	const int num_det = result_blob->height();
	printf("detect %d bbox\n", num_det);
	printf("blob len:%d\n", result_blob->count());
	int cnt = 0;
	for (int k = 0; k < num_det; ++k) {
		if (result[0] == -1 || result[2] < 0.8) {
			// Skip invalid detection.
			result += 7;
			continue;
         }
		cnt++;
		const string& labelstr = g_labels[((int)(result[1])) % 21];
		printf("conf:%.2f,label:%s, position:%.2f %.2f %.2f %.2f\n ", result[2], labelstr.c_str(), result[3], result[4], result[5], result[6]);
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
		drawText(img2, labelstr, cv::Point(x, y));
		
		result += 7;
    }
	printf("\n");
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
			

			
			const string & datastr = record.datum().data();
			const uchar * data_ptr = (const uchar*)(datastr.c_str());
			
			
			std::vector<uchar> vv(datastr.begin(), datastr.end());

			cv::Mat img = imdecode(vv, CV_LOAD_IMAGE_COLOR);
			//printf("c:%d,h:%d,w:%d, \n",img.channels(), img.rows, img.cols);
			cv::Mat result;
			classify(net, img, result);
			imshow("check lmdb", result);
			waitKey(0);
		}
	}

}


int main(int argc, char **argv) {


#if 1
	const char *proto = "E:\\DeepLearning\\ssd\\deploy.prototxt";
	const char *model = "E:\\DeepLearning\\ssd\\snapshot\\ssd_iter_88000.caffemodel";
#else
	const char *proto = "E:\\DeepLearning\\ssd\\official_model\\models\\VGGNet\\VOC0712\\SSD_300x300\\deploy.prototxt";
	const char *model = "E:\\DeepLearning\\ssd\\official_model\\models\\VGGNet\\VOC0712\\SSD_300x300\\VGG_VOC0712_SSD_300x300_iter_120000.caffemodel";
#endif


	Phase phase = TEST;
	Caffe::set_mode(Caffe::GPU); 
	
	
	boost::shared_ptr<Net<float> > net(new caffe::Net<float>(proto, phase));
	
	net->CopyTrainedLayersFrom(model);

	if (argc < 2)
	{
		test(net, "E:\\DeepLearning\\ssd\\train_lmdb\\data.mdb");
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



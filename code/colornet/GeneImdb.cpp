// GeneImdb.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <time.h>
#include <gflags/gflags.h>
#include <google/protobuf/text_format.h>

#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>

#include <stdint.h>
#include <sys/stat.h> 

#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <io.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


#include "caffe.pb.h"

using namespace cv;
using namespace std;

//#include "caffe/util/format.hpp"

//#define TRAIN (1)

#ifdef TRAIN
const uint16_t INPUT_SIZE = 224;
#else
const uint16_t INPUT_SIZE = 224;
#endif


using namespace caffe;  // NOLINT(build/namespaces)

using std::string;

void rmReturn(char * line)
{
	if (line == NULL) { return; }

	int len = strlen(line);
	if (len >= 2 && line[len - 1] == '\n' && line[len - 2] == '\r')
	{
		line[len - 2] = '\0';
	}
	else if (len >= 1 && line[len - 1] == '\n')
	{
		line[len - 1] = '\0';
	}
}

int from_yuv(const Mat & src, Mat & dst);

void check_lmdb(const char * db_path)
{
	MDB_env *env;
	MDB_txn * txn;
	MDB_dbi dbi;
	MDB_cursor * cursor;
	int iRet;
	if (iRet = mdb_env_create(&env))
	{
		fprintf(stderr, "mdb_env_create() failed, iRet=%d\n", iRet);
		return;
	}
	if (iRet = mdb_env_open(env, db_path, MDB_NOSUBDIR|MDB_RDONLY, 0644))
	{
		fprintf(stderr, "mdb_env_open() failed, iRet=%d\n", iRet);
		return;
	}

	if (iRet = mdb_txn_begin(env, NULL, MDB_RDONLY, &txn))
	{
		fprintf(stderr, "mdb_txn_begin() failed, iRet=%d\n", iRet);
		return;
	}

	if (iRet = mdb_dbi_open(txn, NULL, 0, &dbi))
	{
		fprintf(stderr, "mdb_dbi_open() failed, iRet=%d\n", iRet);
		return;
	}
	if (iRet = mdb_cursor_open(txn, dbi, &cursor))
	{
		fprintf(stderr, "mdb_cursor_open() failed, iRet=%d\n", iRet);
		return;
	}

	MDB_val key, data;
	MDB_cursor_op op = MDB_FIRST;
	int i;
	cv::namedWindow("check lmdb", cv::WINDOW_AUTOSIZE);

	for (i = 0; i < 10; i++)
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
			Datum record;
			record.ParseFromString(v);
			
			printf("%d,%d,%d\n", record.channels(), record.width(), record.height());

			cv::Mat img = cv::Mat::ones(cv::Size(INPUT_SIZE, INPUT_SIZE), CV_8UC2);
			int height, width;
			for (height = 0; height < INPUT_SIZE; ++height)
			{
				for (width = 0; width < INPUT_SIZE; ++width)
				{
					cv::Point_<uchar>* p = img.ptr<cv::Point_<uchar> >(height, width);
					p->x = record.data()[0 * INPUT_SIZE * INPUT_SIZE + height * INPUT_SIZE + width];
					p->y = record.data()[1 * INPUT_SIZE * INPUT_SIZE + height * INPUT_SIZE + width];
				//	printf("%u ", p -> y);
				}
				//printf("\n");
			}
			//printf("\n");
			Mat img2;
			from_yuv(img, img2);
			cv::imshow("check lmdb", img2);
			cv::waitKey(0);
		}
		
	}
	mdb_cursor_close(cursor);
	mdb_dbi_close(env, dbi);
	mdb_env_close(env);
}

void salt(cv::Mat& image, int n) //加点噪声
{
	for (int k = 0; k<n; k++) {
		int i = rand() % image.cols;
		int j = rand() % image.rows;

		if (image.channels() == 1) {
			image.at<uchar>(j, i) = 255;
		}
		else {
			image.at<cv::Vec3b>(j, i)[0] = 255;
			image.at<cv::Vec3b>(j, i)[1] = 255;
			image.at<cv::Vec3b>(j, i)[2] = 255;
		}
	}
}
int augment_image(const Mat & image,  vector<Mat> & image_list)
{
	Mat dst;
#if 0
	//椒盐噪声
	image.copyTo(dst);
	salt(dst, dst.cols * 3);
	image_list.push_back(dst.clone());

	

	// 左右翻转
	flip(image, dst, 1);
	image_list.push_back(dst.clone());
	

	//模糊
	image.copyTo(dst);
	blur(image, dst, Size(3, 3));
	image_list.push_back(dst.clone());
#endif
	image_list.push_back(image.clone());

	return 0;
}
int to_yuv(const Mat & src, Mat & dst)
{
	cvtColor(src, dst, CV_BGR2YUV);
	Mat result(src.rows, src.cols, CV_8UC2);
	int height, width;

		for (height = 0; height < dst.rows; ++height)
		{
			for (width = 0; width < dst.cols; ++width)
			{
				const cv::Point3_<uchar>* p = dst.ptr<cv::Point3_<uchar> >(height, width);
				cv::Point_<uchar> *p2 = result.ptr<cv::Point_<uchar> >(height, width);
				p2->x = p -> x; //Y
				p2->y = (p->y & 0xf0) | ( (p->z & 0xf0) >> 4); // compress u / v togethor
			}
		}
	
	dst = result.clone();
	return 0;
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
			p2->x = p->x; //Y
			p2->y = p->y & 0xf0; //u
			p2->z = (p->y & 0x0f) << 4;//v
		}
	}
	//printf("ch:%d\n", tmp.channels());
	cvtColor(tmp, dst, CV_YUV2BGR);
	return 0;
}
void convert_dataset(const string & dir, const string & pattern	, const string & db_path) {
	
	MDB_env *env;
	MDB_txn * txn;
	MDB_dbi dbi;
	int iRet;
	if (iRet = mdb_env_create(&env))
	{
		fprintf(stderr, "mdb_env_create() failed, iRet=%d\n", iRet);
		return;
	}
	
	if (iRet = mdb_env_set_mapsize(env, 42949672960ull))
	{
		fprintf(stderr, "mdb_env_set_mapsize() failed, iRet=%d\n", iRet);
		return;
	}
	if (iRet = mdb_env_open(env, db_path.c_str(), MDB_NOSUBDIR, 0644))
	{
		fprintf(stderr, "mdb_env_open() failed, iRet=%d\n", iRet);
		return;
	}
	
	
	if (iRet = mdb_txn_begin(env, NULL, 0, &txn))
	{
		fprintf(stderr, "mdb_txn_begin() failed, iRet=%d\n", iRet);
		return;
	}
	
	if (iRet = mdb_dbi_open(txn, NULL, 0, &dbi))
	{
		fprintf(stderr, "mdb_dbi_open() failed, iRet=%d\n", iRet);
		return;
	}

	MDB_val key, data;
	
	
	intptr_t handle;
	_finddata_t findData;
	int count = 0;
	int item_id = 1;


	string value;
	Datum datum;
	

	handle = _findfirst((dir+pattern).c_str(), &findData);
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
			cv::Mat img = cv::imread(filename);
			if (img.data == NULL || img.channels() != 3)
			{
				continue;
			}
			cv::Mat inputMat;
			cv::resize(img, inputMat, cv::Size(INPUT_SIZE, INPUT_SIZE), 0, 0, cv::INTER_CUBIC);

			vector<Mat> img_list;
			augment_image(inputMat, img_list);



			static uchar data_input[2][INPUT_SIZE][INPUT_SIZE];
			
			int height, width, i;
			for (i = 0; i < img_list.size(); ++i)
			{
				Mat yuvImg;
				to_yuv(img_list[i], yuvImg);
				for (height = 0; height < INPUT_SIZE; ++height)
				{
					for (width = 0; width < INPUT_SIZE; ++width)
					{
						const cv::Point_<uchar>* p = yuvImg.ptr<cv::Point_<uchar> >(height, width);
						data_input[0][height][width] = p->x;//Y
						data_input[1][height][width] = p->y;//U,V

					}
				}

				datum.set_data(data_input, sizeof(data_input));

				datum.set_height(INPUT_SIZE);
				datum.set_width(INPUT_SIZE);
				datum.set_channels(2);

				char skey[10];
				snprintf(skey, sizeof(skey), "%08d", item_id + 1);
				string key_str = skey;

				datum.SerializeToString(&value);

				key.mv_data = (void*)key_str.c_str();
				key.mv_size = key_str.size();
				data.mv_data = (void*)value.c_str();
				data.mv_size = value.size();


				if (iRet = mdb_put(txn, dbi, &key, &data, 0))
				{
					fprintf(stderr, "mdb_put returns %d\n", iRet);
					goto end;
				}
				item_id++;
				if ((item_id % 100) == 97)
				{

					mdb_txn_commit(txn);
					mdb_txn_begin(env, NULL, 0, &txn);
					printf("put %d images in db\n", item_id);
				}
			}
	
	} while (_findnext(handle, &findData)==0);
end:
	mdb_txn_commit(txn);
	mdb_dbi_close(env, dbi);
	mdb_env_close(env);
	_findclose(handle);


}
#include "errno.h"
int main()
{

#if 0
#ifdef TRAIN
	convert_dataset("E:\\DeepLearning\\data\\coco\\train2014\\", "*.jpg", "E:\\DeepLearning\\myColor\\train_data\\data");
#else
	convert_dataset("E:\\DeepLearning\\data\\coco\\val\\", "*.jpg", "E:\\DeepLearning\\myColor\\val_data\\data");
#endif

#endif


	check_lmdb("E:\\DeepLearning\\myColor\\train_data\\data.mdb");
#if 0
	Mat src, dst,dst2;
	src = imread("E:\\DeepLearning\\data\\coco\\train2014\\COCO_train2014_000000000030.jpg");
	if (src.data == NULL)
	{
		printf("invalid image!\n");
		return -1;
	}
	to_yuv(src, dst);

	from_yuv(dst, dst2);
	imshow("hello", dst2);
	waitKey(0);
#endif
    return 0;
}


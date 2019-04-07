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

#define COLOR_NUM 64

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

	map<int, int> stat;

	for (i = 0; i < 64; ++i)
	{
		stat.insert(std::pair<int, int>(i, 0));
	}

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
			Datum record;
			record.ParseFromString(v);
			
			printf("%d,%d,%d\n", record.channels(), record.width(), record.height());

			cv::Mat img = cv::Mat::ones(cv::Size(INPUT_SIZE, INPUT_SIZE), CV_8UC2);
			int height, width;
			int max = 0;
			
			for (height = 0; height < INPUT_SIZE; ++height)
			{
				for (width = 0; width < INPUT_SIZE; ++width)
				{
					cv::Point_<uchar>* p = img.ptr<cv::Point_<uchar> >(height, width);
					p->x = record.data()[0 * INPUT_SIZE * INPUT_SIZE + height * INPUT_SIZE + width];
					p->y = record.data()[1 * INPUT_SIZE * INPUT_SIZE + height * INPUT_SIZE + width];
					if (p->y > max) { max = p->y; }
					//printf("%d ", p->y);
					stat[p->y]++;
				
				}
				
			}
			for (int j = 0; j < 64; ++j)
			{
				printf("%d, %f\n", j, stat.find(j)->second / (float)(INPUT_SIZE*INPUT_SIZE*(i+1)));
			}
			printf("\n");
			printf("max = %d\n", max);
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

int mulaw(int a /* -128 to 127*/, int u=255)
{
	int neg = 0;
	if (a < 0)
	{
		a = -a;
		neg = 1;
	}
	float y = 128 * ::log((float)(1 + u * a / 128.0)) / ::log((float)u + 1);
	if (neg)
	{
		y = -y;
	}
	return y;
}
int demulaw(int a /* -128 to 127*/, int u = 255)
{
	int neg = 0;
	if (a < 0)
	{
		a = -a;
		neg = 1;
	}
	float y = a/128.0 * log((float)u + 1);
	y = (exp((double)y) - 1) * 128 / u;
	if (neg)
	{
		y = -y;
	}
	return y;
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
				p2->x = p->x; //L
				int y = p->y - 128;
				int z = p->z - 128;

				y = mulaw(y, 10);
				z = mulaw(z, 10);

#if (COLOR_NUM == 256)
				y = y / 16;
				z = z / 16;
				y = y + 8;
				z = z + 8;
				uchar yy = y;
				uchar zz = z;
				p2->y = (yy << 4) | zz;
#elif (COLOR_NUM == 64)
				y = y / 32;
				z = z / 32;
				y = y + 4;
				z = z + 4;
				uchar yy = y;
				uchar zz = z;
				p2->y = (yy << 3) | zz;
#elif (COLOR_NUM == 16)
				y = y / 64;
				z = z / 64;
				y = y + 2;
				z = z + 2;
				uchar yy = y;
				uchar zz = z;
				p2->y = (yy << 2) | zz;

#else
#error invalid color number!
#endif
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
#if 0
int to_Lab(const Mat & src, Mat & dst)
{
	cvtColor(src, dst, CV_BGR2Lab);
	printf("type:%d\n", dst.type()==CV_8UC3);
	Mat result(src.rows, src.cols, CV_8UC2);
	int height, width;

	for (height = 0; height < dst.rows; ++height)
	{
		for (width = 0; width < dst.cols; ++width)
		{
			const cv::Point3_<uchar>* p = dst.ptr<cv::Point3_<uchar> >(height, width);
			cv::Point_<uchar> *p2 = result.ptr<cv::Point_<uchar> >(height, width);
			p2->x = p->x; //L
			int y = p->y - 128;
			int z = p->z - 128;
	
#if (COLOR_NUM == 256)
			y = y / 16;
			z = z / 16;
			y = y + 8;
			z = z + 8;
			uchar yy = y;
			uchar zz = z;
			p2->y = (yy << 4) | zz;
#elif (COLOR_NUM == 64)
			y = y / 32;
			z = z / 32;
			y = y + 4;
			z = z + 4;
			uchar yy = y;
			uchar zz = z;
			p2->y = (yy << 3) | zz;
#elif (COLOR_NUM == 16)
			y = y / 64;
			z = z / 64;
			y = y + 2;
			z = z + 2;
			uchar yy = y;
			uchar zz = z;
			p2->y = (yy << 2) | zz;
														
#else
#error invalid color number!
#endif

		}
	}

	dst = result.clone();
	return 0;
}
int from_Lab(const Mat & src, Mat & dst)
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
			y += 128;
			z += 128;
			p2->y = y;
			p2->z = z ;
		
#elif (COLOR_NUM == 64)

			int y = (p->y & 0x38) >> 3;
			int z = (p->y & 0x07);
			y -= 4;
			z -= 4;
			y *= 32;
			z *= 32;
			y += 128;
			z += 128;
			p2->y = y;
			p2->z = z;

#elif (COLOR_NUM == 16)

			int y = (p->y & 0x0c) >> 2; 
			int z = (p->y & 0x03);
			y -= 2;
			z -= 2;
			y *= 64;
			z *= 64;
			y += 128;
			z += 128;
			p2->y = y;
			p2->z = z;
									  
#else
#error invalid color number!
#endif
		}
	}
	//printf("ch:%d\n", tmp.channels());
	cvtColor(tmp, dst, CV_Lab2BGR);
	return 0;
}
#endif
void compress_gbr(const char * imgname)
{
	Mat pic = imread(imgname);

	if (pic.data == NULL) { return; }
	Mat toShow(0, pic.cols, CV_8UC3);
	Mat pic2 = pic.clone();

	int h, w;
	for (h = 0; h < pic.rows; h++)
	{
		for (w = 0; w < pic.cols; ++w)
		{
			cv::Point3_<uchar>* p = pic2.ptr<cv::Point3_<uchar> >(h, w);
			p->x = p->x > 128 ? 255 : 0;
			p->y = p->y > 128 ? 255 : 0;
			p->z = p->z > 128 ? 255 : 0;
		}
	}
	toShow.push_back(pic);
	toShow.push_back(pic2);

	namedWindow("test color");
	imshow("test color", toShow);
	waitKey(0);
}
void test_color()
{
	Mat pic(256, 256, CV_8UC3);
	uint16_t u,v;
	int w, h;
	for (h = 0; h <= 255; h++)
	{
		for (w = 0; w<= 255; w++)
		{
			cv::Point3_<uchar>* p = pic.ptr<cv::Point3_<uchar> >(h, w);
			p->x = 100;
			p->y = w;
			p->z = h;			
		}
	}
	Mat pic2;
	cvtColor(pic, pic2, CV_YUV2BGR);
	namedWindow("test color");
	imshow("test color", pic2);
	waitKey(0);

}
void test_color(const char * imgname)
{
	Mat pic = imread(imgname);
	
	if (pic.data == NULL) { return; }
	Mat toShow(0, pic.cols, CV_8UC3);
	Mat tmp1, tmp2;

	toShow.push_back(pic);
/*
	to_Lab(pic, tmp1);
	from_Lab(tmp1, tmp2);
	toShow.push_back(tmp2);
*/	
	to_yuv(pic, tmp1);
	from_yuv(tmp1, tmp2);
	toShow.push_back(tmp2);

	

	namedWindow("test color");
	imshow("test color", toShow);
	waitKey(0);
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

#else


	check_lmdb("E:\\DeepLearning\\myColor\\train_data\\data.mdb");
#endif


	/*
	test_color("E:\\DeepLearning\\data\\coco\\train2014\\COCO_train2014_000000000030.jpg");
	test_color("E:\\DeepLearning\\data\\coco\\train2014\\COCO_train2014_000000000009.jpg");
	test_color("E:\\DeepLearning\\data\\coco\\train2014\\COCO_train2014_000000000025.jpg");
	test_color("E:\\DeepLearning\\data\\coco\\train2014\\COCO_train2014_000000000036.jpg");
	test_color("E:\\DeepLearning\\data\\coco\\train2014\\COCO_train2014_000000000071.jpg");
	*/
	//printf("%d, %d, %d, %d,%d", demulaw(mulaw(-128)), demulaw(mulaw(-3)), demulaw(mulaw(0)), demulaw(mulaw(64)), demulaw(mulaw(127)));
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


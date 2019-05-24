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

			width = record.datum().width();
			height = record.datum().height();
			channel = record.datum().channels();
			const string & datastr = record.datum().data();
			const uchar * data_ptr = (const uchar*)(datastr.c_str());
			printf("c:%d,w:%d,h:%d,bytes:%d vs %d \n", channel, width, height, datastr.length(), width*height);
			// load picture
			cv::Mat img;
			if (record.datum().encoded())
			{
				std::vector<uchar> vv(datastr.begin(), datastr.end());

				cv::Mat img = imdecode(vv, CV_LOAD_IMAGE_COLOR);

			}
			else
			{
				int h, w;
				for (h = 0; h < height; ++h)
				{
					for (w = 0; w < width; ++w)
					{
						if (channel == 3)
						{
							img = cv::Mat::ones(cv::Size(width, height), CV_8UC3);
							cv::Point3d * p = img.ptr<cv::Point3d>(h, w);

							p->x = record.datum().data()[0 * height * width + h * width + w];
							p->y = record.datum().data()[1 * height * width + h * width + w];
							p->z = record.datum().data()[2 * height * width + h * width + w];

						}
						else if (channel == 1)
						{
							img = cv::Mat::ones(cv::Size(width, height), CV_8UC1);

							img.at<uchar>(h, w) = data_ptr[h * width + w];


						}
						else
						{
							fprintf(stderr, "invalid channels!\n");
							return;
						}

					}
				}
			}

#if 1
			//draw annotation
			int grp_num = record.annotation_group_size();
			int i;
			for (i = 0; i < grp_num; ++i)
			{
				const AnnotationGroup & grp = record.annotation_group(i);
				int  label = grp.group_label();
				printf("label in group:%s\n", g_labels[label].c_str());
				int anno_num = grp.annotation_size();
				int j;
				for (j = 0; j < anno_num; ++j)
				{
					const NormalizedBBox &  bbox = grp.annotation(j).bbox();
					int x = bbox.xmin() * width;
					int y = bbox.ymin() * height;
					int w = (bbox.xmax() - bbox.xmin())*width;
					int h = (bbox.ymax() - bbox.ymin())*height;
					cv::Rect rec(x,y,w,h);
					
					cv::rectangle(img, rec, cv::Scalar(0, 0, 255));
					//printf("label in bbox:%s\n", g_labels[bbox.label()].c_str()); // label in bbox is invalid

				}

			}
#endif
			cv::imshow("check lmdb", img);
			cv::waitKey(0);
		}
		
	}
	mdb_cursor_close(cursor);
	mdb_dbi_close(env, dbi);
	mdb_env_close(env);
}

#include "errno.h"
int main()
{




	check_lmdb("E:\\DeepLearning\\ssd\\train_lmdb\\data.mdb");

    return 0;
}


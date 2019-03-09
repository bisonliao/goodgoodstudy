// ConvertRegionToClass.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <json/json.h>
#include <list>

#include <stdio.h>
#include "hdf5.h"

#include "H5Cpp.h"

using namespace H5;

using namespace cv;
using namespace std;

#define CROP_SZ (200)
#define BATCH_SZ (80000000 / (CROP_SZ * CROP_SZ*2))

typedef struct
{
	int x;
	int y;
	int width;
	int height;
}bbox_t;



//draw bounding box on outputImg, resize input and output
int draw_box_on_image(Mat & img, bbox_t box[], int boxnum, Mat & outputImg, bool isGray = true)
{
	int i = 0;
	
	
	cv::Mat dest(img.rows, img.cols, CV_8UC1, Scalar(0));
	int thick = ((double)img.rows / CROP_SZ);
	for (i = 0; i < boxnum; ++i)
	{
		Rect r(box[i].x, box[i].y, box[i].width, box[i].height);
		rectangle(dest, r, Scalar(255, 255, 255), thick);
		//rectangle(img, r, Scalar(255, 255, 255), thick);
	}
	Mat tmp = img;
	resize(tmp, img, Size(CROP_SZ, CROP_SZ), 0, 0, INTER_CUBIC);

	if (isGray)
	{
		tmp = img;
		cvtColor(tmp, img, COLOR_BGR2GRAY);
	}

	tmp = dest;
	resize(tmp, dest, Size(CROP_SZ, CROP_SZ), 0, 0, INTER_CUBIC);

	//namedWindow("template", WINDOW_AUTOSIZE);
	//imshow("template", dest);                // Show our image inside it.
#if 1
	static int checkFlag = 1;
	if (checkFlag)
	{
		checkFlag = 0;
		namedWindow("Display window", WINDOW_AUTOSIZE);
		imshow("Display window", img);                // Show our image inside it.
		waitKey(0); // Wait for a keystroke in the window
		imshow("Display window", dest);                // Show our image inside it.
		waitKey(0); // Wait for a keystroke in the window
	}
#endif
	outputImg = dest;
	
	return 0;
}
//draw segment poly on outputImg, resize input and output
int draw_segm_on_image(Mat & img, const char * jsonstr, Mat & outputImg, bool isGray = true)
{
	

	cv::Mat dest(img.rows, img.cols, CV_8UC1, Scalar(0));
	cv::Mat tmp;
	int thick = ((double)img.rows / CROP_SZ);
	
	if (isGray)
	{
		tmp = img;
		cvtColor(tmp, img, COLOR_BGR2GRAY);
	}
	

	Json::Reader r;
	Json::Value root;
	bool suc = r.parse(jsonstr, root);
	if (!suc)
	{
		fprintf(stderr, "failed to parse json string %s\n", jsonstr);
		return -1;
	}
	int segm_num = root.size();// how many segm in this image
	for (int i = 0;  i < segm_num; ++i)
	{
		int partnum = root[i].size(); // how many part in this segm

		for (int j = 0; j < partnum; ++j)
		{
			int coornum = root[i][j].size();//how many coordinary in this part
			static Point points[1][1024];
			int k;
			for (k = 0; k < coornum && k < 2048; k+=2)
			{
				int x = root[i][j][k].asInt();
				int y = root[i][j][k+1].asInt();
				points[0][k / 2] = Point(x, y);
			}
			
			const Point* pts[] = { points[0] };
			int npts[1];
			npts[0] = k / 2;
			//polylines(img, pts, npts, 1, true, Scalar(255), 3);
			fillPoly(dest, pts, npts, 1, Scalar(255));
			//polylines(dest, pts, npts, 1, true, Scalar(255), 3);
		}

	}

	tmp = dest;
	resize(tmp, dest, Size(CROP_SZ, CROP_SZ), 0, 0, INTER_CUBIC);

	tmp = img;
	resize(tmp, img, Size(CROP_SZ, CROP_SZ), 0, 0, INTER_CUBIC);

#if 1
	static int checkFlag = 1;
	if (checkFlag)
	{
		checkFlag = 0;
		namedWindow("Display window", WINDOW_AUTOSIZE);
		imshow("Display window", img);                // Show our image inside it.
		waitKey(0); // Wait for a keystroke in the window
		imshow("Display window", dest);                // Show our image inside it.
		waitKey(0); // Wait for a keystroke in the window
	}
#endif

	outputImg = dest;

	for (int i = 0; i < dest.rows; ++i)
	{
		for (int j = 0; j < dest.cols; ++j)
		{
			if (outputImg.at<uchar>(i, j) > 0)
			{
				outputImg.at<uchar>(i, j) = 1;
			}
		}
	}

	return 0;
}

// write image to directory, fcn official program use this input type
int write_to_directory(const Mat data[], const Mat label[])
{
	static int index = 0;
	const char * dir = "E:\\DeepLearning\\fcn.berkeleyvision.org-master\\voc-fcn8s\\data";
	char filename[1024];

	int i;
	for (i = 0; i < BATCH_SZ; ++i, ++index)
	{
		if (i >= (BATCH_SZ/10))
		{
			snprintf(filename, sizeof(filename), "%s\\train\\image\\train%d.jpg", dir, index);
			imwrite(filename, data[i]);
			snprintf(filename, sizeof(filename), "%s\\train\\label\\train%d.jpg", dir, index);
			imwrite(filename, label[i]);
		}
		else
		{
			snprintf(filename, sizeof(filename), "%s\\test\\image\\train%d.jpg", dir, index);
			imwrite(filename, data[i]);
			snprintf(filename, sizeof(filename), "%s\\test\\label\\train%d.jpg", dir, index);
			imwrite(filename, label[i]);
		}
	}
	return 0;
}

// write image to HDF5, my train procedure use this input type
int write_to_hdf5_color(const char * hdf5file, const Mat data[], const Mat label[])
{
	hid_t       file_id, dataset_id, dataspace_id;
	

	if (data[0].channels() != 3)
	{
		return -1;
	}
	if (label[0].channels() != 1)
	{
		return -2;
	}

	file_id = H5Fcreate(hdf5file, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	hsize_t dims[4];
	dims[0] = BATCH_SZ;
	dims[1] = 3;
	dims[2] = CROP_SZ;
	dims[3] = CROP_SZ;

	

	static unsigned char  buffer[BATCH_SZ][3][CROP_SZ][CROP_SZ];
	static unsigned char  labelBuffer[BATCH_SZ][1][CROP_SZ][CROP_SZ];
	int i, j, k,m;

	dataspace_id = H5Screate_simple(4, dims, NULL);
	

	dataset_id = H5Dcreate(file_id, "/data", H5T_STD_I8BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	for (i = 0; i < BATCH_SZ; ++i)
	{
		for (j = 0; j < CROP_SZ; ++j)//row
		{
			const uchar * rowScanPtr = data[i].ptr<uchar>(j);
			for (k = 0; k < CROP_SZ; ++k)//col
			{
				for (m = 0; m < 3; ++m)
				{
					buffer[i][m][j][k] = *rowScanPtr - 127;
					rowScanPtr++;

				}
			}
		}
	
	}
	H5Dwrite(dataset_id, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
	H5Dclose(dataset_id);
	H5Sclose(dataspace_id);

	dims[1] = 1;
	dataspace_id = H5Screate_simple(4, dims, NULL);
	dataset_id = H5Dcreate(file_id, "/label", H5T_STD_U8BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	for (i = 0; i < BATCH_SZ; ++i)
	{
		for (j = 0; j < CROP_SZ; ++j)
		{
			for (k = 0; k < CROP_SZ; ++k)
			{
				labelBuffer[i][0][j][k] = label[i].at<uchar>(j, k);
			}
		}
	}
	H5Dwrite(dataset_id, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, labelBuffer);
	H5Dclose(dataset_id);
	H5Sclose(dataspace_id);

	H5Fclose(file_id);
	return 0;
}
int write_to_hdf5_gray(const char * hdf5file, const Mat data[], const Mat label[])
{
	hid_t       file_id, dataset_id, dataspace_id;

	int channelNum = data[0].channels();

	if (channelNum != 1)
	{
		return -1;
	}
	if (label[0].channels() != 1)
	{
		return -2;
	}

	file_id = H5Fcreate(hdf5file, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	hsize_t dims[4];
	dims[0] = BATCH_SZ;
	dims[1] = 1;
	dims[2] = CROP_SZ;
	dims[3] = CROP_SZ;



	static unsigned char  buffer[BATCH_SZ][3][CROP_SZ][CROP_SZ];
	int i, j, k, m;

	dataspace_id = H5Screate_simple(4, dims, NULL);
	for (i = 0; i < BATCH_SZ; ++i)
	{
		for (j = 0; j < CROP_SZ; ++j)
		{
			for (k = 0; k < CROP_SZ; ++k)
			{
				buffer[i][0][j][k] = data[i].at<uchar>(j, k);
			}
		}
	}

	dataset_id = H5Dcreate(file_id, "/data", H5T_STD_I8BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	
	H5Dwrite(dataset_id, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
	H5Dclose(dataset_id);
	
	dataset_id = H5Dcreate(file_id, "/label", H5T_STD_U8BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	for (i = 0; i < BATCH_SZ; ++i)
	{
		for (j = 0; j < CROP_SZ; ++j)
		{
			for (k = 0; k < CROP_SZ; ++k)
			{
				buffer[i][0][j][k] = label[i].at<uchar>(j, k);
			}
		}
	}
	H5Dwrite(dataset_id, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
	H5Dclose(dataset_id);

	H5Sclose(dataspace_id);
	H5Fclose(file_id);
	return 0;
}

//genenate train data for segmentation
int gene_train_data_segm(const char * filename)
{
	FILE * fp = NULL;
	fopen_s(&fp, filename, "r");
	const char path[] = "E:\\DeepLearning\\data\\coco\\train2014\\";
	if (fp == NULL) { return -1; }

	


	int count = 0;
	int file_idx = 100;
	bool checkFlag = 1;
	static Mat input[BATCH_SZ], label[BATCH_SZ];
	
	printf("begin a new batch, size:%d\n", BATCH_SZ);
	while (1)
	{
		

		static char buffer[1024 * 100];
		char imagestr[1024];
		if (NULL == fgets(buffer, sizeof(buffer), fp))
		{
			break;
		}
		char * jsonstr = strchr(buffer, ' ');
		*jsonstr = 0;
		jsonstr++;

		snprintf(imagestr, sizeof(imagestr), "%s%s", path, buffer);

		
		cv::Mat img = imread(imagestr);
		cv::Mat outputImg;
		draw_segm_on_image(img,jsonstr, outputImg, false);
		input[count] = img;
		label[count] = outputImg;
		count++;;

		if ((count % 43) == 3)
		{
			printf("processed %d image\n", count);
		}

		if (count == BATCH_SZ)
		{
			count = 0;
			char filename[1024];
			snprintf(filename,sizeof(filename), "E:\\DeepLearning\\data\\coco\\HDF5_color\\train_%d.h5", file_idx++);
			write_to_hdf5_color(filename, input, label);
			//write_to_directory(input, label);

			printf("begin a new batch, size:%d\n", BATCH_SZ);
		}
		if (file_idx > 130)
		{
			break;
		}
	
		
	}
	fclose(fp);
	return 0;
	
}

// generate train data for bbox finding, actually it is not achievable in this way
int gene_train_data_bbox(const char * filename)
{
	FILE * fp = NULL;
	fopen_s(&fp, filename, "r");
	const char path[] = "E:\\DeepLearning\\data\\coco\\train2014\\";
	if (fp == NULL) { return -1; }




	int count = 0;
	int file_idx = 0;
	bool checkFlag = 1;
	static Mat input[BATCH_SZ], label[BATCH_SZ];

	printf("begin a new batch, size:%d\n", BATCH_SZ);
	while (1)
	{


		char buffer[1024 * 10];
		char imagestr[1024];
		if (NULL == fgets(buffer, sizeof(buffer), fp))
		{
			break;
		}
		char * jsonstr = strchr(buffer, ' ');
		*jsonstr = 0;
		jsonstr++;

		snprintf(imagestr, sizeof(imagestr), "%s%s", path, buffer);

		Json::Reader r;
		Json::Value root;
		bool suc = r.parse(jsonstr, root);
		if (!suc)
		{
			fprintf(stderr, "failed to parse json string %s\n", jsonstr);
			break;
		}
		int boxnum = root.size();
		bbox_t boxes[1024];
		for (int i = 0; i < 1024 && i < boxnum; ++i)
		{
			boxes[i].x = root[i][0].asInt();
			boxes[i].y = root[i][1].asInt();
			boxes[i].width = root[i][2].asInt();
			boxes[i].height = root[i][3].asInt();
		}
		cv::Mat img = imread(imagestr);
		cv::Mat outputImg;
		draw_box_on_image(img, boxes, boxnum, outputImg);
		input[count] = img;
		label[count] = outputImg;
		count++;;

		if (checkFlag && (count % 13) == 8)
		{
			checkFlag = 0;
			namedWindow("Display window", WINDOW_AUTOSIZE);
			imshow("Display window", img);                // Show our image inside it.
			waitKey(0); // Wait for a keystroke in the window
			imshow("Display window", outputImg);                // Show our image inside it.
			waitKey(0); // Wait for a keystroke in the window
		}

		if ((count % 43) == 3)
		{
			printf("processed %d image\n", count);
		}

		if (count == BATCH_SZ)
		{
			count = 0;
			char filename[1024];
			snprintf(filename, sizeof(filename), "E:\\DeepLearning\\data\\coco\\HDF5\\train_%d.h5", file_idx++);
			write_to_hdf5_gray(filename, input, label);

			printf("begin a new batch, size:%d\n", BATCH_SZ);
		}

		




	}
	fclose(fp);
	return 0;

}


int main()
{
	
	const char * imagelist = "E:\\DeepLearning\\data\\coco\\imgfile2seg.txt";
	gene_train_data_segm(imagelist);

	
    return 0;
}


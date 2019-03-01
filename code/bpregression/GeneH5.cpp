// GeneImdb.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <time.h>
#include <gflags/gflags.h>
#include <google/protobuf/text_format.h>

#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include "hdf5.h"
#include "H5Cpp.h"

using namespace H5;

#include <stdint.h>
#include <sys/stat.h> 

#include <fstream>  // NOLINT(readability/streams)
#include <string>


#include "caffe.pb.h"

//#include "caffe/util/format.hpp"


#define MAX_FIELD_NUM (100)
#define MAX_ITEM_NUM (10000)


typedef double InputType;



using namespace caffe;  // NOLINT(build/namespaces)

using std::string;

void gen_csv(const char * filename, int max)
{
	FILE * fp = NULL;
	fopen_s(&fp, filename, "wb+");
	if (fp == NULL) { perror("fopen:"); return; }
	int i;
	
	for (i = 0; i < max; ++i)
	{
		double x = ((double)rand()) / RAND_MAX * 10;
		double y = 2 * x*x + x + 3; /* 0 <= x(double)<=10.0  */


		fprintf(fp, "%f,%f,\n", x, y);

	}

	fclose(fp);
}
int split2fields(const char * str, InputType* fields)
{
	int i, j;
	char substr[100];
	int len = strlen(str);
	j = 0;
	int count = 0;
	for (i = 0; i < len && count < MAX_FIELD_NUM; ++i)
	{
		if (str[i] != ',')
		{
			substr[j++] = str[i];
		}
		else
		{
			substr[j++] = '\0';

			fields[count++] = atof(substr);


			j = 0;
		}
	}
	return count;
}
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

int getCropSize(int field_num)
{
	int r = (int)sqrt(field_num * sizeof(InputType));
	while ((r*r) <  (field_num * sizeof(InputType)))
	{
		r++;
	}
	return r;
}





void convert_dataset(const char* fileName,	const char* db_path) {


	hid_t       file_id, dataset_id, dataset_id2, dataspace_id, dataspace_id2;
	herr_t      status;


	file_id = H5Fcreate(db_path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	hsize_t dims[4], labelDims[2];
	dims[0] = MAX_ITEM_NUM;
	dims[1] = 1;
	dims[2] = 1;
	dims[3] = 1;

	labelDims[0] = MAX_ITEM_NUM;
	labelDims[1] = 1;



	dataspace_id = H5Screate_simple(4, dims, NULL);
	dataspace_id2 = H5Screate_simple(2, labelDims, NULL);




	/* Create the dataset. */
	dataset_id = H5Dcreate(file_id, "/data", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_id2 = H5Dcreate(file_id, "/label", H5T_IEEE_F64BE, dataspace_id2, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);


	FILE * fp; //= fopen(fileName, "rb");
	fopen_s(&fp, fileName, "rb");
	if (fp == NULL) { perror("fopen:"); return; }

	int count = 0;

	int item_id;


	static InputType inputData[MAX_ITEM_NUM][1][1][1];
	static InputType inputLabel[MAX_ITEM_NUM][1];

	for (item_id = 0; item_id < MAX_ITEM_NUM; item_id++)
	{
		char line[1024];
		if (NULL == fgets(line, sizeof(line) - 1, fp))
		{
			break;
		}
		rmReturn(line);

		InputType fields[MAX_FIELD_NUM];


		int field_num = split2fields(line, fields);

		inputData[item_id][0][0][0] = fields[0];

		inputLabel[item_id][0] = fields[1];
	






	}
	H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, inputData);
	H5Dwrite(dataset_id2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, inputLabel);

	H5Dclose(dataset_id);
	H5Dclose(dataset_id2);
	H5Sclose(dataspace_id);
	H5Sclose(dataspace_id2);
	H5Fclose(file_id);
	fclose(fp);

}
#include "errno.h"
int main()
{
	srand(time(NULL));
	
	gen_csv("E:\\DeepLearning\\structural_regression\\train.csv", MAX_ITEM_NUM);
	convert_dataset("E:\\DeepLearning\\structural_regression\\train.csv", "E:\\DeepLearning\\structural_regression\\train.h5");
	

	gen_csv("E:\\DeepLearning\\structural_regression\\test.csv", MAX_ITEM_NUM);
	convert_dataset("E:\\DeepLearning\\structural_regression\\test.csv", "E:\\DeepLearning\\structural_regression\\test.h5");
	
    return 0;
}


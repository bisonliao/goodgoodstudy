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


#include "caffe.pb.h"

//#include "caffe/util/format.hpp"


#define MAX_FIELD_NUM (100)

typedef uint16_t InputType;


using namespace caffe;  // NOLINT(build/namespaces)

using std::string;
uint32_t swap_endian(uint32_t val) {
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}
void gen_csv(const char * filename, int max)
{
	FILE * fp = NULL;
	fopen_s(&fp, filename, "wb+");
	if (fp == NULL) { perror("fopen:"); return; }
	int i;
	srand(time(NULL));
	for (i = 0; i < max; ++i)
	{
		int x = ((double)rand()) / RAND_MAX * 65535;
		int y = ((double)rand()) / RAND_MAX * 65535;
		int label = 1;
		if (x <= 32767 && y <= 32767)
		{
			label = 1;
		}
		else if (x > 32767 && y <= 32767)
		{
			label = 3;
		}
		else if (y > 32767 && x <= 32767)
		{
			label = 2;
		}
		fprintf(fp, "%d,%d,%d,\n", x, y, label);
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
			fields[count++] = atoi(substr);

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
			printf("record#%d: %s,%d,%d\n", i, k.c_str(), record.label(),record.channels());
		}
		
	}
	mdb_cursor_close(cursor);
	mdb_dbi_close(env, dbi);
	mdb_env_close(env);
}

void convert_dataset(const char* fileName,
	const char* db_path) {

#if 1
	
	MDB_env *env;
	MDB_txn * txn;
	MDB_dbi dbi;
	int iRet;
	if (iRet = mdb_env_create(&env))
	{
		fprintf(stderr, "mdb_env_create() failed, iRet=%d\n", iRet);
		return;
	}
	
	if (iRet = mdb_env_open(env, db_path, MDB_NOSUBDIR, 0644))
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
	
	


	FILE * fp; //= fopen(fileName, "rb");
	fopen_s(&fp, fileName, "rb");
	if (fp == NULL) { perror("fopen:"); return; }

	int count = 0;

	int item_id;

	unsigned char databuf[MAX_FIELD_NUM * sizeof(InputType)];

	string value;
	Datum datum;
	datum.set_channels(1);


	for (item_id = 0; ; item_id++)
	{
		char line[1024];
		if (NULL == fgets(line, sizeof(line) - 1, fp))
		{
			break;
		}
		rmReturn(line);

		InputType fields[MAX_FIELD_NUM];

		int field_num = split2fields(line, fields);


		if (item_id < 1)
		{
			printf("field number:%d, %d,%d,%d\n", field_num, fields[0], fields[1], fields[field_num - 1]);

			datum.set_height(1);
			datum.set_width((field_num - 1) * sizeof(InputType));
		}



	
		int offset = 0;
		memcpy(databuf, &fields[0], sizeof(InputType)*(field_num - 1));


		offset += sizeof(InputType)*(field_num - 1);


		datum.set_data(databuf, sizeof(InputType)*(field_num - 1));
		datum.set_label(fields[field_num - 1]);
		datum.set_channels(1);

		if (item_id < 3)
		{
			printf("label:%d ", datum.label());
			printf("x:%d, y:%d\n", *(InputType*)(databuf), *(InputType*)(databuf + sizeof(InputType)));
		}

		char skey[10];
		snprintf(skey, sizeof(skey), "%08d", item_id+1);
		string key_str = skey; // caffe::format_int(item_id, 8);

		
		datum.SerializeToString(&value);

		key.mv_data = (void*)key_str.c_str();
		key.mv_size = key_str.size();
		data.mv_data = (void*)value.c_str();
		data.mv_size = value.size();

		if (item_id < 3) { printf("keystr:%s, value len:%d\n", key_str.c_str(), value.length()); }

		if (iRet = mdb_put(txn, dbi, &key, &data, 0))
		{
			fprintf(stderr, "mdb_put returns %d\n", iRet);
			break;
		}
	
		if ((item_id % 100) == 97)
		{
			mdb_txn_commit(txn);
			mdb_txn_begin(env, NULL, 0, &txn);
			
		}

		
	}
	mdb_txn_commit(txn);
	mdb_dbi_close(env, dbi);
	mdb_env_close(env);
	
	fclose(fp);
#endif
}
#include "errno.h"
int main()
{
	
	printf("hello world [%08d]\n", 23);
	gen_csv("e:\\train.csv", 100000);
	convert_dataset("e:\\train.csv", "e:\\train.dat");

	gen_csv("e:\\test.csv", 10000);
	convert_dataset("e:\\test.csv", "e:\\test.dat");
	

	check_lmdb("e:\\train.dat");
	
    return 0;
}


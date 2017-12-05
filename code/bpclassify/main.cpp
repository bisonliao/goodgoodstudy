#include <gflags/gflags.h>
#include <google/protobuf/text_format.h>
         
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
         
#include <stdint.h>
#include <sys/stat.h> 
    
#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "boost/scoped_ptr.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"

#define MAX_FIELD_NUM (100)

typedef uint16_t InputType;
  
  
using namespace caffe;  // NOLINT(build/namespaces)
using boost::scoped_ptr;
using std::string;
uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
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
	    fields[count++] = atoi(substr) ;

		j = 0;
	}
  }
  return count;
}
void rmReturn(char * line)
{
    if (line == NULL) { return;}
    
    int len = strlen(line);
    if (len >= 2 && line[len-1] == '\n' && line[len-2] == '\r')
    {
	line[len-2] = '\0';	
    }
    else if (len >= 1 && line[len-1] == '\n' )
    {
    	line[len-1] = '\0';
    }
}

int getCropSize(int field_num)
{
    int r = (int)sqrt(field_num * sizeof(InputType));
    while ( (r*r) <  (field_num * sizeof(InputType)) )
    {
	r++;
    }
    return r;
}

void convert_dataset(const char* fileName,
        const char* db_path){

#if 1
  scoped_ptr<db::DB> db(db::GetDB("lmdb"));
  db->Open(db_path, db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  FILE * fp = fopen(fileName, "rb");
  if (fp == NULL) { perror("fopen:"); return;}

  int count = 0;

  int item_id;

  unsigned char databuf[MAX_FIELD_NUM*sizeof(InputType)];

  string value;
  Datum datum;
  datum.set_channels(1);

  
  for (item_id = 0; ; item_id++)
  {
    char line[1024];
    if ( NULL == fgets(line, sizeof(line)-1, fp) )
    {
	break;
    }
    rmReturn(line);

    InputType fields[MAX_FIELD_NUM];

    int field_num = split2fields(line, fields);


    if (item_id < 1)
    {
        printf("field number:%d, %d,%d,%d\n", field_num, fields[0], fields[1], fields[field_num-1]); 

  	datum.set_height(1);
  	datum.set_width( (field_num-1) * sizeof(InputType) );
    }



    int j;
    int offset = 0;
    memcpy(databuf, &fields[0], sizeof(InputType)*(field_num-1) );


    offset += sizeof(InputType)*(field_num-1);


    datum.set_data(databuf, sizeof(InputType)*(field_num-1) );
    datum.set_label(fields[field_num-1]);

    if (item_id < 3) 
    { 
        printf("label:%d ", datum.label());
        printf("x:%d, y:%d\n", *(InputType*)(databuf), *(InputType*)(databuf+sizeof(InputType)) ); 
    }

    string key_str = caffe::format_int(item_id, 8);
    datum.SerializeToString(&value);

    if (item_id < 3) { printf("keystr:%s, value len:%d\n", key_str.c_str(), value.length());}

    txn->Put(key_str, value);

    if (++count % 1000 == 0) {
      txn->Commit();
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
      txn->Commit();
  }
  db->Close();
  fclose(fp);
#endif
}

int main(int argc, char** argv) {
    if (argc < 3)
    {
	printf("usage:%s csvfile lmdbfile\n", argv[0]);
	return 0 ;
    }
    convert_dataset(argv[1], argv[2]);
    return 0;
}

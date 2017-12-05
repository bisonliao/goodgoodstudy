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

  
  
using namespace caffe;  // NOLINT(build/namespaces)
using boost::scoped_ptr;
using std::string;
uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

typedef uint16_t InputType;

void read_db( const char* db_path){

  scoped_ptr<db::DB> db(db::GetDB("lmdb"));
  db->Open(db_path, db::READ);
//  scoped_ptr<db::Transaction> txn(db->NewTransaction());
  scoped_ptr<db::Cursor> cursor(db->NewCursor());

  int count = 0;
  while (cursor->valid() && count < 10)
  {
	string key = cursor->key();
	string val = cursor->value();
	printf("key=%s\n", key.c_str());
	count++;
	cursor->Next();

    Datum datum;
	datum.ParseFromString(val);
	string databytes = datum.data();
	InputType x, y;
	x = *(InputType*)(databytes.c_str());
	y = *(InputType*)(databytes.c_str()+sizeof(InputType) );
	printf("w:%d,h:%d,channel:%d,label:%d, data:%u %u\n", 
		datum.width(), datum.height(), datum.channels(), datum.label(),
		x,y);
  }

//  db->Close();
}

int main(int argc, char** argv) {
	read_db("./range_train_lmdb");
	printf("--------------------\n");
	read_db("./range_test_lmdb");
    return 0;
}

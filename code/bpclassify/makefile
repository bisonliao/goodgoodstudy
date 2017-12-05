#caffe.pb.cc  caffe.pb.h  caffe.proto  convert_mnist_data.cpp  main.cpp

INC	= -DUSE_OPENCV  -I/usr/local/include/ -I/data/caffe-1.0/include -I/usr/local/cuda-8.0/targets/x86_64-linux/include/

#LIBS	=   /usr/local/lib/libprotobuf.a -lpthread  -lcaffe -llmdb -lhdf5  -lgflags -lboost_system -lboost_filesystem    -lopencv_calib3d -lopencv_contrib  -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab    -lopencv_core


#用  carlos目录下的opencv 2.4 编译不通过
INC	= -DUSE_OPENCV  -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/androidcamera/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/calib3d/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/contrib/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/core/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/dynamicuda/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/features2d/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/flann/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/gpu/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/highgui/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/imgproc/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/legacy/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/ml/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/nonfree/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/objdetect/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/ocl/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/photo/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/stitching/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/superres/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/ts/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/video/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/videostab/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/viz/include -I/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/modules/world/include -I/usr/local/include/ -I/data/caffe-1.0/include -I/usr/local/cuda-8.0/targets/x86_64-linux/include/

LIBS	=   /usr/local/lib/libprotobuf.a -lpthread  -lcaffe -llmdb -lhdf5  -lgflags -lboost_system -lboost_filesystem    -L/data/carloswong/caffe_compile_pack/opencv-2.4.13.3/build/lib/ -lopencv_calib3d -lopencv_contrib  -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab    -lopencv_core







all:convert_to_lmdb  classify  readlmdb

convert_to_lmdb:caffe.pb.o main.o
	g++ -g -o $@  main.o caffe.pb.o $(LIBS)

readlmdb:caffe.pb.o readlmdb.o
	g++ -g -o $@  readlmdb.o caffe.pb.o $(LIBS)


caffe.pb.o:caffe.proto
	protoc --cpp_out=. caffe.proto
	g++ -g -c  caffe.pb.cc
main.o:main.cpp
	g++ -g -c main.cpp $(INC)
readlmdb.o:readlmdb.cpp
	g++ -g -c readlmdb.cpp $(INC)
	


classify:classification.o
	g++ -g -o classify classification.o $(LIBS)
classification.o:classification.cpp
	g++ -g -c classification.cpp $(INC)

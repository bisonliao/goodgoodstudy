// UseTrainedModel.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include "layer_reg.h"
#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <io.h>
#include <fcntl.h>
#include <sys\types.h>
#include <sys\stat.h>

using namespace caffe; 
using std::string;
using namespace cv;
 

#define input_size (227)
#define input_channel  (3)

#define label_size (13)
#define label_channel  (256)


 
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

void caffe_forward(boost::shared_ptr< Net<float> > & net, float *data_ptr, float* label_ptr)
{
	Blob<float>* data_blobs = net->input_blobs()[0];
	
	Blob<float>* label_blobs = net->input_blobs()[1];
	//printf("input blob count: %d, %d\n", data_blobs->count(), label_blobs->count());

	
	switch (Caffe::mode())
	{
	case Caffe::CPU:
		memcpy(data_blobs->mutable_cpu_data(), data_ptr,
			sizeof(float) * data_blobs->count());

		memcpy(label_blobs->mutable_cpu_data(), label_ptr,
			sizeof(float) * label_blobs->count());
		break;
	case Caffe::GPU:
		
		cudaMemcpy(data_blobs->mutable_gpu_data(), data_ptr,
		sizeof(float) * data_blobs->count(), cudaMemcpyHostToDevice);

		cudaMemcpy(label_blobs->mutable_gpu_data(), label_ptr,
			sizeof(float) * label_blobs->count(), cudaMemcpyHostToDevice);
		
		
		break;
	default:
		LOG(FATAL) << "Unknown Caffe mode.";
	}
	net->Forward();
}
//根据反向传播获得的梯度，更新输入的照片
void update_input(float * input, const float * diff)
{
	int width, height, chn;


	for (chn = 0; chn < 3; ++chn)
	{
		for (height = 0; height < input_size; ++height)
		{
			for (width = 0; width < input_size; ++width)
			{
				int index = chn * input_size * input_size + height * input_size + width;
				
				input[index] = input[index] - 0.5 * diff[index];
			}
		}
	}
}

//将学习到的输入照片展示出来对比查看效果
void show_input(const float * data_input, const Mat & source)
{
	Mat show = cv::Mat::zeros(input_size, input_size, CV_8UC3);

	const float max =  *(std::max_element<const float*>(data_input, data_input + input_size * input_size*input_channel));
	const float min = *(std::min_element<const float*>(data_input, data_input + input_size * input_size*input_channel));
	printf("max and min:%f,%f\n", max, min);
	
	
	for (int height = 0; height < input_size; ++height)
	{
		for (int width = 0; width < input_size; ++width)
		{
			cv::Point3_<uchar>* p = show.ptr<cv::Point3_<uchar> >(height, width);

			int index;
			
			index = 0 * input_size * input_size + height * input_size + width;
			p->x = (data_input[index] - min) / (max - min) * 255;
			index = 1 * input_size * input_size + height * input_size + width;
			p->y = (data_input[index] - min) / (max - min) * 255;
			index = 2 * input_size * input_size + height * input_size + width;
			p->z = (data_input[index] - min) / (max - min) * 255;

		}
	}
	show.push_back(source);
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	cv::imshow("Display window", show);
	cv::waitKey(0);
	cv::destroyWindow("Display window");
}
//读取提前保存好featuremap，来自名画在alexnet里前向传播产生的feature map
void load_featuremap(float * label_ptr)
{
	BlobProto blob;
	caffe::ReadProtoFromBinaryFileOrDie("E:\\DeepLearning\\art_style\\feature_map.binaryproto", &blob);
	if (blob.channels() != label_channel ||
		blob.height() != label_size ||
		blob.width() != label_size)
	{
		fprintf(stderr, "blob size mismatch!\n");
		return;
	}
	int length = blob.data_size();
	for (int i = 0; i < length; ++i)
	{
		label_ptr[i] = blob.data()[i];
	}
	
}
  
void art_style(boost::shared_ptr<Net<float> > net, const Mat & img)
{
	//初始化输入照片数据和作为标注的feature map
	float data_input[input_channel][input_size][input_size];
	float label_input[label_channel][label_size][label_size];

	load_featuremap(&label_input[0][0][0]);
	
	
	int width, height, chn;
	for (height = 0; height < input_size; ++height)
	{
		for (width = 0; width < input_size; ++width)
		{
			const cv::Point3_<uchar>* p = img.ptr<cv::Point3_<uchar> >(height, width);
			
			data_input[0][height][width] = p->x ;//B
			data_input[1][height][width] = p->y;//G
			data_input[2][height][width] = p->z ;//R
		}
	}
	

	int it;
	float loss = 1;
	for (it = 0; it < 10000; ++it)//迭代
	{


		caffe_forward(net, (float*)data_input, (float*)label_input);
		net->Backward();

		//取巧，Scale层的bottom blob的梯度，就是输入图片的梯度
		int index1 = get_blob_index(net, "data");
		boost::shared_ptr<Blob<float> > blob = net->blobs()[index1];
		const float * diff = blob->cpu_diff();//梯度保存在这里
		if (it == 2)
		{
			for (int j = 0; j < 50; ++j)
			{
				printf("%f ", diff[j]);
			}
			printf("\n");
		}

		//计算这次迭代的loss值
		index1 = get_blob_index(net, "loss");
		blob = net->blobs()[index1];
		unsigned int num_data = blob->count();
		loss = blob->cpu_data()[0];
		if ((it % 113) == 0)
		{
			printf("iter %d, loss=%f\n", it, loss);
		}

		//根据梯度更新输入，其他参数不更新
		update_input(&data_input[0][0][0], diff);
	}
	
	//显示更新后的输入照片
	show_input(&data_input[0][0][0], img);

}







int main(int argc, char **argv) {


	cv::Mat img = cv::imread("E:\\DeepLearning\\art_style\\object.jpg");//被修改的照片
	if (img.data == NULL)
	{
		fprintf(stderr, "failed to load image file\n");
		return -1;
	}
	if (img.channels() != input_channel)
	{
		fprintf(stderr, "image channel is not 3,abort!\n");
		return -1;
	}
	cv::Mat img2 = img;
	cv::resize(img, img2, cv::Size(input_size, input_size), 0.0, 0.0, cv::INTER_CUBIC);
	img = img2;


	const char *proto = "E:\\DeepLearning\\art_style\\train_val.prototxt";
	const char *model = "E:\\DeepLearning\\art_style\\alexnet\\bvlc_alexnet.caffemodel";

	Phase phase = TRAIN;
	Caffe::set_mode(Caffe::GPU);


	
	
	boost::shared_ptr<Net<float> > net(new caffe::Net<float>(proto, phase));
	net->CopyTrainedLayersFrom(model);
	
	
	art_style(net,img);




	return 0;

}



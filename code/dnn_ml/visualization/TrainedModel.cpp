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

void caffe_forward(boost::shared_ptr< Net<float> > & net, float *data_ptr)
{
	Blob<float>* input_blobs = net->input_blobs()[0];
	//printf("input blobs size:%d, uchar count:%d\n", net->input_blobs().size(), input_blobs->count());

	float label = 1.0;
	
	Blob<float>* input_blobs2 = net->input_blobs()[1];
	//printf("input blobs size:%d, uchar count:%d\n", net->input_blobs().size(), input_blobs2->count());

	
	switch (Caffe::mode())
	{
	case Caffe::CPU:
		memcpy(input_blobs->mutable_cpu_data(), data_ptr,
			sizeof(float) * input_blobs->count());

		memcpy(input_blobs2->mutable_cpu_data(), &label,
			sizeof(float) * input_blobs2->count());
		break;
	case Caffe::GPU:
		
		cudaMemcpy(input_blobs->mutable_gpu_data(), data_ptr,
		sizeof(float) * input_blobs->count(), cudaMemcpyHostToDevice);

		cudaMemcpy(input_blobs2->mutable_gpu_data(), &label,
			sizeof(float) * input_blobs2->count(), cudaMemcpyHostToDevice);
		
		
		break;
	default:
		LOG(FATAL) << "Unknown Caffe mode.";
	}
	net->Forward();
}
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
				input[index] = input[index] - 4000 * diff[index];
			}
		}
	}
}

void save_input(const float * input)
{
	int fd = _open("e:\\input.dat", _O_BINARY | _O_CREAT|_O_WRONLY);
	if (fd < 0)
	{
		fprintf(stderr, "failed to open file!\n");
		return;
	}
	size_t sz = input_size * input_size * 3 * sizeof(float);
	_write(fd, input, sz);
	_close(fd);
}
void show_input(const float * data_input)
{
	Mat show = cv::Mat::zeros(input_size, input_size, CV_32FC3);
	
	for (int height = 0; height < input_size; ++height)
	{
		for (int width = 0; width < input_size; ++width)
		{
			cv::Point3_<float>* p = show.ptr<cv::Point3_<float> >(height, width);
			int index;
			index = 0 * input_size * input_size + height * input_size + width;
			p->x = data_input[index];
			index = 1 * input_size * input_size + height * input_size + width;
			p->y = data_input[index];
			index = 2 * input_size * input_size + height * input_size + width;
			p->z = data_input[index];


		}
	}
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	cv::imshow("Display window", show);
	cv::waitKey(0);
	cv::destroyWindow("Display window");
}
  
void visulization(boost::shared_ptr<Net<float> > net)
{


	float data_input[input_channel][input_size][input_size];
	
	int width, height, chn;
	cv::Mat img = cv::Mat::ones(input_size, input_size, CV_8UC3);

	
	
	for (height = 0; height < input_size; ++height)
	{
		for (width = 0; width < input_size; ++width)
		{
			cv::Point3_<uchar>* p = img.ptr<cv::Point3_<uchar> >(height, width);
			
			data_input[0][height][width] = p->x ;//B
			data_input[1][height][width] = p->y;//G
			data_input[2][height][width] = p->z ;//R

		}
	}

	int it;
	float loss = 1;
	for (it = 0; it < 100 && loss > 0.001; ++it)
	{

		caffe_forward(net, (float*)data_input);

		int index1 = get_blob_index(net, "pool1");
		//	check_conv_blob(net->blobs()[index1]);

		net->Backward();
		index1 = get_blob_index(net, "data");
		boost::shared_ptr<Blob<float> > blob = net->blobs()[index1];
		const float * diff = blob->cpu_diff();//梯度保存在这里

		//计算这次迭代的loss值
		index1 = get_blob_index(net, "loss");
		blob = net->blobs()[index1];
		unsigned int num_data = blob->count();
		loss = blob->cpu_data()[0];
		printf("iter %d, loss=%f\n", it, loss);

		//根据梯度更新输入，其他参数不更新
		update_input(&data_input[0][0][0], diff);
	}
	
	//显示输入的“图片”
	show_input(&data_input[0][0][0]);

	save_input(&data_input[0][0][0]);

}







int main(int argc, char **argv) {


	


	const char *proto = "E:\\DeepLearning\\visualization\\gender_train_val.prototxt";
	const char *model = "E:\\DeepLearning\\visualization\\gender_net.caffemodel";

	Phase phase = TRAIN;
	Caffe::set_mode(Caffe::GPU);


	
	
	boost::shared_ptr<Net<float> > net(new caffe::Net<float>(proto, phase));
	net->CopyTrainedLayersFrom(model);
	
	
	visulization(net);




	return 0;

}



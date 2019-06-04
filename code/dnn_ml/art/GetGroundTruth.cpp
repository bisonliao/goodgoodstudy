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


void paste_img(Mat & big, Mat & small, int row, int col)
{
	int ch = small.channels();
	int height = small.rows;
	int width = small.cols;
	if (big.channels() != ch || ch != 3 && ch != 1)
	{
		fprintf(stderr, "channel mismatch!\n");
		return;
	}
	for (int c = 0; c < ch; c++)
	{
		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				int hh = row * height + h;
				int ww = col * width + w;
				if (ch == 3)
				{
					*(big.ptr<Point3_<uchar>>(hh, ww)) = *(small.ptr<Point3_<uchar>>(h, w));
					
				}
				else if (ch == 1)
				{
					big.at<char>(hh, ww) = small.at<char>(h, w);
				}
			
			}
		}
	}
}

void check_conv_blob(boost::shared_ptr<const Blob<float> > blob)
{
	int ch = blob->channels();
	int height = blob->height();
	int width = blob->width();
	const float * ptr = blob->cpu_data();
	if (blob->num() != 1)
	{
		fprintf(stderr, "num in blob should be 1! %d\n", blob->num());
		return;
	}

	int cnt = ceil(sqrt(ch));
	Mat big = cv::Mat::zeros(cnt*height, cnt*width, CV_8UC1);

	
	for (int c = 0; c < ch; c++)
	{
		cv::Mat img = cv::Mat::zeros(height, width, CV_8UC1);
		
		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				img.at<uchar>(h, w) = ptr[c*width*height + h * width + w];
				if (c == 0 && h == 13)
				{
					printf("%.2f ", ptr[c*width*height + h * width + w]);
				}
			}
			if (c == 0 && h == 13)
			{
				printf("\n");
			}

		
		}
	
		paste_img(big, img, c / cnt, c%cnt);

		
	}
	cv::namedWindow("blob");
	cv::imshow("blob", big);
	cv::waitKey(0);
	BlobProto proto;
	proto.set_channels(ch);
	proto.set_height(height);
	proto.set_width(width);
	ptr = blob->cpu_data();
	int length = blob->count();
	for (int i = 0; i < length; ++i)
	{
		
		proto.add_data(ptr[i]);
	}
	caffe::WriteProtoToBinaryFile(proto, "e:\\DeepLearning\\art_style\\feature_map.binaryproto");
}


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
	printf("input blobs size:%d, uchar count:%d\n", net->input_blobs().size(), input_blobs->count());
	
	
	switch (Caffe::mode())
	{
	case Caffe::CPU:
		memcpy(input_blobs->mutable_cpu_data(), data_ptr,
			sizeof(float) * input_blobs->count());
		break;
	case Caffe::GPU:
		
		cudaMemcpy(input_blobs->mutable_gpu_data(), data_ptr,
		sizeof(float) * input_blobs->count(), cudaMemcpyHostToDevice);
		
		
		break;
	default:
		LOG(FATAL) << "Unknown Caffe mode.";
	}
	net->Forward();
}
  
void classify(boost::shared_ptr<Net<float> > net,
	cv::Mat & img, const char * meanfile)
{
	BlobProto mean;
	if (!ReadProtoFromBinaryFile(meanfile, &mean))
	{
		fprintf(stderr, "failed to read meanfile! %s\n", meanfile);
		return;
	}
	printf("mean size:[%d, %d, %d, %d]\n", mean.num(), mean.channels(), mean.height(), mean.width());
	// 均值blob的尺寸与输入层blob不一致，怎么用？不减均值，似乎也ok

	float data_input[input_channel][input_size][input_size];
	
	int width, height, chn;


	
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


	
	caffe_forward(net, (float*)data_input);

	int index1 = get_blob_index(net, "conv5");
	check_conv_blob(net->blobs()[index1]);

#if 0
	int index = get_blob_index(net, "prob");
	boost::shared_ptr<Blob<float> > blob = net->blobs()[index];
	unsigned int num_data = blob->count();
	int i;
	for (i = 0; i < num_data; ++i)
	{
		const float *blob_ptr = (const float *)blob->cpu_data();
		printf("%f\n", *(blob_ptr+i) );
	}
	
	printf("\n");
#endif	


}





int main(int argc, char **argv) {

	
	

	cv::Mat img = cv::imread("E:\\DeepLearning\\art_style\\art.jpg" );
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

	
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	cv::imshow("Display window", img);
	cv::waitKey(0);
	cv::destroyWindow("Display window");

	
	
	


	const char *proto = "E:\\DeepLearning\\art_style\\alexnet\\alexnet_deploy.prototxt";
	const char *model = "E:\\DeepLearning\\art_style\\alexnet\\bvlc_alexnet.caffemodel";
	char *mean_file =   "E:\\DeepLearning\\age_gender_classify\\mean.binaryproto";
	Phase phase = TRAIN;
	Caffe::set_mode(Caffe::GPU);
	
	
	boost::shared_ptr<Net<float> > net(new caffe::Net<float>(proto, phase));
	net->CopyTrainedLayersFrom(model);
	
	classify(net, img, mean_file);



	return 0;

}



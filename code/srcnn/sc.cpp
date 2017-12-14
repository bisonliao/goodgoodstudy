#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

const int label_size=21;
const int input_size=33;
const int scale=3;

unsigned int get_blob_index(boost::shared_ptr< Net<float> > & net, char *query_blob_name)
{
    std::string str_query(query_blob_name);    
    vector< string > const & blob_names = net->blob_names();
    for( unsigned int i = 0; i != blob_names.size(); ++i ) 
    { 
        if( str_query == blob_names[i] ) 
        { 
            return i;
        } 
    }
    LOG(FATAL) << "Unknown blob name: " << str_query;
}

void caffe_forward(boost::shared_ptr< Net<float> > & net, float *data_ptr)
{
    Blob<float>* input_blobs = net->input_blobs()[0];
    switch (Caffe::mode())
    {
    case Caffe::CPU:
        memcpy(input_blobs->mutable_cpu_data(), data_ptr,
            sizeof(float) * input_blobs->count());
        break;
    case Caffe::GPU:
	/*
        cudaMemcpy(input_blobs->mutable_gpu_data(), data_ptr,
            sizeof(float) * input_blobs->count(), cudaMemcpyHostToDevice);
			*/
        LOG(FATAL) << "Unknown Caffe mode.";
        break;
    default:
        LOG(FATAL) << "Unknown Caffe mode.";
    } 
    net->Forward();
}

void super_resolution(boost::shared_ptr<Net<float> > net,
	    cv::Mat & img,  cv::Mat & img2,
		int i, int j)
{
		float data_input[input_size][input_size];

		int sub_i, sub_j;
		for (sub_i = 0; sub_i < input_size; ++sub_i)
		{
				for (sub_j = 0; sub_j < input_size; ++sub_j)
				{
						data_input[sub_i][sub_j] = (float)(img.at<uchar>(i+sub_i, j+sub_j));
				}
		}

		caffe_forward(net, (float*)data_input);
		int index = get_blob_index(net, "conv3");
		boost::shared_ptr<Blob<float> > blob = net->blobs()[index];
		unsigned int num_data = blob->count(); 
		const float *blob_ptr = (const float *) blob->cpu_data();

		for (sub_i = 0; sub_i < label_size; ++sub_i)
		{
				for (sub_j = 0; sub_j < label_size; ++sub_j)
				{
						img2.at<uchar>(i+sub_i, j+sub_j) = (unsigned char)(blob_ptr[sub_i*label_size+sub_j]);
				}
		}

}

void get_scaled_pic(cv::Mat& img)
{
	
	cv::imwrite("./sr_source.bmp", img);	

	cv::Mat tmp;
	tmp = img;
	cv::resize(tmp, img, cv::Size(tmp.cols*scale, tmp.rows*scale), 0, 0, cv::INTER_CUBIC);
	cv::imwrite("./sr_bicubic.bmp", img);	
}
		

int main(int argc, char** argv) {

	if (argc < 2)
	{
		printf("usage:%s picture \n", argv[0]);
		return 255;
	}
	cv::Mat img = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	get_scaled_pic(img);

	cv::Mat img2(img.rows-(input_size-label_size), img.cols-(input_size-label_size), CV_8UC1);


	const char *proto = "/data/bisonliao/srcnn/SRCNN_mat.prototxt";
    const char *model = "/data/bisonliao/srcnn/SRCNN_iter_700000.caffemodel";
    //char *mean_file = "H:\\Models\\Caffe\\imagenet_mean.binaryproto";
    Phase phase = TEST;
    Caffe::set_mode(Caffe::CPU);

	boost::shared_ptr<Net<float> > net(new caffe::Net<float>(proto, phase));
    net->CopyTrainedLayersFrom(model);

	int i, j;

	for (i = 0; i < img.rows-input_size; i+=label_size)
	{
    	for (j = 0; j < img.cols-input_size; j+=label_size)
    	{
       		super_resolution(net, img, img2, i, j); 

    	}
	}
	//修一下边幅 ， 最后一行
	i = img2.rows  - label_size;
    for (j = 0; j < img.cols-input_size; j+=label_size)
    {
       	super_resolution(net, img, img2, i, j); 
    }

	
	//修一下边幅 ， 最后一列
	j = img2.cols  - label_size;
    for (i = 0; i < img.rows-input_size; i+=label_size)
    {
       	super_resolution(net, img, img2, i, j); 
    }
	//修一下边幅，右下角
	i = img2.rows  - label_size;
	j = img2.cols  - label_size;
	{
       	super_resolution(net, img, img2, i, j); 
	}

	cv::imwrite("./sr_srcnn.bmp", img2);

	return 0;

}

#endif

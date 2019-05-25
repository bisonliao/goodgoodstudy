// UseOpenCV.cpp: 主项目文件。

#include "stdafx.h"
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string>
using namespace std;

using namespace System;

string classes[] = { "background", "aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
"sofa", "train", "tvmonitor" };

void do_ssd(cv::dnn::experimental_dnn_34_v11::Net & net, const char * imagefile)
{
	cv::Mat img = cv::imread(imagefile);
	cv::Mat resized = img.clone();
	cv::resize(resized, img, cv::Size(300, 300));
	cv::Mat blob = cv::dnn::experimental_dnn_34_v11::blobFromImage(img, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 117.0, 123.0));
	net.setInput(blob);
	cv::Mat detect = net.forward();
	cv::MatSize sz = detect.size;
	printf("dim:%d, [%d,%d,%d,%d]\n", sz.dims(), sz[0], sz[1], sz[2], sz[3]);

	int det_num = sz[2];
	int i;
	const float * result = detect.ptr<float>(); // imageid,label,score, bbox position
	cv::Mat image2 = img.clone();
	for (i = 0; i < det_num; ++i)
	{
		if (result[0] < 0 || result[2] < 0.8)
		{
			result = result + 7;
			continue;
		}
		int index = result[1];
		float xmin = *(result + 3);
		float ymin = *(result + 4);
		float xmax = *(result + 5);
		float ymax = *(result + 6);

		int x = xmin * 300;
		int y = ymin * 300;
		int w = (xmax - xmin) * 300;
		int h = (ymax - ymin) * 300;

		printf("%s, %.2f, [%d, %d, %d, %d]\n", classes[index].c_str(), result[2], x, y, w, h);


		cv::Rect rec(x, y, w, h);
		cv::rectangle(image2, rec, cv::Scalar(0, 0, 255));

		result = result + 7;
	}
	cv::namedWindow("bison", cv::WINDOW_AUTOSIZE);
	cv::imshow("bison", image2);
	cv::waitKey(0);
}

int main(int argc, char ** argv)
{
	string imagefile = "E:/DeepLearning/data/VOCdevkit/VOC2012/JPEGImages/2011_001126.jpg";
	//string imagefile = "E:/DeepLearning/data/VOCdevkit/VOC2012/JPEGImages/2008_002809.jpg";
	/*
	string deploy_prototxt = "E:\\DeepLearning\\ssd\\official_model\\models\\VGGNet\\VOC0712\\SSD_300x300\\deploy.prototxt";
	string model = "E:\\DeepLearning\\ssd\\official_model\\models\\VGGNet\\VOC0712\\SSD_300x300\\VGG_VOC0712_SSD_300x300_iter_120000.caffemodel";
	*/

	string deploy_prototxt = "E:\\DeepLearning\\ssd\\deploy.prototxt";
	string model = "E:\\DeepLearning\\ssd\\snapshot\\ssd_iter_88000.caffemodel";
	float confidence_default = 0.5;

	
	cv::dnn::experimental_dnn_34_v11::Net net = cv::dnn::readNetFromCaffe(deploy_prototxt, model);

	
	if (argc < 2)
	{
		do_ssd(net, imagefile.c_str());
	}
	else
	{
		for (int i = 1; i < argc; ++i)
		{
			printf("%s\n", argv[i]);
			do_ssd(net, argv[i]);
		}
	}

    return 0;
}

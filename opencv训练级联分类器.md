# 使用opencv训练级联分类器 #

网上很多这样的例子和文章，但是在做的过程中，踩了坑，而且opencv_traincascade.exe这个工具也基本上不报错，只crash，让笔者相当痛苦。所以最后下决心写个小文记录下来。

大概的步骤是这样的：

1. 准备正样本，使用opencv_createsamples.exe命令创建.vec文件
2. 准备负样本
3. 使用opencv_traincascade.exe训练分类器
4. 编程使用分类器

下面以人脸检测为例，说明一下这个过程：

## 1 准备正样本 ##

需要标注人脸在图片中的位置，并记录到一个txt文件里，每行一个样本：

	图片名 人脸个数 待检测目标矩形在图片中的位置(左上x 左上y 矩形宽 矩形高)

例如一个pos.txt文件的几行：

	cut\image_0001.jpg 1 5 15 99 108
	cut\image_0002.jpg 1 15 40 100 118

然后调用opencv_createsamples.exe命令根据pos.txt文件创建.vec样本文件。

详细看一下opencv_createsamples.exe的参数

	Usage: opencv_createsamples.exe
	  [-info <collection_file_name>]   ：输入的pos.txt文件，第一种输入方式
	  [-img <image_file_name>]         :输入一张图片，例如公司logo，各种变换后作为样本，第二种输入方式
	  [-vec <vec_file_name>]  ：输出的.vec正样本文件
	  [-num <number_of_samples = 1000>]：输出到.vec中样本个数，主要适合第二种输入方式
	  [-w <sample_width = 24>]：很重要，这里有坑。这是指输出到.vec文件中统一的样本的宽度
	  [-h <sample_height = 24>]：输出到.vec文件中统一的正样本的高度
	  ... ：其他不懂的参数，我就省略掉

命令示例：

	opencv_createsamples.exe -info .\pos.txt   -vec pos.vec -num 408 -w 100 -h 100

这样就生成了正样本pos.vec文件

可以使用下面的命令，查看pos.vec文件中的图片：

	opencv_createsamples.exe   -vec pos.vec -w 100 -h 100

## 2 准备负样本 ##

相比准备正样本，负样本简单很多，只需要一个.txt文件。

需要注意的是，负样本图片中不要包含待检测的目标物体，例如人脸

负样本.txt文件，建议使用绝对路径，例如neg.txt：

	D:\opencv\face\nonface\image_0004.jpg
	D:\opencv\face\nonface\image_0005.jpg
	D:\opencv\face\nonface\image_0006.jpg

## 3 训练分类器 ##

	Usage: opencv_traincascade
	  -data <cascade_dir_name> ：模型文件输出的路径
	  -vec <vec_file_name>：输入的正样本，例如前面的pos.vec
	  -bg <background_file_name>：输入的负样本，例如前面的neg.txt
	  [-numPos <number_of_positive_samples = 2000>]：.vec中正样本个数，很重要，不正确指定可能crash
	  [-numNeg <number_of_negative_samples = 1000>]：负样本个数，很重要，不正确指定可能crash
	  [-numStages <number_of_stages = 20>]：应该是迭代多少次吧
	--cascadeParams--
	  [-stageType <BOOST(default)>]：不懂
	  [-featureType <{HAAR(default), LBP, HOG}>]：三种特征匹配算法
	  [-w <sampleWidth = 24>]：很重要，.vec中正样本的宽度，不正确指定可能crash
	  [-h <sampleHeight = 24>]：很重要，.vec中正样本的高度，不正确指定可能crash
	  ...：其他不懂的参数，我也省略掉了

命令示例：

	opencv_traincascade.exe -data d:\opencv\face -vec d:\opencv\face\pos.vec -bg d:\opencv\face\neg.txt -w 100 -h 100 -numPos 300 -numNeg 300 -numStages 20  -featureType HAAR -mode ALL

经过好长一段时间后，训练如果成功，就会生成一个cascade.xml文件，这就是模型的描述文件。

注意：如果-data指定的目录下已经存在以前训练的.xml结果文件，会导致上述命令执行失败，提示的错误信息牛头不对马嘴，所以要清理一下目录。


## 4 编程使用分类器 ##

代码比较简单：假设argv[1]指定模型描述文件，argv[2]指定待检测的图片：
	
	Mat image, ROI;
	CascadeClassifier Mycascade;
	if (argc < 3) 
	{ 
		printf("%s [mode] [image]\n", argv[0]); 
		return -1; 
	}

	if (!Mycascade.load(argv[1])) { 
		printf("[error] 无法加载级联分类器文件！\n");   
		return -1; 
	}
	
	image = imread(argv[2]);//读取图片  
	if (!image.data) { 
		printf("[error] 没有图片\n");   
		return -5; 
	}

	std::vector<Rect> position;
	vector<double> weights;
	vector<int> levels;

	Mycascade.detectMultiScale(image, position, levels, weights, 
				1.1, 3, 0, Size(), Size(), true);

	printf("识别到%d个目标\n", position.size());
	
	
	for (int i = 0; i < position.size(); i++) {
		rectangle(image, position[i],Scalar(0, 255, 0), 1);                   
	}
	
	namedWindow("result", WINDOW_AUTOSIZE);
	imshow("result", image);
	waitKey(0);

感觉这个鬼玩意有bug，有时候训练跑起来，两个小时也没有进展，一直停留在 stage 0.

最后我训练出来的分类器很差劲，人脸识别不正确，可能样本数300多个不够吧。不过用opencv自带的模型倒是还可以，在下面的目录里：

	opencv\sources\data



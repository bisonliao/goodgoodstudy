#pragma once
#include "caffe/layer.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
	template <typename Dtype>
	class ContrastiveAccuracyLayer :
		public Layer<Dtype>
	{

	public:
		explicit ContrastiveAccuracyLayer(const LayerParameter& param): Layer<Dtype>(param) ,margin_(1.0){}
		
		virtual ~ContrastiveAccuracyLayer(){}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "ContrastiveAccuracyLayer"; }
		virtual inline int ExactNumBottomBlobs() const { return 3; }
		virtual inline int ExactNumTopBlobs() const { return 1; }
	protected:
	
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,	const vector<Blob<Dtype>*>& top);
		                 
		/*��Ϊ���ڲ��Խ׶���ʾ׼ȷ�ʣ�����Ҫ���򴫲�*/
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {};
	protected:
		float margin_;

	};
}


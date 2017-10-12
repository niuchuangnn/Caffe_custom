//
// Created by Niu Chuang on 17-9-30.
//

#include <vector>
#include <cstdio>
#include <iostream>
using namespace std;
#include "caffe/layers/conv_td_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void pos_kernel(const int n, const Dtype* a, Dtype* b) {
  CUDA_KERNEL_LOOP(index, n) {
    if (a[index] > 0)
      b[index] = a[index];
  }
}

template <typename Dtype>
__global__ void div_r_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    if (b[index] != 0)
      y[index] = a[index] / b[index];
    else
      y[index] = 0;
  }
}

template<typename Dtype>
__global__ void Compute_wu(const int nthreads, const Dtype* const W,
		const int num, const int channels, const int height, const int width,
		Dtype* const Wu, const int u) {
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		const int kw = index % width;
		const int kh = (index / width) % height;
//		const int c = (index / width / height) % channels;
		const int n = index / width / height / channels;

		int wu_index = n*channels*height*width + u*height*width + kh*width + kw;
		Wu[index] = W[index] * W[wu_index];
	}
}

template<typename Dtype>
__global__ void sum_substract(const int nthreads, const Dtype* const buff_data,
		const int num, const int channels, const int height, const int width,
		Dtype* const activation_diff, const int u) {
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		const int w = index % width;
		const int h = (index / width) % height;
//		const int c = (index / width / height) % channels;
		const int n = index / width / height / channels;

		int au_index = n*channels*height*width + u*height*width + h*width + w;
		activation_diff[au_index] -= buff_data[index];
	}
}

template<typename Dtype>
__global__ void Substract(const int nthreads_u, const Dtype* const buff_data,
		const int num, const int channels, const int height, const int width,
		Dtype* const activation_diff, const int u) {
	CUDA_KERNEL_LOOP(index, nthreads_u)
	{
		const int w = index % width;
		const int h = (index / width) % height;
//		const int c = (index / width / height) % channels;
		const int n = index / width / height;

		Dtype* activation_diff_u = activation_diff + n*channels*height*width + u*height*width + h*width + w;
		for (int c = 0; c < channels; ++c){
			int ac_index = n*channels*height*width + c*height*width + h*width + w;
			activation_diff_u[0] -= buff_data[ac_index];
		}
	}
}

template<typename Dtype>
void ConvolutionTDLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	// get the new weight W+
	const Dtype* W_data = this->blobs_[0]->gpu_data();
	Blob<Dtype> W_plus(this->blobs_[0]->shape());
	Dtype* W_plus_data = W_plus.mutable_gpu_data();
	caffe_gpu_set<Dtype>(W_plus.count(), Dtype(0), W_plus_data);
	pos_kernel<Dtype> <<<CAFFE_GET_BLOCKS(W_plus.count()),
			CAFFE_CUDA_NUM_THREADS>>>(W_plus.count(), W_data, W_plus_data);

//	Blob<Dtype> NN(bottom[0]->shape());
//	Dtype* NN_data = NN.mutable_gpu_data();
	NF_.Reshape(bottom[0]->shape());
	NN_.Reshape(bottom[0]->shape());
	Dtype* NN_data = NN_.mutable_gpu_data();
	Dtype* NF_data = NF_.mutable_gpu_data();
	for (int i = 0; i < top.size(); ++i) {
		// do forward to compute the normalization factor by forwardpassing using W+
		const Dtype* activation_data = bottom[1]->gpu_data();
		for (int n = 0; n < this->num_; ++n) {
			this->forward_gpu_gemm(activation_data + n * this->top_dim_,
					W_plus_data, NF_data + n * this->bottom_dim_);
		}
		// do normalization
		const Dtype* bottom_data = bottom[0]->mutable_gpu_data();
		div_r_kernel<Dtype><<<CAFFE_GET_BLOCKS(NN_.count()), CAFFE_CUDA_NUM_THREADS>>>(
		            NN_.count(), bottom_data, NF_data, NN_data);

		// do backward
		Dtype* top_data = top[i]->mutable_gpu_data();
		for (int n = 0; n < this->num_; ++n) {
			this->backward_gpu_gemm(NN_data + n * this->bottom_dim_,
					W_plus_data, top_data + n * this->top_dim_);
			if (this->bias_term_) {
				const Dtype* bias = this->blobs_[1]->gpu_data();
				this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
			}
		}

		// multiply the bottom data
		caffe_gpu_mul<Dtype>(bottom[1]->count(), top_data, activation_data,
				top_data);
	}

}

template <typename Dtype>
void ConvolutionTDLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	// Get W+
	const Dtype* W_data = this->blobs_[0]->gpu_data();
	Blob<Dtype> W_plus(this->blobs_[0]->shape());
	Dtype* W_plus_data = W_plus.mutable_gpu_data();
	caffe_gpu_set<Dtype>(W_plus.count(), Dtype(0), W_plus_data);
	pos_kernel<Dtype> <<<CAFFE_GET_BLOCKS(W_plus.count()),
			CAFFE_CUDA_NUM_THREADS>>>(W_plus.count(), W_data, W_plus_data);

	buff_.Reshape(top[0]->shape());
	Dtype* buff_data = buff_.mutable_gpu_data();
	const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* activation_data = bottom[1]->gpu_data();
	const Dtype* NF_data = NF_.gpu_data();
	const Dtype* NN_data = NN_.gpu_data();

	if (propagate_down[0]) {
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

		// Multiply top diff with activation data
		caffe_gpu_mul<Dtype>(top[0]->count(), top_diff, activation_data, buff_data);

		// Do backward
		for (int n = 0; n < this->num_; ++n) {
			this->forward_gpu_gemm(buff_data + n * this->top_dim_, W_plus_data,
					bottom_diff + n * this->bottom_dim_);
		}

		// Multiply normalization data
		//            caffe_mul<Dtype>(bottom[0]->count(), bottom_diff, NF_data, bottom_diff);
		div_r_kernel<Dtype> <<<CAFFE_GET_BLOCKS(NN_.count()),
				CAFFE_CUDA_NUM_THREADS>>>(NN_.count(), bottom[0]->gpu_diff(), NF_data,
				bottom_diff);
	}

	if (propagate_down[1]) {

		Dtype* activation_diff = bottom[1]->mutable_gpu_diff();

		// Do forward
		for (int n = 0; n < this->num_; ++n) {
			this->backward_gpu_gemm(NN_data + n * this->bottom_dim_,
					W_plus_data, activation_diff + n * this->top_dim_);
			if (this->bias_term_) {
				const Dtype* bias = this->blobs_[1]->cpu_data();
				this->forward_gpu_bias(activation_diff + n * this->top_dim_,
						bias);
			}
		}

		// Multiply activation diff with top diff
		caffe_gpu_mul<Dtype>(bottom[1]->count(), activation_diff, top_diff,
				    activation_diff);

//		cout << "first term: " << endl;
//		print_4darray(bottom[1]->cpu_diff(), bottom[1]->shape());

		// Compute the second term and subtract the second term from the activation diff

		// Compute normalization data2

		Dtype* NN_data2 = NN_.mutable_gpu_data();
		div_r_kernel<Dtype> <<<CAFFE_GET_BLOCKS(NN_.count()),
				CAFFE_CUDA_NUM_THREADS>>>(NN_.count(), NN_.gpu_data(),
				NF_data, NN_data2);

//		cout << "NN data 2: " << endl;
//		print_4darray(NN_.cpu_data(), NN_.shape());

		// Compute Wu and
		//            cout << "output num: " << this->blobs_[0]->num() << endl;
		//            cout << "input num: " << this->blobs_[0]->channels() << endl;
		//            cout << "height: " << this->blobs_[0]->height();
		//            cout << "width: " << this-> blobs_[0]->width();
		Blob<Dtype> Wu(W_plus.shape());
		const Dtype* W_plus_data_c = W_plus.gpu_data();
		Dtype* Wu_data = Wu.mutable_cpu_data();
		int height = top[0]->shape()[2];
		int width = top[0]->shape()[3];

//		vector<int> sn_shape(bottom[1]->shape());
//		sn_shape[1] = 1;
//		Blob<Dtype> Sn(sn_shape);
//		Dtype* sn_data = Sn.gpu_data();

		for (int u = 0; u < top[0]->channels(); u++) {

			// Compute Wu
			Compute_wu<Dtype> <<<CAFFE_GET_BLOCKS(W_plus.count()),
					CAFFE_CUDA_NUM_THREADS>>>(W_plus.count(), W_plus_data_c,
					W_plus.num(), W_plus.channels(), W_plus.height(),
					W_plus.width(), Wu_data, u);

//			cout << "Wu: " << endl;
//			print_4darray(Wu.cpu_data(), Wu.shape());

			// Do forward
			for (int n = 0; n < this->num_; ++n) {
				this->backward_gpu_gemm(NN_data2 + n * this->bottom_dim_,
						Wu_data, buff_data + n * this->top_dim_);
				if (this->bias_term_) {
					const Dtype* bias = this->blobs_[1]->cpu_data();
					this->forward_gpu_bias(buff_data + n * this->top_dim_,
							bias);
				}
			}

//			cout << "buff data: " << endl;
//			print_4darray(buff_.cpu_data(), buff_.shape());

			// Multiply top diff and activation data
			caffe_gpu_mul<Dtype>(top[0]->count(), buff_.gpu_data(), top_diff, buff_.mutable_gpu_data());
			caffe_gpu_mul<Dtype>(top[0]->count(), buff_.gpu_data(), activation_data, buff_.mutable_gpu_data());

//			cout << "buff data after multiplied: " << endl;
//			print_4darray(buff_.cpu_data(), buff_.shape());

			// Compute the sum along the output number dimension and subtract the sum from the first term
			Substract<Dtype> <<<CAFFE_GET_BLOCKS(bottom[1]->count()/bottom[1]->channels()),
					CAFFE_CUDA_NUM_THREADS>>>(bottom[1]->count()/bottom[1]->channels(), buff_.gpu_data(),
							bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width(),
							bottom[1]->mutable_gpu_diff(), u);

//			cout << "activation_diff u: " <<  endl;
//			print_4darray(bottom[1]->cpu_diff(), bottom[1]->shape());
		}

	}

	if (this->param_propagate_down_[0]) {
		Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
		caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionTDLayer);

}  // namespace caffe

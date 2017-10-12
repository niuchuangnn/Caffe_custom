//
// Created by Niu Chuang on 17-9-29.
//


#include <vector>
#include <cstdio>
#include <iostream>
using namespace std;
#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_td_layer.hpp"
#include "caffe/util/math_functions.hpp"

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

template <typename Dtype>
void InnerProductTDLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  // get the new weight W+
      const Dtype* W_data = this->blobs_[0]->gpu_data();
      Blob<Dtype> W_plus(this->blobs_[0]->shape());
      Dtype* W_plus_data = W_plus.mutable_gpu_data();
      caffe_gpu_set<Dtype>(W_plus.count(), Dtype(0), W_plus_data);
      pos_kernel<Dtype><<<CAFFE_GET_BLOCKS(W_plus.count()), CAFFE_CUDA_NUM_THREADS>>>(
            W_plus.count(), W_data, W_plus_data);

      // compute the normalization factor by forward passing using W+
      Dtype* NN_data = NN_.mutable_gpu_data();
	  Dtype* NF_data = NF_.mutable_gpu_data();
      const Dtype* activation_data = bottom[1]->gpu_data();

      if (M_ == 1) {
        caffe_gpu_gemv<Dtype>(CblasTrans, N_, K_, (Dtype)1.,
                             W_plus_data, activation_data, (Dtype)0., NF_data);
      } else {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                             M_, K_, N_, (Dtype)1.,
                             activation_data, W_plus_data, (Dtype)0., NF_data);
      }

      // do normalization
      const Dtype* bottom_data = bottom[0]->gpu_data();
      div_r_kernel<Dtype><<<CAFFE_GET_BLOCKS(NN_.count()), CAFFE_CUDA_NUM_THREADS>>>(
            NN_.count(), bottom_data, NF_data, NN_data);

      // do backward pass
      Dtype* top_data = top[0]->mutable_gpu_data();
      if (transpose_) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            M_, N_, K_,
            (Dtype)1., NN_data, W_plus_data,
            (Dtype)0., top_data);
      } else {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
            M_, N_, K_,
           (Dtype)1., NN_data, W_plus_data,
           (Dtype)0., top_data);
      }

      // multiply the bottom data
      caffe_gpu_mul<Dtype>(top[0]->count(), top[0]->gpu_data(), activation_data, top_data);

}

template <typename Dtype>
void InnerProductTDLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		// Gradient with respect to bottom data

		// Multiply G_{n-1} with A_{n-1}
		const Dtype* top_diff = top[0]->gpu_diff();
		const Dtype* activation_data = bottom[1]->gpu_data();

		Dtype* buff_data = buff_.mutable_gpu_data();
		caffe_gpu_mul<Dtype>(buff_.count(), activation_data, top_diff,
				buff_data);

		// get the new weight W+
		const Dtype* W_data = this->blobs_[0]->gpu_data();
		Blob<Dtype> W_plus(this->blobs_[0]->shape());
		Dtype* W_plus_data = W_plus.mutable_gpu_data();

		caffe_gpu_set<Dtype>(W_plus.count(), Dtype(0), W_plus_data);
		pos_kernel<Dtype> <<<CAFFE_GET_BLOCKS(W_plus.count()),
				CAFFE_CUDA_NUM_THREADS>>>(W_plus.count(), W_data, W_plus_data);

		// do backward
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		if (M_ == 1) {
			caffe_gpu_gemv<Dtype>(CblasTrans, N_, K_, (Dtype) 1., W_plus_data,
					buff_data, (Dtype) 0., bottom_diff);
		} else {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_,
					(Dtype) 1., buff_data, W_plus_data, (Dtype) 0.,
					bottom_diff);
		}

		// Normalization
		// Get normalization data
		const Dtype* NF_data = NF_.gpu_data();

		div_r_kernel<Dtype> <<<CAFFE_GET_BLOCKS(NF_.count()),
				CAFFE_CUDA_NUM_THREADS>>>(NF_.count(), bottom[0]->gpu_diff(), NF_data,
				bottom[0]->mutable_gpu_diff());
	}

	if (propagate_down[1]) {
		// Normalization
		// Get normalization data
		const Dtype* NN_data = NN_.gpu_data();

		// get the new weight W+
		const Dtype* W_data = this->blobs_[0]->gpu_data();
		Blob<Dtype> W_plus(this->blobs_[0]->shape());
		Dtype* W_plus_data = W_plus.mutable_gpu_data();
		caffe_gpu_set<Dtype>(W_plus.count(), Dtype(0), W_plus_data);
		pos_kernel<Dtype> <<<CAFFE_GET_BLOCKS(W_plus.count()),
				CAFFE_CUDA_NUM_THREADS>>>(W_plus.count(), W_data, W_plus_data);

		// Do forward
		Dtype* activation_diff = bottom[1]->mutable_gpu_diff();

		if (transpose_) {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
					(Dtype) 1., NN_data, W_plus_data, (Dtype) 0., activation_diff);
		} else {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
					(Dtype) 1., NN_data, W_plus_data, (Dtype) 0., activation_diff);
		}

		// Multiply top diff
		const Dtype* top_diff = top[0]->gpu_diff();
        caffe_gpu_mul<Dtype>(top[0]->count(), activation_diff, top_diff, activation_diff);


        // Compute the second term and subtract the second term from the activation diff

		// Compute P_{n} / N_{n}^{2}
		Dtype* NN_data2 = NN_.mutable_gpu_data();
		const Dtype* NF_data = NF_.gpu_data();
		div_r_kernel<Dtype> <<<CAFFE_GET_BLOCKS(NN_.count()),
				CAFFE_CUDA_NUM_THREADS>>>(NN_.count(), NN_data2, NF_data,
				NN_data2);

		// Compute W_{u}
		const Dtype* W_plus_data_c = W_plus.gpu_data();
		Blob<Dtype> Wu(W_plus.shape());
		Dtype* Wu_data = Wu.mutable_gpu_data();
		Dtype* buff_data = buff_.mutable_gpu_data();

		Dtype* c_data = C_.mutable_gpu_data();
		caffe_gpu_set<Dtype>(C_.count(), (Dtype) 1., c_data);
		Dtype* cm_data = CM_.mutable_gpu_data();
		caffe_gpu_set<Dtype>(CM_.count(), (Dtype) 1., cm_data);
		Dtype* sr_data = SR_.mutable_gpu_data();

		for (int u = 0; u < N_; ++u) {
			const Dtype* W_plus_data_c_u = W_plus_data_c + K_ * u;

			// Compute Wu
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, K_, 1,
					(Dtype) 1., c_data, W_plus_data_c_u, (Dtype) 0., Wu_data);
			caffe_gpu_mul<Dtype>(Wu.count(), Wu_data, W_plus_data_c, Wu_data);

			// Do forward
			caffe_gpu_gemm<Dtype>(CblasNoTrans,
					transpose_ ? CblasNoTrans : CblasTrans, M_, N_, K_,
					(Dtype) 1., NN_data2, Wu_data, (Dtype) 0., buff_data);

			// Multiply top diff and activation data
			const Dtype* activation_data = bottom[1]->gpu_data();
			caffe_gpu_mul<Dtype>(buff_.count(), buff_data, top_diff, buff_data);
			caffe_gpu_mul<Dtype>(buff_.count(), buff_data, activation_data,
					    buff_data);

			// Compute the sum along the feature dimension
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, 1, N_,
					(Dtype) 1., buff_data, c_data, (Dtype) 0., cm_data);
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
					(Dtype) 1., cm_data, c_data, (Dtype) 0., buff_data);
			// Subtract the sum from the first term along the batch size dimension
			caffe_gpu_set<Dtype>(SR_.count(), (Dtype) 0, sr_data);
			Dtype* sr_data_u = sr_data + u * N_ + u;
			caffe_gpu_set<Dtype>(1, (Dtype) 1, sr_data_u);
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, N_,
					Dtype(-1.), buff_data, sr_data, (Dtype) 1.,
					activation_diff);
		}

	}

	if (this->param_propagate_down_[0]) {
		caffe_gpu_set<Dtype>(this->blobs_[0]->count(), Dtype(0),
				this->blobs_[0]->mutable_gpu_diff());
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductTDLayer);

}  // namespace caffe

//
// Created by Niu Chuang on 17-9-30.
//

#include <cfloat>
#include <vector>

#include "caffe/layers/eltwise_td_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SumBackward(const int nthreads, const Dtype* bottom_data,
    const Dtype* bottom_data_a, const Dtype* out_data, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    top_data[index] = out_data[index] == 0 ? Dtype(0):bottom_data[index]*bottom_data_a[index]/out_data[index];
  }
}

template <typename Dtype>
void EltwiseTDLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
//  int* mask = NULL;
//  const int count = top[0]->count();
//  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* out_data = bottom[1]->gpu_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
	 NOT_IMPLEMENTED;
    break;
  case EltwiseParameter_EltwiseOp_SUM:
//    caffe_gpu_set(count, Dtype(0.), top_data);
    // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
//    for (int i = 0; i < bottom.size(); ++i) {
//      caffe_gpu_axpy(count, coeffs_[i], bottom[i]->gpu_data(), top_data);
//    }
	  for (int i = 0; i < top.size(); ++i){
		  const int count = top[i]->count();
	      Dtype* top_data = top[i]->mutable_gpu_data();
	      const Dtype* bottom_data_a = bottom[i+2]->gpu_data();
	      SumBackward<Dtype>
	      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	      count, bottom_data, bottom_data_a, out_data, top_data);
//	      for (int j = 0; j < count; ++j){
//	          top_data[j] = out_data[j] == 0 ? Dtype(0):bottom_data_a[j]*bottom_data[j]/out_data[j];
//	      }
	  }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
	NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

template <typename Dtype>
__global__ void Backward_all(const int nthreads, const Dtype* top_diff_a, const Dtype* top_diff_b,
    const Dtype* activation_data_a, const Dtype* activation_data_b, const Dtype* bottom_data,
    const Dtype* out_data, Dtype* bottom_diff, Dtype* activation_diff_a, Dtype* activation_diff_b) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    bottom_diff[index] = out_data[index] == 0 ? Dtype(0):(top_diff_a[index]*activation_data_a[index] + top_diff_b[index]*activation_data_b[index])/out_data[index];
    activation_diff_a[index] = out_data[index] == 0 ? Dtype(0):(top_diff_a[index] - top_diff_b[index]) * bottom_data[index] * activation_data_b[index] / (out_data[index] * out_data[index]);
    activation_diff_b[index] = out_data[index] == 0 ? Dtype(0):(top_diff_b[index] - top_diff_a[index]) * bottom_data[index] * activation_data_a[index] / (out_data[index] * out_data[index]);
  }
}

template <typename Dtype>
void EltwiseTDLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	if (!(propagate_down[0] || propagate_down[2] || propagate_down[3])) {
	    return;
	}

	const Dtype* top_diff_a = top[0]->gpu_diff();
	const Dtype* top_diff_b = top[1]->gpu_diff();
	const Dtype* out_data = bottom[1]->gpu_data();
	const Dtype* activation_data_a = bottom[2]->gpu_data();
	const Dtype* activation_data_b = bottom[3]->gpu_data();
	Dtype* activation_diff_a = bottom[2]->mutable_gpu_diff();
	Dtype* activation_diff_b = bottom[3]->mutable_gpu_diff();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const int count = bottom[0]->count();
	switch (op_) {
	case EltwiseParameter_EltwiseOp_PROD:
		NOT_IMPLEMENTED;
		break;
	case EltwiseParameter_EltwiseOp_SUM:
		Backward_all<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, top_diff_a, top_diff_b, activation_data_a,
				activation_data_b, bottom_data, out_data, bottom_diff,
				activation_diff_a, activation_diff_b);

		break;
	case EltwiseParameter_EltwiseOp_MAX:
		NOT_IMPLEMENTED;
		break;
	default:
		LOG(FATAL)<< "Unknown elementwise operation.";
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(EltwiseTDLayer);

}  // namespace caffe

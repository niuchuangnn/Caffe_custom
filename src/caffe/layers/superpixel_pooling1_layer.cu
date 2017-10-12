#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/superpixel_pooling1_layer.hpp"
#include "caffe/util/math_functions.hpp"
//#include "cuda_runtime.h"

namespace caffe {

template <typename Dtype>
__global__ void AvePoolForward1(int nthreads, const Dtype* bottom_data, const int channels,
		const int sp_num, const int* sp_label_nums, const int height, const int width,
		Dtype* sp_label_weight, const Dtype* sp_fea_maps, Dtype* result_cp) {
  CUDA_KERNEL_LOOP(index, nthreads) {
//	extern __shared__ float ss[];
//	const int p = threadIdx.x;
//	const int c = blockIdx.x;
	const int p = index % sp_num;
	const int c = index / sp_num;
//	Dtype x = 0;
	for (int h = 0; h < height; h++){
		for (int w = 0; w < width; w++){
			result_cp[c*sp_num + p] += ((sp_fea_maps[p*height*width + h*width+w] * bottom_data[c*height*width + h*width+w] * sp_label_weight[p]) / sp_label_nums[p]);
//			ss[p] += (sp_fea_maps[p*height*width + h*width+w] * bottom_data[c*height*width + h*width+w] * sp_label_weight[p]);
//			x += (sp_fea_maps[p*height*width + h*width+w] * bottom_data[h*width+w] * sp_label_weight[p]) / sp_label_nums[p]);
		}
	}

//	__syncthreads();
//	ss[0] /= sp_label_nums[0];
//	for (int i = 1; i < sp_num; i++){
//		ss[0] += (ss[i]/sp_label_nums[i]);
//	}
//	top_data[c] = ss[0] / sp_num;


  }
}

template <typename Dtype>
void SuperpixelPooling1Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* top_data_cpu = top[0]->mutable_cpu_data();
  switch (this->layer_param_.superpixel_pooling1_param().pool()) {
  case SuperpixelPooling1Parameter_PoolMethod_MAX:
	  NOT_IMPLEMENTED;
	  break;
  case SuperpixelPooling1Parameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
	  for (int n = 0; n < batch_size_; n++){
		  const int* sp_label_nums_n = sp_label_nums_[n]->gpu_data();
		  const Dtype* sp_fea_maps_n = sp_fea_maps_[n]->gpu_data();
		  const int sp_nums_n = sp_nums_[n];
		  sp_label_weight_.clear();
		  if (sp_label_ == -1){
			  sp_label_weight_.resize(sp_nums_n, 1);
		  } else {
			  sp_label_weight_.resize(sp_nums_n, 0);
			  sp_label_weight_[sp_label_] = 1;
		  }
		  Dtype* sp_label_weight;
		  int nthreads = channels_*sp_nums_n;
		  Dtype* result_cp;
		  cudaMalloc((void**) &sp_label_weight, sizeof(Dtype)*sp_label_weight_.size());
		  cudaMemcpy(sp_label_weight, &sp_label_weight_[0], sizeof(Dtype)*sp_nums_n, cudaMemcpyHostToDevice);
		  cudaMalloc((void**) &result_cp, sizeof(Dtype)*nthreads);
		  cudaMemset(result_cp, 0, sizeof(Dtype)*nthreads);
		  AvePoolForward1<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
		          nthreads, bottom_data, channels_, sp_nums_n, sp_label_nums_n,
		          height_, width_, sp_label_weight, sp_fea_maps_n, result_cp);
//		  CHECK_LT(sp_nums_n, 1024) << "sp_num must be less than 1024.";
//		  AvePoolForward1<Dtype><<<channels_, sp_nums_n, sizeof(Dtype)*sp_nums_n>>>(
//		  		          bottom_data, channels_, sp_nums_n, sp_label_nums_n,
//		  		          height_, width_, sp_label_weight, sp_fea_maps_n, top_data);

		  Dtype* result_cp_cpu = (Dtype* ) malloc(sizeof(Dtype)*nthreads);
		  cudaMemcpy(result_cp_cpu, result_cp, sizeof(Dtype)*nthreads, cudaMemcpyDeviceToHost);

		  for (int c = 0; c < channels_; c++){
			  Dtype o_c = 0;
			  for (int p = 0; p < sp_nums_n; p++){
				  o_c += result_cp_cpu[c*sp_nums_n+p];
			  }

//			  cudaMemset(&top_data[c], o_c/sp_nums_n, sizeof(Dtype));
			  top_data_cpu[c] = o_c / sp_nums_n;
		  }

		  cudaFree(result_cp);
		  cudaFree(sp_label_weight);
		  free(result_cp_cpu);

		  top_data += top[0]->offset(1,0,0,0);
		  bottom_data += bottom[0]->offset(1,0,0,0);
		  top_data_cpu += top[0]->offset(1,0,0,0);
	  }

    break;
  case SuperpixelPooling1Parameter_PoolMethod_STOCHASTIC:
	  NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void AvePoolBackward1(const int nthreads, const Dtype* const top_diff,
    const int channels, const int height, const int width, const int sp_num,
    const int* sp_label_nums, const Dtype* sp_fea_maps, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = index / width / height;

    Dtype weight_hw = 0;
    for (int p = 0; p < sp_num; ++p){
        weight_hw += (sp_fea_maps[p*height*width + h*width+w]) / ((Dtype) sp_label_nums[p]);
    }
    bottom_diff[index] += (top_diff[c] * weight_hw);
  }
}


template <typename Dtype>
void SuperpixelPooling1Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  switch (this->layer_param_.superpixel_pooling1_param().pool()) {
  case SuperpixelPoolingParameter_PoolMethod_MAX:
	NOT_IMPLEMENTED;
    break;
  case SuperpixelPooling1Parameter_PoolMethod_AVE:
	  for (int n = 0; n < batch_size_; n++){
	  		  const int* sp_label_nums_n = sp_label_nums_[n]->gpu_data();
	  		  const Dtype* sp_fea_maps_n = sp_fea_maps_[n]->gpu_data();
	  		  const int sp_nums_n = sp_nums_[n];
//	  		  CHECK_LT(sp_nums_n, 1024) << "sp_num must be less than 1024.";
	  		AvePoolBackward1<Dtype><<<CAFFE_GET_BLOCKS(count/batch_size_), CAFFE_CUDA_NUM_THREADS>>>(
	  		        count/batch_size_, top_diff, channels_,
	  		        height_, width_, sp_nums_n, sp_label_nums_n, sp_fea_maps_n, bottom_diff);
	  		  top_diff += top[0]->offset(1,0,0,0);
	  		  bottom_diff += bottom[0]->offset(1,0,0,0);
	  	  }
    break;
  case SuperpixelPooling1Parameter_PoolMethod_STOCHASTIC:
	NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(SuperpixelPooling1Layer);


}  // namespace caffe

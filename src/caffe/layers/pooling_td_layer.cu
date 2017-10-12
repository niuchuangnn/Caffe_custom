//
// Created by Niu Chuang on 17-9-29.
//

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_td_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void AvePoolForward(const int nthreads, const Dtype* const bottom_data,
	const Dtype* const activation_data, const Dtype* const pooled_data,
    const int num, const int channels, const int unpooled_height,
    const int unpooled_width, const int height, const int width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % unpooled_width + pad_w;
    const int h = (index / unpooled_width) % unpooled_height + pad_h;
    const int c = (index / unpooled_width / unpooled_height) % channels;
    const int n = index / unpooled_width / unpooled_height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, width);
    Dtype top = 0;
    const Dtype* const bottom_data_slice =
        bottom_data + (n * channels + c) * height * width;
    const Dtype* const pooled_data_slice =
    	pooled_data + (n * channels + c) * height * width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, unpooled_height + pad_h);
        int wend = min(wstart + kernel_w, unpooled_width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
//        gradient += bottom_data_slice[ph * width + pw] / pool_size;
        Dtype bsum = pooled_data_slice[ph * width + pw] * pool_size;
        top += bsum > Dtype(0) ?
        		(bottom_data_slice[ph * width + pw] * activation_data[index] / bsum):Dtype(0);
      }
    }
    top_data[index] = top;
  }
}


template <typename Dtype>
__global__ void AvePoolBackwardEB(const int nthreads, const Dtype* const top_diff,
    const Dtype* const tsum_data, const Dtype* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const tsum_slice =
        tsum_data + (n * channels + c) * pooled_height * pooled_width;

    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        Dtype ts = tsum_slice[ph * pooled_width + pw];
        gradient += ts > Dtype(0) ?
            (top_diff_slice[ph * pooled_width + pw] * bottom_data[index] / ts):Dtype(0);
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void PoolingTDLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  caffe_gpu_set(count, Dtype(0.), top_data);
  const Dtype* activation_data = bottom[1]->gpu_data();
  const Dtype* pooled_data = bottom[2]->gpu_data();
  // We'll output the mask to top[1] if it's of size >1.
//  const bool use_top_mask = top.size() > 1;
//  int* mask = NULL;
//  Dtype* top_mask = NULL;
  switch (this->layer_param_.unpooling_param().unpool()) {
  case UnpoolingParameter_UnpoolMethod_MAX:
	  NOT_IMPLEMENTED;
    break;
  case UnpoolingParameter_UnpoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
	  AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	          count, bottom_data, activation_data, pooled_data, bottom[0]->num(), channels_,
	          unpooled_height_, unpooled_width_, height_, width_, kernel_h_,
	          kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
    break;
  default:
    LOG(FATAL) << "Unknown unpooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* const top_diff,
		const Dtype* const bottom_data, const Dtype* const activation_data,
		const Dtype* const pooled_data, const int num, const int channels,
		const int unpooled_height, const int unpooled_width, const int height,
		const int width, const int kernel_h, const int kernel_w,
		const int stride_h, const int stride_w, const int pad_h,
		const int pad_w, Dtype* const bottom_diff,
		Dtype* const activation_diff) {
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		// find out the local index
		// find out the local offset
		const int pw = index % width;
		const int ph = (index / width) % height;
		const int c = (index / width / height) % channels;
		const int n = index / width / height / channels;
		int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;
		int hend = min(hstart + kernel_h, unpooled_height + pad_h);
		int wend = min(wstart + kernel_w, unpooled_width + pad_w);
		const int pool_size = (hend - hstart) * (wend - wstart);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		hend = min(hend, unpooled_height);
		wend = min(wend, unpooled_width);
		const Dtype* const activation_data_slice = activation_data
				+ (n * channels + c) * unpooled_height * unpooled_width;
		const Dtype* const top_diff_slice = top_diff
				+ (n * channels + c) * unpooled_height * unpooled_width;
		Dtype bsum = pooled_data[index] * pool_size;
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				bottom_diff[index] +=
						bsum > Dtype(0) ?
								(top_diff_slice[h*unpooled_width + w]
										* activation_data_slice[h*unpooled_width + w] / bsum) :
								Dtype(0);
			}
		}

		Dtype* const activation_diff_slice = activation_diff
				+ (n * channels + c) * unpooled_height * unpooled_width;
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				activation_diff_slice[h * unpooled_width + w] += bsum > Dtype(0) ?
								(top_diff_slice[h * unpooled_width + w] - bottom_diff[index])
								* (bottom_data[index] / bsum) : Dtype(0);
			}
		}
	}
}


template<typename Dtype>
void PoolingTDLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	if (!(propagate_down[0] || propagate_down[1])) {
		return;
	}

	const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	Dtype* activation_diff = bottom[1]->mutable_gpu_diff();
	const Dtype* activation_data = bottom[1]->gpu_data();
	const Dtype* pooled_data = bottom[2]->gpu_data();
	// Different unpooling methods. We explicitly do the switch outside the for
	// loop to save time, although this results in more codes.
	caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
	caffe_gpu_set(bottom[1]->count(), Dtype(0), activation_diff);
	const int count = bottom[0] -> count();
	switch (this->layer_param_.unpooling_param().unpool()) {
	case UnpoolingParameter_UnpoolMethod_MAX:
		NOT_IMPLEMENTED;
		break;
	case UnpoolingParameter_UnpoolMethod_AVE:
		// NOLINT_NEXT_LINE(whitespace/operators)
		AvePoolBackward<Dtype> <<<CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, bottom_data, activation_data,
						pooled_data, bottom[0]->num(), channels_, unpooled_height_,
						unpooled_width_, height_, width_, kernel_h_, kernel_w_, stride_h_,
						stride_w_, pad_h_, pad_w_, bottom_diff, activation_diff);
		break;
	default:
		LOG(FATAL)<< "Unknown pooling method.";
	}
	CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(PoolingTDLayer);


}  // namespace caffe

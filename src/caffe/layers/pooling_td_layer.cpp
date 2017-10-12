//
// Created by Niu Chuang on 17-9-26.
//

#include <algorithm>
#include <cfloat>
#include <vector>
#include <cstdio>
#include <iostream>
using namespace std;
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/pooling_td_layer.hpp"

namespace caffe {

    using std::min;
    using std::max;

    template <typename Dtype>
    void PoolingTDLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
        UnpoolingParameter unpool_param = this->layer_param_.unpooling_param();

        CHECK(!unpool_param.has_kernel_size() !=
              !(unpool_param.has_kernel_h() && unpool_param.has_kernel_w()))
        << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
        CHECK(unpool_param.has_kernel_size() ||
              (unpool_param.has_kernel_h() && unpool_param.has_kernel_w()))
        << "For non-square filters both kernel_h and kernel_w are required.";
        CHECK((!unpool_param.has_pad() && unpool_param.has_pad_h()
               && unpool_param.has_pad_w())
              || (!unpool_param.has_pad_h() && !unpool_param.has_pad_w()))
        << "pad is pad OR pad_h and pad_w are required.";
        CHECK((!unpool_param.has_stride() && unpool_param.has_stride_h()
               && unpool_param.has_stride_w())
              || (!unpool_param.has_stride_h() && !unpool_param.has_stride_w()))
        << "Stride is stride OR stride_h and stride_w are required.";
        CHECK((!unpool_param.has_unpool_size() && unpool_param.has_unpool_h()
               && unpool_param.has_unpool_w())
              || (unpool_param.has_unpool_size() &&!unpool_param.has_unpool_h()
                  && !unpool_param.has_unpool_w()))
        << "Unpool is unpool_size OR unpool_h and unpool_w are required.";

        if (unpool_param.has_kernel_size()) {
            kernel_h_ = kernel_w_ = unpool_param.kernel_size();
        } else {
            kernel_h_ = unpool_param.kernel_h();
            kernel_w_ = unpool_param.kernel_w();
        }
        CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
        CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";

        if (!unpool_param.has_pad_h()) {
            pad_h_ = pad_w_ = unpool_param.pad();
        } else {
            pad_h_ = unpool_param.pad_h();
            pad_w_ = unpool_param.pad_w();
        }
        CHECK_EQ(pad_h_, 0) << "currently, only zero padding is allowed.";
        CHECK_EQ(pad_w_, 0) << "currently, only zero padding is allowed.";
        if (!unpool_param.has_stride_h()) {
            stride_h_ = stride_w_ = unpool_param.stride();
        } else {
            stride_h_ = unpool_param.stride_h();
            stride_w_ = unpool_param.stride_w();
        }
        if (pad_h_ != 0 || pad_w_ != 0) {
            CHECK(UnpoolingParameter_UnpoolMethod_MAX == unpool_param.unpool())
            << "Padding implemented only for max unpooling.";
            CHECK_LT(pad_h_, kernel_h_);
            CHECK_LT(pad_w_, kernel_w_);
        }

        if (unpool_param.has_unpool_size()) {
            unpooled_height_ = unpooled_width_ = unpool_param.unpool_size();
        } else if (unpool_param.has_unpool_h() &&
                   unpool_param.has_unpool_w()) {
            unpooled_height_ = unpool_param.unpool_h();
            unpooled_width_ = unpool_param.unpool_w();
        } else {
            // in this case, you should recompute unpooled_width, height
            unpooled_height_ = -1;
            unpooled_width_ = -1;
        }
    }

    template <typename Dtype>
    void PoolingTDLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
        channels_ = bottom[0]->channels();
        height_ = bottom[0]->height();
        width_ = bottom[0]->width();

        if (unpooled_height_ < 0 || unpooled_width_ < 0) {
//            unpooled_height_ = (height_ - 1) * stride_h_ + kernel_h_ - 2 * pad_h_;
//            unpooled_width_ = (width_ - 1) * stride_w_ + kernel_w_ - 2 * pad_w_;
            unpooled_height_ = max((height_ - 1) * stride_h_ + kernel_h_ - 2 * pad_h_,
                                   height_ * stride_h_ - pad_h_ + 1);
            unpooled_width_ = max((width_ - 1) * stride_w_ + kernel_w_ - 2 * pad_w_,
                                  width_ * stride_w_ - pad_w_ + 1);
        }

        top[0]->Reshape(bottom[0]->num(), channels_, unpooled_height_,
                        unpooled_width_);

        // check bottom[1] and top shape
        vector<int> activation_shape;
        activation_shape = bottom[1]->shape();
        vector<int> top_shape;
        top_shape = top[0]->shape();
        for (int i=0; i < activation_shape.size(); ++i){
            CHECK_EQ(top_shape[i], activation_shape[i]) << "top shape must equal to bottom[1] shape";
        }
        vector<int> pooled_shape;
        pooled_shape = bottom[2]->shape();
        vector<int> bottom_shape;
        bottom_shape = bottom[0]->shape();
        for (int i=0; i < bottom_shape.size(); ++i){
            CHECK_EQ(pooled_shape[i], bottom_shape[i]) << "bottom[0] shape must equal to bottom[2] shape";
        }
    }

// TODO(Yangqing): Is there a faster way to do unpooling in the channel-first
// case?
    template <typename Dtype>
    void PoolingTDLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        const Dtype* activation_data = bottom[1]->cpu_data();
        const Dtype* pooled_data = bottom[2]->cpu_data();
        const int top_count = top[0]->count();
        caffe_set(top_count, Dtype(0), top_data);
        // We'll get the mask from bottom[1] if it's of size >1.
//        const bool use_bottom_mask = bottom.size() > 1;
//        const Dtype* bottom_mask = NULL;
        // Different unpooling methods. We explicitly do the switch outside the for
        // loop to save time, although this results in more code.
        switch (this->layer_param_.unpooling_param().unpool()) {
            case UnpoolingParameter_UnpoolMethod_MAX:
                NOT_IMPLEMENTED;
                break;
            case UnpoolingParameter_UnpoolMethod_AVE:
                // The main loop
                for (int n = 0; n < top[0]->num(); ++n) {
                    for (int c = 0; c < channels_; ++c) {
                        for (int ph = 0; ph < height_; ++ph) {
                            for (int pw = 0; pw < width_; ++pw) {
                                int hstart = ph * stride_h_ - pad_h_;
                                int wstart = pw * stride_w_ - pad_w_;
                                int hend = min(hstart + kernel_h_, unpooled_height_ + pad_h_);
                                int wend = min(wstart + kernel_w_, unpooled_width_ + pad_w_);
                                int pool_size = (hend - hstart) * (wend - wstart);
                                hstart = max(hstart, 0);
                                wstart = max(wstart, 0);
                                hend = min(hend, unpooled_height_);
                                wend = min(wend, unpooled_width_);
                                Dtype bsum = pooled_data[ph * width_ + pw] * pool_size;
                                for (int h = hstart; h < hend; ++h) {
                                    for (int w = wstart; w < wend; ++w) {
//                                        top_data[h * unpooled_width_ + w] +=
//                                                bottom_data[ph * width_ + pw] / pool_size;
                                        top_data[h * unpooled_width_ + w] += bsum > Dtype(0) ?
                                                                             (activation_data[h * unpooled_width_ + w] * bottom_data[ph * width_ + pw] / bsum):Dtype(0);
                                    }
                                }
                            }
                        }
                        // offset
                        bottom_data += bottom[0]->offset(0, 1);
                        top_data += top[0]->offset(0, 1);
                        pooled_data += bottom[2]->offset(0, 1);
                        activation_data += bottom[1]->offset(0, 1);
                    }
                }
                break;
            case UnpoolingParameter_UnpoolMethod_TILE:
                NOT_IMPLEMENTED;
                break;
            default:
                LOG(FATAL) << "Unknown unpooling method.";
        }
    }

    template <typename Dtype>
    void PoolingTDLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                             const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        if (!(propagate_down[0] || propagate_down[1])) {
            return;
        }
        const Dtype* top_diff = top[0]->cpu_diff();
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        Dtype* activation_diff = bottom[1]->mutable_cpu_diff();
        const Dtype* activation_data = bottom[1]->cpu_data();
        const Dtype* pooled_data = bottom[2]->cpu_data();
        // Different unpooling methods. We explicitly do the switch outside the for
        // loop to save time, although this results in more codes.
        caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
        caffe_set(bottom[1]->count(), Dtype(0), activation_diff);

        // We'll output the mask to top[1] if it's of size >1.
//        const bool use_bottom_mask = bottom.size() > 1;
//        const Dtype* bottom_mask = NULL;
        switch (this->layer_param_.unpooling_param().unpool()) {
            case UnpoolingParameter_UnpoolMethod_MAX:
                NOT_IMPLEMENTED;
                break;
            case UnpoolingParameter_UnpoolMethod_AVE:
                for (int i = 0; i < bottom[0]->count(); ++i) {
                    bottom_diff[i] = 0;
                }
                for (int i = 0; i < bottom[1]->count(); ++i) {
                    activation_diff[i] = 0;
                }
                // The main loop
                for (int n = 0; n < bottom[0]->num(); ++n) {
                    for (int c = 0; c < channels_; ++c) {
                        for (int ph = 0; ph < height_; ++ph) {
                            for (int pw = 0; pw < width_; ++pw) {
                                int hstart = ph * stride_h_ - pad_h_;
                                int wstart = pw * stride_w_ - pad_w_;
                                int hend = min(hstart + kernel_h_, unpooled_height_ + pad_h_);
                                int wend = min(wstart + kernel_w_, unpooled_width_ + pad_w_);
                                int pool_size = (hend - hstart) * (wend - wstart);
                                hstart = max(hstart, 0);
                                wstart = max(wstart, 0);
                                hend = min(hend, unpooled_height_);
                                wend = min(wend, unpooled_width_);
                                Dtype bsum = pooled_data[ph * width_ + pw] * pool_size;
                                for (int h = hstart; h < hend; ++h) {
                                    for (int w = wstart; w < wend; ++w) {
//                                        bottom_diff[ph * width_ + pw] +=
//                                                top_diff[h * unpooled_width_ + w];
                                        bottom_diff[ph * width_ + pw] += bsum > Dtype(0) ?
                                                                         (activation_data[h * unpooled_width_ + w] * top_diff[h * unpooled_width_ + w] / bsum) : Dtype(0);
                                    }
                                }

                                for (int h = hstart; h < hend; ++h){
                                    for (int w = wstart; w < wend; ++w){
                                        activation_diff[h * unpooled_width_ + w] += bsum > Dtype(0) ?
                                        		(top_diff[h*unpooled_width_ + w] - bottom_diff[ph * width_ + pw]) * (bottom_data[ph * width_ + pw] / bsum) : Dtype(0);
                                    }
                                }
//                                bottom_diff[ph * width_ + pw] /= pool_size;
                            }
                        }
                        // compute offset
                        bottom_diff += bottom[0]->offset(0, 1);
                        top_diff += top[0]->offset(0, 1);
                        bottom_data += bottom[0]->offset(0, 1);
                        pooled_data += bottom[2]->offset(0, 1);
                        activation_data += bottom[1]->offset(0, 1);
                        activation_diff += bottom[1]->offset(0, 1);
                    }
                }
                break;
            case UnpoolingParameter_UnpoolMethod_TILE:
                NOT_IMPLEMENTED;
                break;
            default:
                LOG(FATAL) << "Unknown unpooling method.";
        }
    }


#ifdef CPU_ONLY
    STUB_GPU(PoolingTDLayer);
#endif
    INSTANTIATE_CLASS(PoolingTDLayer);
    REGISTER_LAYER_CLASS(PoolingTD);


}  // namespace caffe

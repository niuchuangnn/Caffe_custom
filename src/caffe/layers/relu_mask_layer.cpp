//
// Created by Niu Chuang on 17-7-31.
//

#include <algorithm>
#include <vector>

#include "caffe/layers/relu_mask_layer.hpp"

namespace caffe {

    template <typename Dtype>
    void ReLUMaskLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
        top[0]->ReshapeLike(*bottom[0]);

        if (top.size() > 1){
            top[1]->ReshapeLike(*top[0]);
        }
    }

    template <typename Dtype>
    void ReLUMaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        const int count = bottom[0]->count();
        const bool use_top_mask = top.size() > 1;
        Dtype* top_mask = NULL;
        Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
        if (use_top_mask) {
            top_mask = top[1]->mutable_cpu_data();
            caffe_set(count, Dtype(0), top_mask);
        }
        for (int i = 0; i < count; ++i) {
            top_data[i] = std::max(bottom_data[i], Dtype(0))
                          + negative_slope * std::min(bottom_data[i], Dtype(0));
            if (use_top_mask){
                top_mask[i] = (bottom_data[i] > 0 ? 1 : 0);
            }
        }
    }

    template <typename Dtype>
    void ReLUMaskLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom) {
        if (propagate_down[0]) {
            const Dtype* bottom_data = bottom[0]->cpu_data();
            const Dtype* top_diff = top[0]->cpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            const int count = bottom[0]->count();
            Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
            for (int i = 0; i < count; ++i) {
                bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
                                                + negative_slope * (bottom_data[i] <= 0));
            }
        }
    }


#ifdef CPU_ONLY
    STUB_GPU(ReLUMaskLayer);
#endif

    INSTANTIATE_CLASS(ReLUMaskLayer);
    REGISTER_LAYER_CLASS(ReLUMask);

}  // namespace caffe

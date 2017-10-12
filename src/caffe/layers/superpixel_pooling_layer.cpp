//
// Created by Niuchuang on 17-7-3.
//

#include <algorithm>
#include <numeric>
#include <cfloat>
#include <vector>
#include <cstdio>
#include <iostream>
using namespace std;

#include "caffe/layers/superpixel_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    using std::min;
    using std::max;

    template <typename Dtype>
    void SuperpixelPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
        SuperpixelPoolingParameter superpixel_pool_param = this->layer_param_.superpixel_pooling_param();
        CHECK(superpixel_pool_param.has_scale()) << "scale is required.";
        CHECK_GT(superpixel_pool_param.scale(), 0);
        scale_ = superpixel_pool_param.scale();
        CHECK(superpixel_pool_param.has_shift()) << "shift is required.";
        shift_ = superpixel_pool_param.shift();
        sp_label_ = superpixel_pool_param.sp_label();

//        CHECK(superpixel_pool_param.has_rf) << "rf (receptive field) is required.";
//        CHECK_LT(superpixel_pool_param.rf(), 0);
//        rf_ = superpixel_pool_param.rf();

//        if (pool_param.global_pooling()) {
//            CHECK(!(pool_param.has_kernel_size() ||
//                    pool_param.has_kernel_h() || pool_param.has_kernel_w()))
//            << "With Global_pooling: true Filter size cannot specified";
//        } else {
//            CHECK(!pool_param.has_kernel_size() !=
//                  !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
//            << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
//            CHECK(pool_param.has_kernel_size() ||
//                  (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
//            << "For non-square filters both kernel_h and kernel_w are required.";
//        }
//        CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
//               && pool_param.has_pad_w())
//              || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
//        << "pad is pad OR pad_h and pad_w are required.";
//        CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
//               && pool_param.has_stride_w())
//              || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
//        << "Stride is stride OR stride_h and stride_w are required.";
//        global_pooling_ = pool_param.global_pooling();
//        if (global_pooling_) {
//            kernel_h_ = bottom[0]->height();
//            kernel_w_ = bottom[0]->width();
//        } else {
//            if (pool_param.has_kernel_size()) {
//                kernel_h_ = kernel_w_ = pool_param.kernel_size();
//            } else {
//                kernel_h_ = pool_param.kernel_h();
//                kernel_w_ = pool_param.kernel_w();
//            }
//        }
//        CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
//        CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
//        if (!pool_param.has_pad_h()) {
//            pad_h_ = pad_w_ = pool_param.pad();
//        } else {
//            pad_h_ = pool_param.pad_h();
//            pad_w_ = pool_param.pad_w();
//        }
//        if (!pool_param.has_stride_h()) {
//            stride_h_ = stride_w_ = pool_param.stride();
//        } else {
//            stride_h_ = pool_param.stride_h();
//            stride_w_ = pool_param.stride_w();
//        }
//        if (global_pooling_) {
//            CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
//            << "With Global_pooling: true; only pad = 0 and stride = 1";
//        }
//        if (pad_h_ != 0 || pad_w_ != 0) {
//            CHECK(this->layer_param_.pooling_param().pool()
//                  == PoolingParameter_PoolMethod_AVE
//                  || this->layer_param_.pooling_param().pool()
//                     == PoolingParameter_PoolMethod_MAX)
//            << "Padding implemented only for average and max pooling.";
//            CHECK_LT(pad_h_, kernel_h_);
//            CHECK_LT(pad_w_, kernel_w_);
//        }
    }

    template <typename Dtype>
    void SuperpixelPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
        // for debug
//        cout << "begin reshpe" << endl;
        CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
                                           << "corresponding to (num, channels, height, width)";
        channels_ = bottom[0]->channels();
        height_ = bottom[0]->height();
        width_ = bottom[0]->width();

        mask_height_ = bottom[1]->height();
        mask_width_ = bottom[1]->width();

        mask_count_ = mask_height_ * mask_width_;

        CHECK_EQ(4, bottom[1]->num_axes()) << "Superpixel mask must have 4 axes, "
                                           << "corresponding to (num, 1, height, width)";
        CHECK_EQ(1, bottom[1]->channels()) << "Channel of superpixel mask must be 1";

        CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "Data num must equal to mask num.";

        centers_w_high_.clear();
        centers_w_.clear();
        centers_w_low_.clear();
        centers_h_high_.clear();
        centers_h_.clear();
        centers_h_low_.clear();
        centers_h_.resize((unsigned long) height_);
        centers_w_.resize((unsigned long) width_);
        centers_h_low_.resize((unsigned long) height_);
        centers_h_high_.resize((unsigned long) height_);
        centers_w_low_.resize((unsigned long) width_);
        centers_w_high_.resize((unsigned long) width_);

        for (int i=0; i < height_; ++i){
            centers_h_[i] = i * scale_ + shift_;
            centers_h_low_[i] = centers_h_[i] - (scale_/2);
            centers_h_high_[i] = centers_h_[i] + (scale_/2);
        }

        for (int j=0; j < width_; ++j){
            centers_w_[j] = j * scale_ + shift_;
            centers_w_low_[j] = centers_w_[j] - (scale_/2);
            centers_w_high_[j] = centers_w_[j] + (scale_/2);
        }
        centers_h_low_[0] = 0;
        centers_w_low_[0] = 0;
        centers_h_high_[height_-1] = mask_height_;
        centers_w_high_[width_-1] = mask_width_;

        batch_size_ = bottom[0]->num();
        sp_fea_maps_.clear();
        sp_fea_maps_.resize((unsigned long) batch_size_);
        mask_labels_.clear();
        mask_labels_.resize((unsigned long) batch_size_);
        sp_label_nums_.clear();
        sp_label_nums_.resize((unsigned long) batch_size_);
        sp_nums_.clear();
        sp_nums_.resize((unsigned long) batch_size_);
        const Dtype* bottom_mask = bottom[1]->cpu_data();

        for (int i = 0; i < batch_size_; ++i){
            vector<int> vector_mask((unsigned long) mask_count_);
            for (int m=0; m < mask_count_; ++m) {
                vector_mask[m] = (int) bottom_mask[m];
//                sp_num = max(sp_num, (int) bottom_mask[m]);
            }
            sort(vector_mask.begin(), vector_mask.end());
            vector_mask.erase(unique(vector_mask.begin(), vector_mask.end()), vector_mask.end());
            CHECK_GE(vector_mask[0], -1) << "mask label cannot be less than -1.";
            if (vector_mask[0] == -1){
                vector_mask.erase(remove(vector_mask.begin(), vector_mask.end(), -1), vector_mask.end());
            }
            mask_labels_[i].assign(vector_mask.begin(), vector_mask.end());
            int sp_num = (int) mask_labels_[i].size();
            sp_nums_[i] = sp_num;

            // for debug
//            cout << "image id: " << i << endl;
//            cout << "superpixel number: " << sp_nums_[i] << endl;
            sp_label_nums_[i].resize((unsigned long) sp_num, 0);
//            sp_fea_maps_[i]->Reshape(sp_num, height_*width_, 1, 1);  // failed!
            sp_fea_maps_[i].resize((unsigned long) sp_num);
            for (int re = 0; re < sp_num; ++re){
                sp_fea_maps_[i][re].resize((unsigned long) (height_ * width_), 0);
            }
            for (int mh = 0; mh < mask_height_; ++mh){
                for (int mw= 0; mw < mask_width_; ++mw){
                    // ignore superpixel label of minus number
                    if (bottom_mask[mh*mask_width_+mw] >=0){
                        int label = (int) bottom_mask[mh*mask_width_+mw];
                        vector<int>::iterator result = find(mask_labels_[i].begin(),mask_labels_[i].end(), label);
                        int position = (int) distance(mask_labels_[i].begin(), result);
//                        sp_label_nums_[i][bottom_mask[mh*mask_width_+mw]] += 1;
                        sp_label_nums_[i][position] += 1;
                        bool is_continue = true;
                        for (int h = 0; h < height_ && is_continue; ++h){
                            if (mh >= centers_h_low_[h] && mh < centers_h_high_[h]){
                                for (int w = 0; w < width_ && is_continue; ++w){
                                    if (mw >= centers_w_low_[w] && mw < centers_w_high_[w]){
//                                    sp_fea_maps_[i][((int) bottom_mask[mh*mask_width_+mw])*height_*width_+h*width_+w];
//                                        sp_fea_maps_[i][(int) bottom_mask[mh*mask_width_+mw]][h*width_+w] += 1;
                                        sp_fea_maps_[i][position][h*width_+w] += 1;
                                        is_continue = false;
                                    }
                                }
                            }
                        }
                    }
                }
            }
//            int test_count = accumulate(sp_fea_maps_[i][0].begin(), sp_fea_maps_[i][0].end(),0);
            bottom_mask += bottom[1]->offset(1);
        }

//        activation_response_->Reshape(bottom[0]->num(), channels_, height_, width_);


//        if (global_pooling_) {
//            kernel_h_ = bottom[0]->height();
//            kernel_w_ = bottom[0]->width();
//        }
//        pooled_height_ = static_cast<int>(ceil(static_cast<float>(
//                                                       height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
//        pooled_width_ = static_cast<int>(ceil(static_cast<float>(
//                                                      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
//        if (pad_h_ || pad_w_) {
//            // If we have padding, ensure that the last pooling starts strictly
//            // inside the image (instead of at the padding); otherwise clip the last.
//            if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
//                --pooled_height_;
//            }
//            if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
//                --pooled_width_;
//            }
//            CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
//            CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
//        }
        top[0]->Reshape(bottom[0]->num(), channels_, 1, 1);
        if (top.size() > 1) {
            top[1]->ReshapeLike(*top[0]);
        }
        // If max pooling, we will initialize the vector index part.
//        if (this->layer_param_.pooling_param().pool() ==
//            PoolingParameter_PoolMethod_MAX && top.size() == 1) {
//            max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
//                             pooled_width_);
//        }
//        // If stochastic pooling, we will initialize the random index part.
//        if (this->layer_param_.pooling_param().pool() ==
//            PoolingParameter_PoolMethod_STOCHASTIC) {
//            rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
//                              pooled_width_);
//        }
    }

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
    template <typename Dtype>
    void SuperpixelPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
        // for debug
//        cout << "begin forward" << endl;

        const Dtype* bottom_data = bottom[0]->cpu_data();

//        const Dtype* bottom_mask = bottom[1]->cpu_data();

//        Dtype* acti_res_data = activation_response_->mutable_cpu_data();
//
//        caffe_set(activation_response_.count(), Dtype(0), acti_res_data);

        Dtype* top_data = top[0]->mutable_cpu_data();
        const int top_count = top[0]->count();
        // We'll output the mask to top[1] if it's of size >1.
//        const bool use_top_mask = top.size() > 1;
//        int* mask = NULL;  // suppress warnings about uninitalized variables
//        Dtype* top_mask = NULL;
        // Different pooling methods. We explicitly do the switch outside the for
        // loop to save time, although this results in more code.
        switch (this->layer_param_.superpixel_pooling_param().pool()) {
            case SuperpixelPoolingParameter_PoolMethod_MAX:
                // Initialize
                NOT_IMPLEMENTED;
                break;
            case SuperpixelPoolingParameter_PoolMethod_AVE:
                for (int i = 0; i < top_count; ++i) {
                    top_data[i] = 0;
                }
                // The main loop
                for (int n = 0; n < bottom[0]->num(); ++n) {
//                    int sp_num = 0;
//                    for (int m=0; m < mask_count_; ++m) {
//                        sp_num = max(sp_num, (int) bottom_mask[m]);
//                    }
//                    sp_num += 1;
                    int sp_num = sp_nums_[n];
                    sp_label_weight_.clear();
                    if (sp_label_ == -1){
                        sp_label_weight_.resize(sp_num, 1);
                    } else {
                        sp_label_weight_.resize(sp_num, 0);
                        sp_label_weight_[sp_label_] = 1;
                    }

                    for (int c = 0; c < channels_; ++c) {
                        vector<Dtype> sp_values(sp_num, 0);
                        for (int spi = 0; spi < sp_num; ++spi) {
                            int spi_num = sp_label_nums_[n][spi];
                            for (int h = 0; h < height_; ++h){
                                for (int w = 0; w < width_; ++w){
                                    sp_values[spi] += (sp_fea_maps_[n][spi][h*width_+w] * bottom_data[h*width_+w] * sp_label_weight_[spi]);
                                }
                            }
//                            for (int m=0; m < mask_count_; ++m){
//                                if (bottom_mask[m] == spi){
//                                    spi_num += 1;
//                                }
//                            }
//                            for (int mh = 0; mh < mask_height_; ++mh){
//                                for (int mw = 0; mw < mask_width_; ++mw) {
//                                    if (bottom_mask[mh*mask_height_+mw] == spi){
//                                        spi_num += 1;
//                                        bool isBreakLoop = true;
//                                        for (int h = 0; h < height_ && isBreakLoop; ++h){
//                                            if (mh >= centers_h_low_[h] && mh < centers_h_high_[h]){
//                                                for (int w = 0; w < width_ && isBreakLoop; ++w){
//                                                    if (mw >= centers_w_low_[w] && mw < centers_w_high_[w]){
//                                                        sp_values[spi] += bottom_data[h*width_+w];
//                                                        isBreakLoop = false;
//                                                    }
//                                                }
//                                            }
//
//                                        }
//                                    }
//
//                                }
//                            }
                            sp_values[spi] /= spi_num;
                        }
                        Dtype sp_values_sum = accumulate(sp_values.begin(), sp_values.end(), 0.0);
                        Dtype sp_values_mean = sp_values_sum / sp_values.size();
                        top_data[0] = sp_values_mean;
                        // compute offset
                        bottom_data += bottom[0]->offset(0, 1);
                        top_data += top[0]->offset(0, 1);
                    }
//                    bottom_mask += bottom[1]->offset(1);
                    // for debug
//                    for (int test_i = 0; test_i < 20; test_i ++){
//                        cout << "superpixel id from forward: " << test_i << " pixel number:" << sp_label_nums_[n][test_i] << endl;
//                    }
                }
                break;
            case SuperpixelPoolingParameter_PoolMethod_STOCHASTIC:
                NOT_IMPLEMENTED;
                break;
            default:
                LOG(FATAL) << "Unknown pooling method.";
        }
    }

    template <typename Dtype>
    void SuperpixelPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        // for debug
//        cout << "begin backward" << endl;
        if (!propagate_down[0]) {
            return;
        }
        const Dtype* top_diff = top[0]->cpu_diff();
//        const Dtype* bottom_data = bottom[0]->cpu_data();
//        const Dtype* bottom_mask = bottom[1]->cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        // Different pooling methods. We explicitly do the switch outside the for
        // loop to save time, although this results in more codes.
        caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
        // We'll output the mask to top[1] if it's of size >1.
//        const bool use_top_mask = top.size() > 1;
//        const int* mask = NULL;  // suppress warnings about uninitialized variables
//        const Dtype* top_mask = NULL;

        // for debug
//        cout << "top-> height: " << top[0]->height() << endl;
//        cout << "top-> width: " << top[0]->width() << endl;

        switch (this->layer_param_.superpixel_pooling_param().pool()) {
            case SuperpixelPoolingParameter_PoolMethod_MAX:
                // The main lool
                NOT_IMPLEMENTED;
                break;
            case SuperpixelPoolingParameter_PoolMethod_AVE:
                // The main loop
                for (int n = 0; n < top[0]->num(); ++n) {
//                    int sp_num = 0;
//                    for (int m=0; m < mask_count_; ++m) {
//                        sp_num = max(sp_num, (int) bottom_mask[m]);
//                    }
//                    sp_num += 1;
                    int sp_num = sp_nums_[n];
                    for (int c = 0; c < channels_; ++c) {
                        for (int ph = 0; ph < top[0]->height(); ++ph) {
                            for (int pw = 0; pw < top[0]->width(); ++pw) {
                                for (int h = 0; h < height_; ++h){
                                    for (int w = 0; w < width_; ++w){
                                        Dtype weight_hw = 0;
                                        for (int p = 0; p < sp_num; ++p){
                                            weight_hw += ((Dtype) sp_fea_maps_[n][p][h*width_+w]) / ((Dtype) sp_label_nums_[n][p]);
                                        }
//                                        int hstart, hend, wstart, wend;
//                                        hstart = static_cast<int>(ceil(static_cast<float>(centers_h_low_[h])));
//                                        hend = static_cast<int>(ceil(static_cast<float>(centers_h_high_[h])));
//                                        wstart = static_cast<int>(ceil(static_cast<float>(centers_w_low_[w])));
//                                        wend = static_cast<int>(ceil(static_cast<float>(centers_w_high_[w])));
//                                        hstart = max(hstart, 0);
//                                        wstart = max(wstart, 0);
//                                        hend = min(hend, mask_height_);
//                                        wend = min(wend, mask_width_);
//                                        int labels_num = (hend-hstart) * (wend-wstart);
//                                        vector<int> labels((unsigned long) labels_num);
//                                        int li = 0;
//                                        for (int subh = hstart; subh < hend; ++subh){
//                                            for (int subw = wstart; subw < wend; ++subw){
//                                                labels[li] = bottom_mask[subh*mask_width_+subw];
//                                                li++;
//                                            }
//                                        }
//                                        sort(labels.begin(), labels.end());
//                                        vector<int> labels_unique(labels);
//                                        labels_unique.erase(unique(labels_unique.begin(), labels_unique.end()), labels_unique.end());
//                                        vector<int> labels_count(labels_unique.size());
//                                        vector<int> sp_count(labels_unique.size(), 0);
//                                        Dtype weight_wh = 0;
//                                        for (int lc = 0; lc < labels_unique.size(); ++lc){
//                                            labels_count[lc] = (int) count(labels.begin(), labels.end(), labels_unique[lc]);
//                                            for (int mh = 0; mh < mask_height_; ++mh){
//                                                for (int mw = 0; mw < mask_width_; ++mw){
//                                                    if (bottom_mask[mh*mask_width_ + mw] == labels_unique[lc]){
//                                                        sp_count[lc] += 1;
//                                                    }
//                                                }
//                                            }
//                                            weight_wh += ((Dtype) labels_count[lc] / (Dtype) sp_count[lc]);
//                                        }
//                                        weight_wh /= ((Dtype) sp_num);
                                        bottom_diff[h * width_ + w] += (top_diff[ph * top[0]->width() + pw] * weight_hw);
                                    }
                                }
                            }
                        }
                        // offset
                        bottom_diff += bottom[0]->offset(0, 1);
                        top_diff += top[0]->offset(0, 1);
                    }
                    // for debug
//                    cout << "sp number from backward: " << sp_num << endl;
//                    bottom_mask += bottom[1]->offset(1);
                }
                break;
            case SuperpixelPoolingParameter_PoolMethod_STOCHASTIC:
                NOT_IMPLEMENTED;
                break;
            default:
                LOG(FATAL) << "Unknown pooling method.";
        }
    }


#ifdef CPU_ONLY
    STUB_GPU(SuperpixelPoolingLayer);
#endif

    INSTANTIATE_CLASS(SuperpixelPoolingLayer);
    REGISTER_LAYER_CLASS(SuperpixelPooling);

}  // namespace caffe

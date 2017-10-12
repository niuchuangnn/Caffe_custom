//
// Created by Niu Chuang on 17-9-28.
//

#include <cfloat>
#include <vector>
#include <cstdio>
#include <iostream>
#include <algorithm>
using namespace std;

#include "caffe/layers/eltwise_td_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    template <typename Dtype>
    void EltwiseTDLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
        CHECK(this->layer_param().eltwise_param().coeff_size() == 0
              || this->layer_param().eltwise_param().coeff_size() == bottom.size()) <<
                                                                                    "Eltwise Layer takes one coefficient per bottom blob.";
        CHECK(!(this->layer_param().eltwise_param().operation()
                == EltwiseParameter_EltwiseOp_PROD
                && this->layer_param().eltwise_param().coeff_size())) <<
                                                                      "Eltwise layer only takes coefficients for summation.";

        op_ = this->layer_param_.eltwise_param().operation();
        // Blob-wise coefficients for the elementwise operation.
        coeffs_ = vector<Dtype>(bottom.size(), 1);
        if (this->layer_param().eltwise_param().coeff_size()) {
            for (int i = 0; i < bottom.size(); ++i) {
                coeffs_[i] = this->layer_param().eltwise_param().coeff(i);
            }
        }
        stable_prod_grad_ = this->layer_param_.eltwise_param().stable_prod_grad();
    }

    template <typename Dtype>
    void EltwiseTDLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
        for (int i = 1; i < bottom.size(); ++i) {
            CHECK(bottom[i]->shape() == bottom[0]->shape());
        }

        for (int i = 0; i < top.size(); ++i){
            top[i]->ReshapeLike(*bottom[0]);
        }
        // If max operation, we will initialize the vector index part.
        if (this->layer_param_.eltwise_param().operation() ==
            EltwiseParameter_EltwiseOp_MAX && top.size() == 1) {
            max_idx_.Reshape(bottom[0]->shape());
        }
    }

    template <typename Dtype>
    void EltwiseTDLayer<Dtype>::Forward_cpu(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//        int* mask = NULL;
        const Dtype* bottom_data = bottom[0]->cpu_data();

        // scale the max value to 1
        Dtype max_value = *max_element(bottom_data, bottom_data+bottom[0]->count());
//        Dtype max_value = -100000;
//        for (int i = 0; i < bottom[0]->count(); ++i) {
//            max_value = bottom_data[i] > max_value ? bottom_data[i] : max_value;
//        }
//        cout << max_value << endl;

        if (max_value > 0){
            Blob<Dtype> MAX_VALUE(bottom[0]->shape());
            Dtype* max_value_data = MAX_VALUE.mutable_cpu_data();
            caffe_set<Dtype>(bottom[0]->count(), max_value, max_value_data);
            caffe_div<Dtype>(bottom[0]->count(), bottom_data, max_value_data, bottom[0]->mutable_cpu_data());
        }

        if (max_value < 0) {
            Blob<Dtype> MAX_VALUE(bottom[0]->shape());
            Dtype* max_value_data = MAX_VALUE.mutable_cpu_data();
            caffe_set<Dtype>(bottom[0]->count(), -max_value, max_value_data);
            caffe_div<Dtype>(bottom[0]->count(), bottom_data, max_value_data, bottom[0]->mutable_cpu_data());
        }

        const Dtype* out_data = bottom[1]->cpu_data();
        switch (op_) {
            case EltwiseParameter_EltwiseOp_PROD:
                NOT_IMPLEMENTED;
                break;
            case EltwiseParameter_EltwiseOp_SUM:
//                caffe_set(count, Dtype(0), top_data);
                // TODO(shelhamer) does BLAS optimize to sum for coeff = 1?
//                for (int i = 0; i < bottom.size(); ++i) {
//                    caffe_axpy(count, coeffs_[i], bottom[i]->cpu_data(), top_data);
//                }
                for (int i = 0; i < top.size(); ++i){
                    const int count = top[i]->count();
                    Dtype* top_data = top[i]->mutable_cpu_data();
                    const Dtype* bottom_data_a = bottom[i+2]->cpu_data();
                    for (int j = 0; j < count; ++j){
                        top_data[j] = out_data[j] == 0 ? Dtype(0):bottom_data_a[j]*bottom_data[j]/out_data[j];
                    }
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
    void EltwiseTDLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

        const Dtype* top_diff_a = top[0]->cpu_diff();
        const Dtype* top_diff_b = top[1]->cpu_diff();
        const Dtype* out_data = bottom[1]->cpu_data();
        const Dtype* activation_data_a = bottom[2]->cpu_data();
        const Dtype* activation_data_b = bottom[3]->cpu_data();
        Dtype* activation_diff_a = bottom[2]->mutable_cpu_diff();
        Dtype* activation_diff_b = bottom[3]->mutable_cpu_diff();
        const Dtype* bottom_data = bottom[0]->cpu_data();

        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        switch (op_) {
            case EltwiseParameter_EltwiseOp_PROD:
                NOT_IMPLEMENTED;
                break;
            case EltwiseParameter_EltwiseOp_SUM:
                if (propagate_down[0]) {
                    for (int i = 0; i < bottom[0]->count(); ++i) {
                        bottom_diff[i] = out_data[i] > 0 ?
                                         (top_diff_a[i] * activation_data_a[i] + top_diff_b[i] * activation_data_b[i]) /
                                         out_data[i] : Dtype(0);
                    }
                }
                if (propagate_down[2]) {
                    for (int i = 0; i < bottom[2]->count(); ++i) {
                        activation_diff_a[i] = out_data[i] > 0 ?
                                               (top_diff_a[i] - top_diff_b[i]) * bottom_data[i] * activation_data_b[i] / (out_data[i] * out_data[i]) : Dtype(0);
                    }
                }

                if (propagate_down[3]) {
                    for (int i = 0; i < bottom[3]->count(); ++i) {
                        activation_diff_b[i] = out_data[i] > 0 ?
                                               (top_diff_b[i] - top_diff_a[i]) * bottom_data[i] * activation_data_a[i] / (out_data[i] * out_data[i]) : Dtype(0);
                    }
                }
                break;
            case EltwiseParameter_EltwiseOp_MAX:
                NOT_IMPLEMENTED;
                break;
            default:
                LOG(FATAL) << "Unknown elementwise operation.";
        }


    }

#ifdef CPU_ONLY
    STUB_GPU(EltwiseTDLayer);
#endif

    INSTANTIATE_CLASS(EltwiseTDLayer);
    REGISTER_LAYER_CLASS(EltwiseTD);

}  // namespace caffe

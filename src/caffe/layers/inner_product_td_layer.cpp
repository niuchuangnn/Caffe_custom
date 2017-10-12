#include <vector>
#include <cstdio>
#include <iostream>
#include <algorithm>
using namespace std;
#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_td_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    template <typename Dtype>
    void InnerProductTDLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
        const int num_output = this->layer_param_.inner_product_td_param().num_output();
        bias_term_ = this->layer_param_.inner_product_td_param().bias_term();
        transpose_ = this->layer_param_.inner_product_td_param().transpose();
        N_ = num_output;
        const int axis = bottom[0]->CanonicalAxisIndex(
                this->layer_param_.inner_product_td_param().axis());
        // Dimensions starting from "axis" are "flattened" into a single
        // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
        // and axis == 1, N inner products with dimension CHW are performed.
        K_ = bottom[0]->count(axis);
        // Check if we need to set up the weights
        if (this->blobs_.size() > 0) {
            LOG(INFO) << "Skipping parameter initialization";
        } else {
            if (bias_term_) {
                this->blobs_.resize(2);
            } else {
                this->blobs_.resize(1);
            }
            // Initialize the weights
            vector<int> weight_shape(2);
            if (transpose_) {
                weight_shape[0] = K_;
                weight_shape[1] = N_;
            } else {
                weight_shape[0] = N_;
                weight_shape[1] = K_;
            }
            this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
            // fill the weights
            shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
                    this->layer_param_.inner_product_td_param().weight_filler()));
            weight_filler->Fill(this->blobs_[0].get());
            // If necessary, intiialize and fill the bias term
            if (bias_term_) {
                vector<int> bias_shape(1, N_);
                this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
                shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                        this->layer_param_.inner_product_td_param().bias_filler()));
                bias_filler->Fill(this->blobs_[1].get());
            }
        }  // parameter initialization
        this->param_propagate_down_.resize(this->blobs_.size(), true);
    }

    template <typename Dtype>
    void InnerProductTDLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
        // Figure out the dimensions
        const int axis = bottom[0]->CanonicalAxisIndex(
                this->layer_param_.inner_product_td_param().axis());
        const int new_K = bottom[0]->count(axis);
        CHECK_EQ(K_, new_K)
            << "Input size incompatible with inner product parameters.";
        // The first "axis" dimensions are independent inner products; the total
        // number of these is M_, the product over these dimensions.
        M_ = bottom[0]->count(0, axis);
        // The top shape will be the bottom shape with the flattened axes dropped,
        // and replaced by a single axis with dimension num_output (N_).
        vector<int> top_shape = bottom[0]->shape();
        top_shape.resize(axis + 1);
        top_shape[axis] = N_;
        top[0]->Reshape(top_shape);
        // Set up the bias multiplier
        if (bias_term_) {
            vector<int> bias_shape(1, M_);
            bias_multiplier_.Reshape(bias_shape);
            caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
        }

        vector<int> activation_shape;
        activation_shape = bottom[1]->shape();
        activation_shape.resize(axis + 1);
        for (int i=0; i < top_shape.size(); ++i){
            CHECK_EQ(top_shape[i], activation_shape[i]) << "top shape must equal to bottom[1] shape";
        }

        NN_.Reshape(bottom[0]->shape());
        NF_.Reshape(bottom[0]->shape());
        buff_.Reshape(top[0]->shape());
        vector<int> c_shape(2);
        c_shape[0] = N_;
        c_shape[1] = 1;
//        cout << "c_shape: " << c_shape[0] << " * " << c_shape[1] << endl;
        C_.Reshape(c_shape);
        c_shape[0] = M_;
        c_shape[1] = 1;
        CM_.Reshape(c_shape);
        c_shape[0] = N_;
        c_shape[1] = N_;
        SR_.Reshape(c_shape);
    }

    template <typename Dtype>
    void InnerProductTDLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
//        const Dtype* bottom_data = bottom[0]->cpu_data();
//        Dtype* top_data = top[0]->mutable_cpu_data();
//        const Dtype* weight = this->blobs_[0]->cpu_data();
//        caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
//                              M_, N_, K_, (Dtype)1.,
//                              bottom_data, weight, (Dtype)0., top_data);
//        if (bias_term_) {
//            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
//                                  bias_multiplier_.cpu_data(),
//                                  this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
//        }

        // get the new weight W+
        const Dtype* W_data = this->blobs_[0]->cpu_data();
        Blob<Dtype> W_plus(this->blobs_[0]->shape());
        Dtype* W_plus_data = W_plus.mutable_cpu_data();

        for (int i = 0; i < W_plus.count(); ++i) {
            W_plus_data[i] = std::max(W_data[i], Dtype(0));
        }

        // do backwardpass to compute the normalization factor by forwardpassing using W+
//        Blob<Dtype> NN(bottom[0]->shape());
        Dtype* NF_data = NF_.mutable_cpu_data();
        const Dtype* activation_data = bottom[1]->cpu_data();
        if (transpose_) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
                                  M_, K_, N_,
                                  (Dtype)1., activation_data, W_plus_data,
                                  (Dtype)0., NF_data);
        } else {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                                  M_, K_, N_,
                                  (Dtype)1., activation_data, W_plus_data,
                                  (Dtype)0., NF_data);
        }

        // do normalization
        Dtype* NN_data = NN_.mutable_cpu_data();
        const Dtype* bottom_data = bottom[0]->mutable_cpu_data();
        for (int i = 0; i < NN_.count(); ++i) {
            NN_data[i] = NF_data[i] == Dtype(0) ? Dtype(0):(bottom_data[i]/NF_data[i]);
        }

        // do forward
        Dtype* top_data = top[0]->mutable_cpu_data();
        caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
                              M_, N_, K_, (Dtype)1.,
                              NN_data, W_plus_data, (Dtype)0., top_data);

        // multiply the bottom data
        caffe_mul<Dtype>(top[0]->count(), top[0]->cpu_data(), activation_data, top[0]->mutable_cpu_data());

        // scale the max value to 1
//        Dtype max_value = *max_element(top[0]->cpu_data(), top[0]->cpu_data()+top[0]->count());
//        Dtype max_value = -100000;
//        for (int i = 0; i < top[0]->count(); ++i) {
//            max_value = top_data[i] > max_value ? top_data[i] : max_value;
//        }
//        cout << max_value << endl;
//
//        if (max_value > 0){
//            Blob<Dtype> MAX_VALUE(top[0]->shape());
//            Dtype* max_value_data = MAX_VALUE.mutable_cpu_data();
//            caffe_set<Dtype>(top[0]->count(), max_value, max_value_data);
//            caffe_div<Dtype>(top[0]->count(), top_data, max_value_data, top_data);
//        }
//
//        if (max_value < 0) {
//            Blob<Dtype> MAX_VALUE(top[0]->shape());
//            Dtype* max_value_data = MAX_VALUE.mutable_cpu_data();
//            caffe_set<Dtype>(top[0]->count(), -max_value, max_value_data);
//            caffe_div<Dtype>(top[0]->count(), top_data, max_value_data, top_data);
//        }
    }

    template <typename Dtype>
    void InnerProductTDLayer<Dtype>::print_data(const Dtype* start, const int n){
        cout << "size: " << n << "  data: ";
        for (int i = 0; i < n; ++i){
            cout << start[i] << " ";
            if (i == n-1)
                cout << endl;
        }
    }

    template <typename Dtype>
    void InnerProductTDLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                const vector<bool>& propagate_down,
                                                const vector<Blob<Dtype>*>& bottom) {
        if (this->param_propagate_down_[0]) {
//            const Dtype* top_diff = top[0]->cpu_diff();
//            const Dtype* bottom_data = bottom[0]->cpu_data();
//            // Gradient with respect to weight
//            if (transpose_) {
//                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
//                                      K_, N_, M_,
//                                      (Dtype)1., bottom_data, top_diff,
//                                      (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
//            } else {
//                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
//                                      N_, K_, M_,
//                                      (Dtype)1., top_diff, bottom_data,
//                                      (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
//            }
            caffe_set<Dtype>(this->blobs_[0]->count(), Dtype(0), this->blobs_[0]->mutable_cpu_diff());
        }
//        if (bias_term_ && this->param_propagate_down_[1]) {
//            const Dtype* top_diff = top[0]->cpu_diff();
//            // Gradient with respect to bias
//            caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
//                                  bias_multiplier_.cpu_data(), (Dtype)1.,
//                                  this->blobs_[1]->mutable_cpu_diff());
//        }
        if (propagate_down[0]) {

            // for debug
//            cout << "backpropagate 0" << endl;

            // Gradient with respect to bottom data

            // Multiply G_{n-1} with A_{n-1}
            const Dtype* top_diff = top[0]->cpu_diff();
            const Dtype* activation_data = bottom[1]->cpu_data();
//            cout << "top diff: " << endl;
//            print_data(top_diff, top[0]->count());

//            cout << "bottom[1]: " << endl;
//            print_data(activation_data, bottom[1]->count());

//            Blob<Dtype> buff(top[0]->shape());
            Dtype* buff_data = buff_.mutable_cpu_data();
            caffe_mul<Dtype>(buff_.count(), activation_data, top_diff, buff_data);

            // do backward
            // get the new weight W+
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            const Dtype* W_data = this->blobs_[0]->cpu_data();
            Blob<Dtype> W_plus(this->blobs_[0]->shape());
            Dtype* W_plus_data = W_plus.mutable_cpu_data();

            for (int i = 0; i < W_plus.count(); ++i) {
                W_plus_data[i] = std::max(W_data[i], Dtype(0));
            }

//            cout << "W plus data: " << endl;
//            print_data(W_plus_data, this->blobs_[0]->count());

            if (transpose_) {
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
                                      M_, K_, N_,
                                      (Dtype)1., buff_data, W_plus_data,
                                      (Dtype)0., bottom_diff);
            } else {
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                                      M_, K_, N_,
                                      (Dtype)1., buff_data, W_plus_data,
                                      (Dtype)0., bottom_diff);
            }

//            cout << "bottom diff: " << endl;
//            print_data(bottom_diff, bottom[0]->count());

            // Normalization
            // Get normalization data
            const Dtype* NF_data = NF_.cpu_data();

//            cout << "NN data: " << endl;
//            print_data(NN_data, NN_.count());

            // Multiplication
//            caffe_mul<Dtype>(bottom[0]->count(), bottom[0]->cpu_diff(), NF_data, bottom_diff);
            for (int i = 0; i < NF_.count(); ++i) {
                bottom_diff[i] = NF_data[i] == Dtype(0) ? Dtype(0):(bottom_diff[i]/NF_data[i]);
            }

//            cout << " final bottom diff: " << endl;
//            print_data(bottom_diff, bottom[0]->count());

        }

        if (propagate_down[1]) {

            // Normalization
            // Get normalization data
            const Dtype* NN_data = NN_.cpu_data();

//            cout << "NN data: " << endl;
//            print_data(NN_data, NN_.count());

            // Do forward
            // get the new weight W+
            const Dtype* W_data = this->blobs_[0]->cpu_data();
            Blob<Dtype> W_plus(this->blobs_[0]->shape());
            Dtype* W_plus_data = W_plus.mutable_cpu_data();

            for (int i = 0; i < W_plus.count(); ++i) {
                W_plus_data[i] = std::max(W_data[i], Dtype(0));
            }

//            cout << "W plus data: " << endl;
//            print_data(W_plus_data, this->blobs_[0]->count());

            Dtype* activation_diff = bottom[1]->mutable_cpu_diff();

            caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
                                  M_, N_, K_, (Dtype)1.,
                                  NN_data, W_plus_data, (Dtype)0., activation_diff);

//            cout << "first activation diff: " << endl;
//            print_data(activation_diff, bottom[1]->count());

            // Multiply top diff
            const Dtype* top_diff = top[0]->cpu_diff();

//            cout << "top diff: " << endl;
//            print_data(top_diff, top[0]->count());

            caffe_mul<Dtype>(top[0]->count(), activation_diff, top_diff, activation_diff);

//            cout << "activation diff multtiplied by top diff: " << endl;
//            print_data(activation_diff, bottom[1]->count());

            // Compute the second term and subtract the second term from the activation diff

            // Compute P_{n} / N_{n}^{2}
            Dtype* NN_data2 = NN_.mutable_cpu_data();

//            cout << "NN data: " << endl;
//            print_data(NN_data2, NN_.count());

            const Dtype* NF_data = NF_.cpu_data();

//            cout << "NF data: " << endl;
//            print_data(NF_data, NF_.count());

            for (int i = 0; i < NN_.count(); ++i) {
                NN_data2[i] = NF_data[i] == Dtype(0) ? Dtype(0):(NN_data2[i]/NF_data[i]);
            }

//            cout << "NN data2: " << endl;
//            print_data(NN_data2, NN_.count());

            // Compute W_{u}
            const Dtype* W_plus_data_c = W_plus.cpu_data();

//            cout << "W plus data c: " << endl;
//            print_data(W_plus_data_c, this->blobs_[0]->count());

            Blob<Dtype> Wu(W_plus.shape());
            Dtype* Wu_data = Wu.mutable_cpu_data();

            Dtype* buff_data = buff_.mutable_cpu_data();
            Dtype* c_data = C_.mutable_cpu_data();
            caffe_set<Dtype>(C_.count(), (Dtype)1., c_data);
            Dtype* cm_data = CM_.mutable_cpu_data();
            caffe_set<Dtype>(CM_.count(), (Dtype)1., cm_data);

            Dtype* sr_data = SR_.mutable_cpu_data();

//
//            cout << "N_: " << N_ << endl;
//            cout << "K_: " << K_ << endl;
//            cout << "W shape: " << this->blobs_[0]->shape()[0] << " * " << this->blobs_[0]->shape()[1] << endl;
//            cout << "C_ shape: " << C_.shape()[0] << " * " << C_.shape()[1] << endl;
//            cout << "C_ data: " << endl;
//            print_data(c_data, C_.count());
            for (int u = 0; u < N_; ++u){
//                caffe_copy<Dtype>(W_plus.count(), W_plus_data_c, Wu_data);

//                cout << "Wu data copied: " << endl;
//                print_data(Wu_data, this->blobs_[0]->count());

                const Dtype* W_plus_data_c_u = W_plus_data_c + K_ * u;

//                cout << "W_plus_data_c_u: " << endl;
//                print_data(W_plus_data_c_u, K_);

//                for (int n = 0; n < N_; ++n) {
//                    const Dtype* W_plus_data_c_n = W_plus_data_c + K_ * n;
//
////                    cout << "W_plus_data_c_n: " << endl;
////                    print_data(W_plus_data_c_n, K_);
//
//                    Dtype* Wu_data_n = Wu_data + K_ * n;
//                    caffe_mul<Dtype>(K_, W_plus_data_c_n, W_plus_data_c_u, Wu_data_n);
//
////                    cout << "Wu_data_n: " << endl;
////                    print_data(Wu_data_n, K_);
//                }

                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, K_, 1,
                                      (Dtype)1., c_data, W_plus_data_c_u, (Dtype)0., Wu_data);

                caffe_mul<Dtype>(Wu.count(), Wu_data, W_plus_data_c, Wu_data);

//                cout << "Wu data: " << endl;
//                print_data(Wu_data, this->blobs_[0]->count());

                // Do forward
                caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
                                      M_, N_, K_, (Dtype)1.,
                                      NN_data2, Wu_data, (Dtype)0., buff_data);

//                cout << "buff data:" << endl;
//                print_data(buff_data, buff_.count());

                // Multiply top diff and activation data
                const Dtype* activation_data = bottom[1]->cpu_data();
                caffe_mul<Dtype>(buff_.count(), buff_data, top_diff, buff_data);
                caffe_mul<Dtype>(buff_.count(), buff_data, activation_data, buff_data);

//                cout << "buff data multiplied by top diff and activation data: " << endl;
//                print_data(buff_data, buff_.count());

                // Compute the sum along the feature dimension and subtract the sum from the first term
//                Dtype* activation_diff_u = activation_diff + u;
//                for (int m = 0; m < M_; ++m){
//                    Dtype* buff_data_s = buff_data + N_ * m;
//                    for (int mn = 1; mn < N_; ++mn){
//                        buff_data_s[0] += buff_data_s[mn];
//                    }
//
////                    cout << "buff data sum: " << buff_data_s[0] << " of m:" << m << " u: " << u << endl;
//
//                    activation_diff_u[N_ * m] -= buff_data_s[0];
//                }

                // Compute the sum along the feature dimension
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, 1, N_,
                                      (Dtype)1., buff_data, c_data, (Dtype)0., cm_data);
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
                                      (Dtype)1., cm_data, c_data, (Dtype)0., buff_data);

                // Subtract the sum from the first term along the batch size dimension
                caffe_set<Dtype>(SR_.count(), (Dtype)0, sr_data);
                sr_data[u * N_ + u] = 1;
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, N_,
                                      Dtype(-1.), buff_data, sr_data, (Dtype)1., activation_diff);


            }

//            cout << "final activation_diff_u:" << endl;
//            print_data(activation_diff, bottom[1]->count());


        }


    }

#ifdef CPU_ONLY
    STUB_GPU(InnerProductTDLayer);
#endif

    INSTANTIATE_CLASS(InnerProductTDLayer);
    REGISTER_LAYER_CLASS(InnerProductTD);

}  // namespace caffe

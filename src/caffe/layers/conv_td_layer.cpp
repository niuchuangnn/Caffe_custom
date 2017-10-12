#include <vector>
#include <cstdio>
#include <iostream>
using namespace std;
#include "caffe/layers/conv_td_layer.hpp"

namespace caffe {

    template <typename Dtype>
    void ConvolutionTDLayer<Dtype>::compute_output_shape() {
        const int* kernel_shape_data = this->kernel_shape_.cpu_data();
        const int* stride_data = this->stride_.cpu_data();
        const int* pad_data = this->pad_.cpu_data();
        const int* dilation_data = this->dilation_.cpu_data();
        this->output_shape_.clear();
        for (int i = 0; i < this->num_spatial_axes_; ++i) {
            // i + 1 to skip channel axis
            const int input_dim = this->input_shape(i + 1);
            const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
            const int output_dim = stride_data[i] * (input_dim - 1)
                                   + kernel_extent - 2 * pad_data[i] + stride_data[i] - 1;

            this->output_shape_.push_back(output_dim);
        }
    }

    template <typename Dtype>
    void ConvolutionTDLayer<Dtype>::print_2darray(const Dtype *start, const int H, const int W) {

        for (int h = 0; h < H; ++h){
            for (int w = 0; w < W; ++w){
                cout << start[h * W + w] << " ";
                if (w == W-1)
                    cout << endl;
            }
        }
    }

    template <typename Dtype>
    void ConvolutionTDLayer<Dtype>::print_4darray(const Dtype *start, vector<int> shape) {
        int N = shape[0];
        int C = shape[1];
        int H = shape[2];
        int W = shape[3];
        const Dtype* start_nc;
        for (int n = 0; n < N; ++n){
            for (int c = 0; c < C; ++c){
                cout << "(" << n << "," << c << ")" << endl;
                start_nc = start + n * C*H*W + c * H*W;
                print_2darray(start_nc, H, W);
            }
        }
    }

    template <typename Dtype>
    void ConvolutionTDLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
        vector<int> activation_shape;
        activation_shape = bottom[1]->shape();
        vector<int> top_shape;
        top_shape = top[0]->shape();
//        cout << "num_spatial_axes_: " << top_shape.size() << endl;
//        for (int i=0; i < top_shape.size(); ++i){
//            cout << i << " activation: " << activation_shape[i] << " output: " << top_shape[i] << endl;
//        }
        for (int i=0; i < top_shape.size(); ++i){
            CHECK_EQ(activation_shape[i], top_shape[i]) << "bottom[1] shape must equal to top shape";
        }

        // get the new weight W+
        const Dtype* W_data = this->blobs_[0]->cpu_data();
        Blob<Dtype> W_plus(this->blobs_[0]->shape());
        Dtype* W_plus_data = W_plus.mutable_cpu_data();
        for (int i = 0; i < W_plus.count(); ++i) {
            W_plus_data[i] = std::max(W_data[i], Dtype(0));
        }

//        cout << "W_plus_data: " << endl;
//        print_4darray(W_plus_data, this->blobs_[0]->shape());

//        Blob<Dtype> NN(bottom[0]->shape());
        NF_.Reshape(bottom[0]->shape());
        NN_.Reshape(bottom[0]->shape());
        Dtype* NF_data = NF_.mutable_cpu_data();
        Dtype* NN_data = NN_.mutable_cpu_data();
        for (int i = 0; i < top.size(); ++i) {
            // do forward to compute the normalization factor by forwardpassing using W+
            const Dtype* activation_data = bottom[1]->cpu_data();
            for (int n = 0; n < this->num_; ++n) {
                this->forward_cpu_gemm(activation_data + n * this->top_dim_, W_plus_data,
                                       NF_data + n * this->bottom_dim_);
            }

//            cout << "NF_data: " << endl;
//            print_4darray(NF_data, NF_.shape());
            // do normalization
            const Dtype* bottom_data = bottom[0]->mutable_cpu_data();
            for (int j = 0; j < NF_.count(); ++j) {
                NN_data[j] = NF_data[j] == Dtype(0) ? Dtype(0):(bottom_data[j]/NF_data[j]);
            }

//            cout << "NN_data:" << endl;
//            print_4darray(NN_data, NN_.shape());

            // do backward
            Dtype* top_data = top[i]->mutable_cpu_data();
            for (int n = 0; n < this->num_; ++n) {
                this->backward_cpu_gemm(NN_data + n * this->bottom_dim_, W_plus_data,
                                        top_data + n * this->top_dim_);
                if (this->bias_term_) {
                    const Dtype* bias = this->blobs_[1]->cpu_data();
                    this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
                }
            }

//            cout << "top data:" << endl;
//            print_4darray(top_data, top[0]->shape());

            // multiply the bottom data
            caffe_mul<Dtype>(bottom[1]->count(), top_data, activation_data, top_data);

//            cout << "final top data:" << endl;
//            print_4darray(top_data, top[0]->shape());

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

    }

    template <typename Dtype>
    void ConvolutionTDLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                               const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        // Get W+
        const Dtype* W_data = this->blobs_[0]->cpu_data();
        Blob<Dtype> W_plus(this->blobs_[0]->shape());
        Dtype* W_plus_data = W_plus.mutable_cpu_data();
        for (int i = 0; i < W_plus.count(); ++i) {
            W_plus_data[i] = std::max(W_data[i], Dtype(0));
        }

        buff_.Reshape(top[0]->shape());
        Dtype* buff_data = buff_.mutable_cpu_data();
        const Dtype* top_diff = top[0]->cpu_diff();
        const Dtype* activation_data = bottom[1]->cpu_data();
        const Dtype* NF_data = NF_.cpu_data();
        const Dtype* NN_data = NN_.cpu_data();

//        cout << "NF data:" << endl;
//        print_4darray(NF_data, NF_.shape());
//
//        cout << "NN data:" << endl;
//        print_4darray(NN_data, NN_.shape());

        if (propagate_down[0]){
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

            // Multiply top diff with activation data
            caffe_mul<Dtype>(top[0]->count(), top_diff, activation_data, buff_data);

//            cout << "Multiply top diff and activation data: " << endl;
//            print_4darray(buff_data, top[0]->shape());

            // Do backward
            for (int n = 0; n < this->num_; ++n) {
                this->forward_cpu_gemm(buff_data + n * this->top_dim_, W_plus_data,
                                       bottom_diff + n * this->bottom_dim_);
            }

//            cout << "bottom diff: " << endl;
//            print_4darray(bottom_diff, bottom[0]->shape());

            // Multiply normalization data
//            caffe_mul<Dtype>(bottom[0]->count(), bottom_diff, NF_data, bottom_diff);
            for (int j = 0; j < NF_.count(); ++j) {
                bottom_diff[j] = NF_data[j] == Dtype(0) ? Dtype(0):(bottom_diff[j]/NF_data[j]);
            }

//            cout << "final bottom diff: " << endl;
//            print_4darray(bottom_diff, bottom[0]->shape());

        }

        if (propagate_down[1]){

            Dtype* activation_diff = bottom[1]->mutable_cpu_diff();

            // Do forward
            for (int n = 0; n < this->num_; ++n) {
                this->backward_cpu_gemm(NN_data + n * this->bottom_dim_, W_plus_data,
                                        activation_diff + n * this->top_dim_);
                if (this->bias_term_) {
                    const Dtype* bias = this->blobs_[1]->cpu_data();
                    this->forward_cpu_bias(activation_diff + n * this->top_dim_, bias);
                }
            }

            cout << "activation diff: " << endl;
            print_4darray(activation_diff, bottom[1]->shape());

            // Multiply activation diff with top diff
            caffe_mul<Dtype>(bottom[1]->count(), activation_diff, top_diff, activation_diff);

            cout << "first term: " << endl;
            print_4darray(activation_diff, bottom[1]->shape());

            // Compute the second term and subtract the second term from the activation diff

            // Compute normalization data2

            Dtype* NN_data2 = NN_.mutable_cpu_data();
            for (int j = 0; j < NF_.count(); ++j) {
                NN_data2[j] = NF_data[j] == Dtype(0) ? Dtype(0):(NN_data[j]/NF_data[j]);
            }

            cout << "NN data 2: " << endl;
            print_4darray(NN_data2, NN_.shape());

            // Compute Wu and
//            cout << "output num: " << this->blobs_[0]->num() << endl;
//            cout << "input num: " << this->blobs_[0]->channels() << endl;
//            cout << "height: " << this->blobs_[0]->height();
//            cout << "width: " << this-> blobs_[0]->width();
            Blob<Dtype> Wu(W_plus.shape());
            Dtype* Wu_data = Wu.mutable_cpu_data();
            int height = top[0]->shape()[2];
            int width = top[0]->shape()[3];
            for (int u = 0; u < top[0]->channels(); u++){

                // Compute Wu
                for (int n = 0; n < Wu.num(); ++n){
                    for (int u1 = 0; u1 < Wu.channels(); ++u1){
                        for (int h = 0; h < Wu.height(); ++h){
                            for (int w = 0; w < Wu.width(); ++w){
                                int index_Wu = n*Wu.channels()*Wu.height()*Wu.width() + u1*Wu.height()*Wu.width() + h*Wu.width() + w;
                                int index_W = n*Wu.channels()*Wu.height()*Wu.width() + u*Wu.height()*Wu.width() + h*Wu.width() + w;
                                Wu_data[index_Wu] = W_plus_data[index_Wu] * W_plus_data[index_W];
                            }
                        }
                    }
                }

                // Do forward
                for (int n = 0; n < this->num_; ++n) {
                    this->backward_cpu_gemm(NN_data2 + n * this->bottom_dim_, Wu_data,
                                            buff_data + n * this->top_dim_);
                    if (this->bias_term_) {
                        const Dtype* bias = this->blobs_[1]->cpu_data();
                        this->forward_cpu_bias(buff_data + n * this->top_dim_, bias);
                    }
                }

                // Multiply top diff and activation data
                caffe_mul<Dtype>(top[0]->count(), buff_data, top_diff, buff_data);
                caffe_mul<Dtype>(top[0]->count(), buff_data, activation_data, buff_data);

                // Compute the sum along the output number dimension and subtract the sum from the first term

                for (int n = 0; n < this->num_; ++n){
                    for (int c = 0; c < this->num_output_; ++c){
                        for (int h = 0; h < height; ++h){
                            for (int w = 0; w < width; ++w){
//                                buff_data[n*this->top_dim_ + 0*height*width + h*width + w] += buff_data[n*this->top_dim_ + c*height*width + h*width + w];
                                activation_diff[n*this->top_dim_ + u*height*width + h*width + w] -= buff_data[n*this->top_dim_ + c*height*width + h*width + w];
                            }
                        }
                    }

                    // subtract the sum from the first term
//                    for (int h1 = 0; h1 < height; ++h1){
//                        for (int w1 = 0; w1 < width; ++w1){
//                            activation_diff[n*this->top_dim_ + u*height*width + h1*width + w1] -= buff_data[n*this->top_dim_ + 0*height*width + h1*width + w1];
//                        }
//                    }
                }


            }

        }

        if (this->param_propagate_down_[0]){
            Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
            caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
        }


    }

#ifdef CPU_ONLY
    STUB_GPU(ConvolutionTDLayer);
#endif

    INSTANTIATE_CLASS(ConvolutionTDLayer);
    REGISTER_LAYER_CLASS(ConvolutionTD);

}  // namespace caffe

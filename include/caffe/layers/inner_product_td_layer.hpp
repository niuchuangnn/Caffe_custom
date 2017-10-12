//
// Created by Niu Chuang on 17-9-24.
//

#ifndef CAFFE_INNERPRODUCT_TD_LAYER_HPP
#define CAFFE_INNERPRODUCT_TD_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
    template <typename Dtype>
    class InnerProductTDLayer : public Layer<Dtype> {
    public:
        explicit InnerProductTDLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "InnerProductTD"; }
        virtual inline int ExactNumBottomBlobs() const { return 2; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        void print_data(const Dtype* start, const int n);

        int M_;
        int K_;
        int N_;
        bool bias_term_;
        Blob<Dtype> bias_multiplier_;
        Blob<Dtype> NN_;
        Blob<Dtype> NF_;
        Blob<Dtype> buff_;
        Blob<Dtype> C_;
        Blob<Dtype> CM_;
        Blob<Dtype> SR_;
        bool transpose_;  ///< if true, assume transposed weights
    };

}  // namespace caffe


#endif //CAFFE_INNERPRODUCTTDLAYER_HPP

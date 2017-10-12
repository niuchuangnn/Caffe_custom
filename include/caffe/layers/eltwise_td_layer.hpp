//
// Created by Niu Chuang on 17-9-28.
//

#ifndef CAFFE_ELTWISE_TD_LAYER_HPP
#define CAFFE_ELTWISE_TD_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Compute elementwise operations, such as product and sum,
 *        along multiple input Blobs.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
    template <typename Dtype>
    class EltwiseTDLayer : public Layer<Dtype> {
    public:
        explicit EltwiseTDLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "EltwiseTD"; }
        virtual inline int ExactNumBottomBlobs() const { return 4; }
        virtual inline int ExactNumTopBlobs() const { return 2; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        EltwiseParameter_EltwiseOp op_;
        vector<Dtype> coeffs_;
        Blob<int> max_idx_;

        bool stable_prod_grad_;
    };

}  // namespace caffe


#endif //CAFFE_ELTWISE_TD_LAYER_HPP

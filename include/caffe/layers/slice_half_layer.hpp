//
// Created by Niuchuang on 17-6-6.
//

#ifndef CAFFE_SLICE_HALF_LAYER_HPP
#define CAFFE_SLICE_HALF_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
    template<typename Dtype>
    class SliceHalfLayer : public Layer<Dtype> {
    public:
        explicit SliceHalfLayer(const LayerParameter &param)
                : Layer<Dtype>(param) {}

        virtual void LayerSetUp(const vector<Blob < Dtype> *

        >& bottom,
        const vector<Blob < Dtype>*>& top);

        virtual void Reshape(const vector<Blob < Dtype> *

        >& bottom,
        const vector<Blob < Dtype>*>& top);

        virtual inline const char *type() const { return "Slice"; }

        virtual inline int ExactNumBottomBlobs() const { return 1; }

        virtual inline int MinTopBlobs() const { return 2; }

    protected:
        virtual void Forward_cpu(const vector<Blob < Dtype> *

        >& bottom,
        const vector<Blob < Dtype>*>& top);

        virtual void Backward_cpu(const vector<Blob < Dtype> *

        >& top,
        const vector<bool> &propagate_down,
        const vector<Blob < Dtype>*>& bottom);

        int count_;
        int num_slices_;
        int slice_size_;
        int slice_axis_;
        vector<int> slice_point_;
    };
}
#endif //CAFFE_SLICE_HALF_LAYER_HPP

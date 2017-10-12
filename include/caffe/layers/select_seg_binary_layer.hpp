//
// Created by Niuchuang on 17-8-7.
//

#ifndef CAFFE_SELECT_SEG_BINARY_HPP
#define CAFFE_SELECT_SEG_BINARY_HPP

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "hdf5.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/image_dim_prefetching_layer.hpp"

namespace caffe {
    template <typename Dtype>
    class SelectSegBinaryLayer : public ImageDimPrefetchingDataLayer<Dtype> {
    public:
        explicit SelectSegBinaryLayer(const LayerParameter& param)
                : ImageDimPrefetchingDataLayer<Dtype>(param) {}
        virtual ~SelectSegBinaryLayer();
        virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "SelectSegBinary"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int ExactNumTopBlobs() const { return 3; }
        virtual inline bool AutoTopBlobs() const { return true; }

    protected:
        virtual void ShuffleImages();
        virtual void load_batch(BatchDim<Dtype>* batch);

    protected:
        Blob<Dtype> transformed_label_;
        Blob<Dtype> class_label_;

        shared_ptr<Caffe::RNG> prefetch_rng_;

        typedef struct SegItems {
            std::string imgfn;
            std::string segfn;
            int x1, y1, x2, y2;
            vector<int> cls_label;
        } SEGITEMS;

        vector<SEGITEMS> lines_;
        int lines_id_;
        int label_dim_;
    };

}


#endif //CAFFE_SELECT_SEG_BINARY_HPP

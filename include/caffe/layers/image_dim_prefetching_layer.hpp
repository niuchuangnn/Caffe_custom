#ifndef CAFFE_IMAGE_DIM_PREFETCHING_LAYER_HPP_
#define CAFFE_IMAGE_DIM_PREFETCHING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "base_data_layer.hpp"

namespace caffe {

//** Jay add
// * @brief prefetching data layer which also prefetches data dimensions
// *
// * TODO(dox): thorough documentation for Forward and proto params.
// */
//
    template <typename Dtype>
    class BatchDim {
    public:
        Blob<Dtype> data_, label_, dim_;
    };

    template <typename Dtype>
    class ImageDimPrefetchingDataLayer :
            public BaseDataLayer<Dtype>, public InternalThread {

    public:
        explicit ImageDimPrefetchingDataLayer(const LayerParameter& param);
        virtual ~ImageDimPrefetchingDataLayer() {}
        // LayerSetUp: implements common data layer setup functionality, and calls
        // DataLayerSetUp to do special data layer setup for individual layer types.
        // This method may not be overridden.
        void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top);

        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);

    protected:
        virtual void InternalThreadEntry();
        virtual void load_batch(BatchDim<Dtype>* batch) = 0;

        vector<shared_ptr<BatchDim<Dtype> > > prefetch_;
        BlockingQueue<BatchDim<Dtype>*> prefetch_free_;
        BlockingQueue<BatchDim<Dtype>*> prefetch_full_;
        BatchDim<Dtype>* prefetch_current_;

        Blob<Dtype> transformed_data_;
//    Blob<Dtype> prefetch_data_dim_;
        bool output_data_dim_;
    };

}  // namespace caffe

#endif  // CAFFE_IMAGE_DIM_PREFETCHING_LAYER_HPP_

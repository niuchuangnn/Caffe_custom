#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/image_dim_prefetching_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

/*
     * Jay add
     *
     */
/*
 notice:
 this code is based on the following implementation.
 https://bitbucket.org/deeplab/deeplab-public/
*/
    template <typename Dtype>
    ImageDimPrefetchingDataLayer<Dtype>::ImageDimPrefetchingDataLayer(
            const LayerParameter& param)
            : BaseDataLayer<Dtype>(param),
              prefetch_(param.data_param().prefetch()),
              prefetch_free_(), prefetch_full_(), prefetch_current_() {
        for (int i = 0; i < prefetch_.size(); ++i) {
            prefetch_[i].reset(new BatchDim<Dtype>());
            prefetch_free_.push(prefetch_[i].get());
        }
    }

    template <typename Dtype>
    void ImageDimPrefetchingDataLayer<Dtype>::LayerSetUp(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
        if (top.size() == 3) {
            output_data_dim_ = true;
        } else {
            output_data_dim_ = false;
        }
        // Now, start the prefetch thread. Before calling prefetch, we make two
        // cpu_data calls so that the prefetch thread does not accidentally make
        // simultaneous cudaMalloc calls when the main thread is running. In some
        // GPUs this seems to cause failures if we do not so.
        for (int i = 0; i < prefetch_.size(); ++i) {
            prefetch_[i]->data_.mutable_cpu_data();
            if (this->output_labels_) {
                prefetch_[i]->label_.mutable_cpu_data();
            }
            if (this->output_data_dim_) {
                prefetch_[i]->dim_.mutable_cpu_data();
            }
        }
        #ifndef CPU_ONLY
        if (Caffe::mode() == Caffe::GPU) {
            for (int i = 0; i < prefetch_.size(); ++i) {
                prefetch_[i]->data_.mutable_gpu_data();
                if (this->output_labels_) {
                    prefetch_[i]->label_.mutable_gpu_data();
                }
                if (this->output_data_dim_) {
                    prefetch_[i]->dim_.mutable_gpu_data();
                }
            }
        }
        #endif
        DLOG(INFO) << "Initializing prefetch";
        this->data_transformer_->InitRand();
        StartInternalThread();
        DLOG(INFO) << "Prefetch initialized.";

    }

    template <typename Dtype>
    void ImageDimPrefetchingDataLayer<Dtype>::InternalThreadEntry() {
        #ifndef CPU_ONLY
        cudaStream_t stream;
        if (Caffe::mode() == Caffe::GPU) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        }
        #endif

        try {
            while (!must_stop()) {
                BatchDim<Dtype>* batch = prefetch_free_.pop();
                load_batch(batch);
        #ifndef CPU_ONLY
                if (Caffe::mode() == Caffe::GPU) {
                    batch->data_.data().get()->async_gpu_push(stream);
                    if (this->output_labels_) {
                        batch->label_.data().get()->async_gpu_push(stream);
                    }
                    if (this->output_data_dim_) {
                        batch->dim_.data().get()->async_gpu_push(stream);
                    }
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
        #endif
                prefetch_full_.push(batch);
            }
        } catch (boost::thread_interrupted&) {
            // Interrupted exception is expected on shutdown
        }
        #ifndef CPU_ONLY
        if (Caffe::mode() == Caffe::GPU) {
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
        #endif
    }

    template <typename Dtype>
    void ImageDimPrefetchingDataLayer<Dtype>::Forward_cpu(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        if (prefetch_current_) {
            prefetch_free_.push(prefetch_current_);
        }
        prefetch_current_ = prefetch_full_.pop("Waiting for data");
        // Reshape to loaded data.
        top[0]->ReshapeLike(prefetch_current_->data_);
        top[0]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());
        if (this->output_labels_) {
            // Reshape to loaded labels.
            top[1]->ReshapeLike(prefetch_current_->label_);
            top[1]->set_cpu_data(prefetch_current_->label_.mutable_cpu_data());
        }
        if (this->output_data_dim_) {
            // Reshape to loaded dim.
            top[2]->ReshapeLike(prefetch_current_->dim_);
            top[2]->set_cpu_data(prefetch_current_->dim_.mutable_cpu_data());
        }
    }


#ifdef CPU_ONLY
STUB_GPU_FORWARD(ImageDimPrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(ImageDimPrefetchingDataLayer);

}  // namespace caffe

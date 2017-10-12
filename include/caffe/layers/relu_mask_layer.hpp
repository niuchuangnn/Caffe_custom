//
// Created by Niu Chuang on 17-7-31.
//

#ifndef CAFFE_RELU_MASK_LAYER_HPP
#define CAFFE_RELU_MASK_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Rectified Linear Unit non-linearity @f$ y = \max(0, x) @f$.
 *        The simple max is fast to compute, and the function does not saturate.
 */
    template <typename Dtype>
    class ReLUMaskLayer : public Layer<Dtype> {
    public:
        /**
         * @param param provides ReLUParameter relu_param,
         *     with ReLULayer options:
         *   - negative_slope (\b optional, default 0).
         *     the value @f$ \nu @f$ by which negative values are multiplied.
         */
        explicit ReLUMaskLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {}

        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        virtual inline int ExactNumBottomBlobs() const { return 1; }

        virtual inline int MaxTopBlobs() const { return 2; }

        virtual inline const char* type() const { return "ReLUMask"; }

    protected:
        /**
         * @param bottom input Blob vector (length 1)
         *   -# @f$ (N \times C \times H \times W) @f$
         *      the inputs @f$ x @f$
         * @param top output Blob vector (length 1)
         *   -# @f$ (N \times C \times H \times W) @f$
         *      the computed outputs @f$
         *        y = \max(0, x)
         *      @f$ by default.  If a non-zero negative_slope @f$ \nu @f$ is provided,
         *      the computed outputs are @f$ y = \max(0, x) + \nu \min(0, x) @f$.
         */
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);

        /**
         * @brief Computes the error gradient w.r.t. the ReLU inputs.
         *
         * @param top output Blob vector (length 1), providing the error gradient with
         *      respect to the outputs
         *   -# @f$ (N \times C \times H \times W) @f$
         *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
         *      with respect to computed outputs @f$ y @f$
         * @param propagate_down see Layer::Backward.
         * @param bottom input Blob vector (length 1)
         *   -# @f$ (N \times C \times H \times W) @f$
         *      the inputs @f$ x @f$; Backward fills their diff with
         *      gradients @f$
         *        \frac{\partial E}{\partial x} = \left\{
         *        \begin{array}{lr}
         *            0 & \mathrm{if} \; x \le 0 \\
         *            \frac{\partial E}{\partial y} & \mathrm{if} \; x > 0
         *        \end{array} \right.
         *      @f$ if propagate_down[0], by default.
         *      If a non-zero negative_slope @f$ \nu @f$ is provided,
         *      the computed gradients are @f$
         *        \frac{\partial E}{\partial x} = \left\{
         *        \begin{array}{lr}
         *            \nu \frac{\partial E}{\partial y} & \mathrm{if} \; x \le 0 \\
         *            \frac{\partial E}{\partial y} & \mathrm{if} \; x > 0
         *        \end{array} \right.
         *      @f$.
         */
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    };

}  // namespace caffe

#endif //CAFFE_RELU_MASK_LAYER_HPP

﻿#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include "../../utilities/cuda/cudnn_workspace.hh"
#include "cudnn_impl.h"

namespace refactor::kernel::cudnn {
    using namespace runtime;

    Routine ConvInfo::lower() const {
        // RAII for closure
        struct Descriptors {
            cudnnTensorDescriptor_t x, y;
            cudnnFilterDescriptor_t w;
            cudnnConvolutionDescriptor_t conv;
            cudnnConvolutionFwdAlgo_t algo;

            Descriptors() {
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&x));
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&y));
                CUDNN_ASSERT(cudnnCreateFilterDescriptor(&w));
                CUDNN_ASSERT(cudnnCreateConvolutionDescriptor(&conv));
            }
            ~Descriptors() {
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(x));
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(y));
                CUDNN_ASSERT(cudnnDestroyFilterDescriptor(w));
                CUDNN_ASSERT(cudnnDestroyConvolutionDescriptor(conv));
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>();

        d->algo = static_cast<cudnnConvolutionFwdAlgo_t>(algo);

        TODO("");

        // nvcc at c++11 doesn't support real move capture
        return [d_ = std::move(d)](Resources &res, Addresses inputs, Addresses outputs) {
            // fetch cudnn handle from resources
            auto handle = res.fetchOrStore<CudnnContext>()->handle;
            auto const &workspace = *res.fetchOrStore<CudnnWorkspace>();
            auto const &d = *d_;
            // name inputs and outputs
            auto x = inputs[0],
                 w = inputs[1];
            auto y = outputs[0];
            // build alpha/beta for double
            float alpha = 1, beta = 0;
            CUDNN_ASSERT(cudnnConvolutionForward(
                handle,
                &alpha,
                d.x, x,
                d.w, w,
                d.conv, d.algo,
                workspace.ptr, workspace.size,
                &beta,
                d.y, y));
        };
    }

}// namespace refactor::kernel::cudnn
#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include "cudnn_kernel.hh"

namespace refactor::kernel {
    using namespace cudnn;
    using namespace runtime;

    Routine SoftmaxCudnn::lower(Resources &res) const noexcept {
        // RAII for closure
        struct Descriptors {
            cudnnTensorDescriptor_t x, y;
            cudnnSoftmaxAlgorithm_t algo;

            Descriptors(cudnnSoftmaxAlgorithm_t algo_) : algo(algo_) {
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&x));
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&y));
            }
            ~Descriptors() {
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(x));
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(y));
            }
            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };

        auto d = std::make_shared<Descriptors>(static_cast<cudnnSoftmaxAlgorithm_t>(algo));
        setCudnnTensor(d->x, dataType, dim);
        setCudnnTensor(d->y, dataType, dim);

        res.fetchOrStore<CudnnContext>();

        return [d_ = std::move(d)](Resources &res, void const **inputs, void **outputs) {
            auto const &d = *d_;
            auto x = inputs[0];
            auto y = outputs[0];
            float alpha = 1, beta = 0;
            CUDNN_ASSERT(cudnnSoftmaxForward(
                res.fetchOrStore<CudnnContext>()->handle,
                d.algo,
                CUDNN_SOFTMAX_MODE_CHANNEL,
                &alpha, d.x, x,
                &beta, d.y, y));
        };
    }
}// namespace refactor::kernel

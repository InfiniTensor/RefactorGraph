#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include "cudnn_kernel.hh"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {
    using namespace cudnn;
    using namespace runtime;
    Routine SoftmaxCudnn::lower() const noexcept {
        struct Descriptors {
            cudnnTensorDescriptor_t x, y;
            cudnnSoftmaxAlgorithm_t algo;
            cudnnSoftmaxMode_t mode;

            Descriptors() {
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
        auto d = std::make_shared<Descriptors>();
        d->algo = static_cast<cudnnSoftmaxAlgorithm_t>(algo);
        d->mode = static_cast<cudnnSoftmaxMode_t>(mode);
        setCudnnTensor(d->x, dataType, dim);
        setCudnnTensor(d->y, dataType, dim);

        return [d_ = std::move(d)](Resources &res, void const **inputs, void **outputs) {
            auto handle = res.fetchOrStore<CudnnContext>()->handle;
            auto const &d = *d_;
            auto x = inputs[0];
            auto y = outputs[0];
            float alpha = 1, beta = 0;
            CUDNN_ASSERT(cudnnSoftmaxForward(
                handle,
                d.algo,
                d.mode,
                &alpha,
                d.x,
                x,
                &beta,
                d.y,
                y));
        };
    }
}// namespace refactor::kernel
#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include "cudnn_kernel.hh"

namespace refactor::kernel {
    using namespace cudnn;
    using namespace runtime;

    auto SoftmaxCudnn::lower(Resources &res) const noexcept -> RoutineWorkspace {
        // RAII for closure
        struct Descriptors {
            cudnnTensorDescriptor_t t;
            cudnnSoftmaxAlgorithm_t algo;

            Descriptors(cudnnSoftmaxAlgorithm_t algo_) : algo(algo_) {
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&t));
            }
            ~Descriptors() {
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(t));
            }
            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };

        auto d = std::make_shared<Descriptors>(static_cast<cudnnSoftmaxAlgorithm_t>(algo));
        CUDNN_ASSERT(cudnnSetTensor4dDescriptor(
            d->t, CUDNN_TENSOR_NCHW, cudnnDataTypeConvert(dataType), pre, mid, post, 1));

        res.fetchOrStore<CudnnContext>();

        return [d_ = std::move(d)](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            auto const &d = *d_;
            float alpha = 1, beta = 0;
            CUDNN_ASSERT(cudnnSoftmaxForward(
                res.fetchOrStore<CudnnContext>()->handle,
                d.algo,
                CUDNN_SOFTMAX_MODE_CHANNEL,
                &alpha, d.t, inputs[0],
                &beta, d.t, outputs[0]));
        };
    }
}// namespace refactor::kernel

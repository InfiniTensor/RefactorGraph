#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include "cudnn_activation_kernel.hh"
#include <cudnn.h>

namespace refactor::kernel {
    using namespace cudnn;
    using namespace runtime;
    using Ty = SimpleUnaryType;

    Routine ActivationCudnn::lower(Resources &res) const noexcept {
        // RAII for closure
        struct Descriptors {
            cudnnActivationDescriptor_t activation;
            cudnnTensorDescriptor_t tensor;

            Descriptors() : activation(nullptr), tensor(nullptr) {
                CUDNN_ASSERT(cudnnCreateActivationDescriptor(&activation));
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&tensor));
            }
            ~Descriptors() {
                CUDNN_ASSERT(cudnnDestroyActivationDescriptor(activation));
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(tensor));
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>();

        // clang-format off
        cudnnActivationMode_t
        mode = type == Ty::Relu    ? CUDNN_ACTIVATION_RELU
             : type == Ty::Sigmoid ? CUDNN_ACTIVATION_SIGMOID
             : type == Ty::Tanh    ? CUDNN_ACTIVATION_TANH
             : UNREACHABLEX(cudnnActivationMode_t, "");
        // clang-format on

        CUDNN_ASSERT(cudnnSetActivationDescriptor(d->activation, mode, CUDNN_PROPAGATE_NAN, 0.0));
        CUDNN_ASSERT(cudnnSetTensor4dDescriptor(d->tensor, CUDNN_TENSOR_NCHW, cudnnDataTypeConvert(dataType), 1, 1, 1, size));

        res.fetchOrStore<CudnnContext>();
        // nvcc at c++11 doesn't support real move capture
        return [d = std::move(d)](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            // fetch cudnn handle from resources
            auto handle = res.fetchOrStore<CudnnContext>()->handle;
            // name inputs and outputs
            auto x = inputs[0];
            auto y = outputs[0];
            // call cudnn activation
            float alpha = 1, beta = 0;
            CUDNN_ASSERT(cudnnActivationForward(handle, d->activation, &alpha, d->tensor, x, &beta, d->tensor, y));
        };
    }

}// namespace refactor::kernel

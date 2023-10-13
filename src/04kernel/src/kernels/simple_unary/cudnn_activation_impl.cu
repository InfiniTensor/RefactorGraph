#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include "cudnn_activation_impl.hh"
#include <cudnn.h>

namespace refactor::kernel::cudnn {
    using namespace runtime;
    using Op = SimpleUnaryType;

    Routine lower(Op op, common::DataType dt, int size) noexcept {
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
        mode = op == Op::Relu ? CUDNN_ACTIVATION_RELU
             : op == Op::Sigmoid ? CUDNN_ACTIVATION_SIGMOID
             : op == Op::Tanh ? CUDNN_ACTIVATION_TANH
             : UNREACHABLEX(cudnnActivationMode_t, "");
        // clang-format on
        int stride = 1;

        CUDNN_ASSERT(cudnnSetActivationDescriptor(d->activation, mode, CUDNN_PROPAGATE_NAN, 0.0));
        CUDNN_ASSERT(cudnnSetTensorNdDescriptor(d->tensor, cudnnDataTypeConvert(dt), 1, &size, &stride));

        // nvcc at c++11 doesn't support real move capture
        return [d = std::move(d)](Resources &res, Addresses inputs, Addresses outputs) {
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

}// namespace refactor::kernel::cudnn

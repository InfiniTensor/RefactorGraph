#include "cudnn_activation_kernel.hh"
#include "kernel/collectors/simple_unary.h"
#include <unordered_set>

#ifdef USE_CUDA
#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include <cudnn.h>
#endif

namespace refactor::kernel {
    using K = ActivationCudnn;
    using DT = DataType;
    using Op = SimpleUnaryType;

    K::ActivationCudnn(Op type_, DT dataType_, int size_) noexcept
        : Kernel(), type(type_), dataType(dataType_), size(size_) {}

    auto K::build(Op op, Tensor const &a) noexcept -> KernelBox {
        static const std::unordered_set<Op> ARTHIMETIC{Op::Sigmoid, Op::Relu, Op::Tanh};

#ifndef USE_CUDA
        return nullptr;
#endif

        return ARTHIMETIC.contains(op) && a.dataType.isCpuNumberic()
                   ? std::make_unique<K>(op, a.dataType, static_cast<int>(a.elementsSize()))
                   : nullptr;
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing activation using CUDNN";
    }

#ifdef USE_CUDA

    auto ActivationCudnn::lower(Resources &res) const -> RoutineWorkspace {
        using namespace cudnn;
        using namespace runtime;
        using Ty = SimpleUnaryType;

        // RAII for closure
        struct Descriptors {
            cudnnActivationDescriptor_t activation;
            cudnnTensorDescriptor_t tensor;

            Descriptors() : activation(nullptr), tensor(nullptr) {
                CUDNN_ASSERT(cudnnCreateActivationDescriptor(&activation));
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&tensor));
            }
            ~Descriptors() noexcept(false) {
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

#endif

}// namespace refactor::kernel

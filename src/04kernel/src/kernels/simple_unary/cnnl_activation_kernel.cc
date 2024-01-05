#include "cnnl_activation_kernel.hh"
#include "kernel/collectors/simple_unary.h"
#include <unordered_set>

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#include <cnnl.h>
#endif

namespace refactor::kernel {
    using K = ActivationCnnl;
    using DT = DataType;
    using Op = SimpleUnaryType;

    K::ActivationCnnl(Op type_, DT dataType_, int size_) noexcept
        : Kernel(), type(type_), dataType(dataType_), size(size_) {}

    auto K::build(Op op, Tensor const &a) noexcept -> KernelBox {
        static const std::unordered_set<Op> ARTHIMETIC{Op::Sigmoid, Op::Relu, Op::Tanh};

#ifndef USE_BANG
        return nullptr;
#endif

        return ARTHIMETIC.contains(op)
                   ? std::make_unique<K>(op, a.dataType, static_cast<int>(a.elementsSize()))
                   : nullptr;
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing activation using CNNL";
    }

#ifdef USE_BANG

    auto ActivationCnnl::lower(Resources &res) const -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;
        using Ty = SimpleUnaryType;

        // RAII for closure
        struct Descriptors {
            cnnlActivationDescriptor_t activation;
            cnnlTensorDescriptor_t tensor;

            Descriptors() : activation(nullptr), tensor(nullptr) {
                CNNL_ASSERT(cnnlCreateActivationDescriptor(&activation));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&tensor));
            }
            ~Descriptors() noexcept(false) {
                CNNL_ASSERT(cnnlDestroyActivationDescriptor(activation));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(tensor));
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>();

        // clang-format off
        auto mode = type == Ty::Relu    ? CNNL_ACTIVATION_RELU
                  : type == Ty::Sigmoid ? CNNL_ACTIVATION_SIGMOID
                  : type == Ty::Tanh    ? CNNL_ACTIVATION_TANH
                  : UNREACHABLEX(cnnlActivationMode_t, "");
        // clang-format on

        setCnnlTensor(d->tensor, dataType, slice(&size, 1));
        CNNL_ASSERT(cnnlSetActivationDescriptor_v2(d->activation, mode, CNNL_ACTIVATION_HIGH_PRECISION,
                                                   CNNL_NOT_PROPAGATE_NAN, 0.0));

        res.fetchOrStore<CnnlContext>();
        return [d = std::move(d)]//
            (Resources & res, void *, void const *const *inputs, void *const *outputs) {
                float alpha = 1, beta = 0;
                CNNL_ASSERT(cnnlActivationForward(
                    res.fetchOrStore<CnnlContext>()->handle,
                    d->activation,
                    &alpha, d->tensor, inputs[0],
                    &beta, d->tensor, outputs[0]));
            };
    }

#endif

}// namespace refactor::kernel

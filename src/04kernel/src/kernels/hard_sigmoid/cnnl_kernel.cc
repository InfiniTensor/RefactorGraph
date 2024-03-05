#include "cnnl_kernel.hh"
#include "kernel/collectors/hard_sigmoid.h"
#include <unordered_set>

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#include <cnnl.h>
#endif

namespace refactor::kernel {
    using K = HardSigmoidCnnl;
    using DT = DataType;

    K::HardSigmoidCnnl(float alpha_, float beta_, DT dataType_, int size_) noexcept
        : Kernel(), alpha(alpha_), beta(beta_), dataType(dataType_), size(size_) {}

    auto K::build(float alpha_, float beta_, Tensor const &a) noexcept -> KernelBox {

#ifndef USE_BANG
        return nullptr;
#endif

        return std::make_unique<K>(alpha_, beta_, a.dataType, a.elementsSize());
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing hardsigmoid using CNNL";
    }

#ifdef USE_BANG

    auto HardSigmoidCnnl::lower(Resources &res) const -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;

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

        setCnnlTensor(d->tensor, dataType, slice(&size, 1));
        CNNL_ASSERT(cnnlSetActivationDescriptor_v5(d->activation, CNNL_ACTIVATION_HARDSIGMOID,
                                                   CNNL_ACTIVATION_HIGH_PRECISION,
                                                   CNNL_NOT_PROPAGATE_NAN, 0.0,
                                                   0.0, alpha, beta, true));

        res.fetchOrStore<CnnlContext>();
        return [d = std::move(d)]//
            (Resources & res, void *, void const *const *inputs, void *const *outputs) {
                float alpha = 1.f, beta = 0.f;
                CNNL_ASSERT(cnnlActivationForward(
                    res.fetchOrStore<CnnlContext>()->handle,
                    d->activation,
                    &alpha, d->tensor, inputs[0],
                    &beta, d->tensor, outputs[0]));
            };
    }

#endif

}// namespace refactor::kernel

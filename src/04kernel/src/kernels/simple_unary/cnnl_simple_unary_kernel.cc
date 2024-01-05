#include "cnnl_simple_unary_kernel.hh"
#include "kernel/collectors/simple_unary.h"
#include <unordered_set>

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#include <cnnl.h>
#endif

namespace refactor::kernel {
    using K = SimpleUnaryCnnl;
    using DT = DataType;
    using Op = SimpleUnaryType;

    K::SimpleUnaryCnnl(Op type_, DT dataType_, int size_) noexcept
        : Kernel(), type(type_), dataType(dataType_), size(size_) {}

    auto K::build(Op op, Tensor const &a) noexcept -> KernelBox {
        static const std::unordered_set<Op> supportedOp{Op::Abs, Op::Sqrt, Op::Neg};

#ifndef USE_BANG
        return nullptr;
#endif

        return supportedOp.contains(op)
                   ? std::make_unique<K>(op, a.dataType, static_cast<int>(a.elementsSize()))
                   : nullptr;
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing simple unary using CNNL";
    }

#ifdef USE_BANG

    auto SimpleUnaryCnnl::lower(Resources &res) const -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;
        using Ty = SimpleUnaryType;

        // RAII for closure
        struct Descriptors {
            cnnlTensorDescriptor_t tensor;

            Descriptors() : tensor(nullptr) {
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&tensor));
            }
            ~Descriptors() noexcept(false) {
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(tensor));
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>();

        setCnnlTensor(d->tensor, dataType, slice(&size, 1));

        auto cnnlUnaryForward = [this](cnnlHandle_t handle,
                                       const cnnlTensorDescriptor_t x_desc,
                                       const void *x,
                                       const cnnlTensorDescriptor_t y_desc,
                                       void *y) -> cnnlStatus_t {
            switch (this->type) {
                case Ty::Abs:
                    return cnnlAbs(handle, x_desc, x, y_desc, y);
                case Ty::Neg:
                    return cnnlNegTensor(handle, x_desc, x, y_desc, y);
                case Ty::Sqrt:
                    return cnnlSqrt_v2(handle, CNNL_COMPUTATION_HIGH_PRECISION, x_desc, x, y_desc, y);
                default:
                    UNREACHABLE();
            }
        };

        res.fetchOrStore<CnnlContext>();
        return [d = std::move(d), cnnlUnaryForward]//
            (Resources & res, void *, void const *const *inputs, void *const *outputs) {
                CNNL_ASSERT(cnnlUnaryForward(
                    res.fetchOrStore<CnnlContext>()->handle,
                    d->tensor, inputs[0],
                    d->tensor, outputs[0]));
            };
    }

#endif

}// namespace refactor::kernel

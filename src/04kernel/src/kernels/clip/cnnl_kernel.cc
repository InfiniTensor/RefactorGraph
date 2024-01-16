#include "cnnl_kernel.hh"

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#endif

namespace refactor::kernel {
    using K = ClipCnnl;

    K::ClipCnnl(decltype(dataType) dt,
                decltype(shape) shape_,
                decltype(hasMax) hasMax_) noexcept
        : dataType(dt), shape(shape_), hasMax(hasMax_) {
    }

    auto K::build(Tensor const &data, bool hasMax) noexcept -> KernelBox {
        return data.dataType.isCpuNumberic()
                   ? std::make_unique<K>(data.dataType,
                                         std::vector<int>(data.shape.begin(), data.shape.end()),
                                         hasMax)
                   : nullptr;
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing clip operation using CNNL";
    }

#ifdef USE_BANG
    auto K::lower(Resources &res) const noexcept -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;

        struct Descriptors {
            cnnlTensorDescriptor_t t;

            Descriptors() : t(nullptr) {
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&t));
            }
            ~Descriptors() noexcept(false) {
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(t));
            }
        };
        auto d = std::make_shared<Descriptors>();
        setCnnlTensor(d->t, dataType, slice(shape.data(), shape.size()));

        res.fetchOrStore<CnnlContext>();
        return [d = std::move(d), hasMax = this->hasMax](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            CNNL_ASSERT(cnnlClip_v2(res.fetchOrStore<CnnlContext>()->handle,
                                    CNNL_POINTER_MODE_DEVICE, d->t,
                                    inputs[0], inputs[1], hasMax ? inputs[2] : nullptr,
                                    d->t, outputs[0]));
            BANG_ASSERT(cnrtQueueSync(res.fetchOrStore<CnnlContext>()->queue));
        };
    }

#endif

}// namespace refactor::kernel

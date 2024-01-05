#include "cnnl_kernel.hh"

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#include <cnnl.h>
#endif

namespace refactor::kernel {
    using K = TransposeCnnl;
    using Info = TransposeInfo;

    K::TransposeCnnl(DataType dataType_, Shape dimIn_, Shape dimOut_, Permutation perm_) noexcept
        : Kernel(), dataType(dataType_), dimIn(std::move(dimIn_)),
          dimOut(std::move(dimOut_)), perm(std::move(perm_)) {}

    auto K::build(DataType dataType, Shape shape_, Permutation perm_) noexcept -> KernelBox {
#ifndef USE_BANG
        return nullptr;
#endif
        Shape dimOut_;
        for (uint32_t i = 0; i < shape_.size(); i++) {
            dimOut_.push_back(shape_[perm_[i]]);
        }
        return std::make_unique<K>(dataType, std::move(shape_), std::move(dimOut_), std::move(perm_));
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing transpose operation using CNNL";
    }

#ifdef USE_BANG
    auto TransposeCnnl::lower(Resources &res) const noexcept -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;
        using DT = DataType;

        struct Descriptors {
            cnnlTensorDescriptor_t x, y;
            cnnlTransposeDescriptor_t trans;
            bool f32;

            explicit Descriptors(decltype(f32) f32_)
                : x(nullptr), y(nullptr), trans(nullptr), f32(f32_) {
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&x));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&y));
                CNNL_ASSERT(cnnlCreateTransposeDescriptor(&trans));
            }
            ~Descriptors() noexcept(false) {
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(x));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(y));
                CNNL_ASSERT(cnnlDestroyTransposeDescriptor(trans));
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };

        auto d = std::make_shared<Descriptors>(dataType != DT::F64);
        setCnnlTensor(d->x, dataType, slice((int *)(dimIn.data()), dimIn.size()));
        setCnnlTensor(d->y, dataType, slice((int *)(dimOut.data()), dimOut.size()));
        CNNL_ASSERT(cnnlSetTransposeDescriptor(d->trans, perm.size(), (int *)perm.data()));

        auto handle = res.fetchOrStore<CnnlContext>()->handle;
        size_t workspaceSize;
        CNNL_ASSERT(cnnlGetTransposeWorkspaceSize(handle, d->x, d->trans, &workspaceSize));

        res.fetchOrStore<CnnlContext>();
        auto routine = [d = std::move(d), workspaceSize](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            // fetch cnnl handle from resources
            auto handle = res.fetchOrStore<CnnlContext>()->handle;

            // name inputs and outputs
            auto x = inputs[0];
            auto y = outputs[0];

            CNNL_ASSERT(cnnlTranspose_v2(handle, d->trans, d->x, x,
                                         d->y, y, workspace, workspaceSize));
        };

        return {std::move(routine), workspaceSize};
    }
#endif

}// namespace refactor::kernel

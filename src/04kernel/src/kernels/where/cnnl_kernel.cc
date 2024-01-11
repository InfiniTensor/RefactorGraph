#include "cnnl_kernel.hh"

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#include <cnnl.h>
#endif

namespace refactor::kernel {
    using K = WhereCnnl;

    K::WhereCnnl(decltype(info) info_) noexcept
        : Kernel(), info(info_) {}

    auto K::build(TensorRefs const &inputs, TensorRefs const &outputs) noexcept -> KernelBox {
#ifndef USE_BANG
        return nullptr;
#endif
        return std::make_unique<K>(decltype(info) {
            inputs[1].get().dataType,
            inputs[0].get().shape,
            inputs[1].get().shape,
            inputs[2].get().shape,
            outputs[0].get().shape,
        });
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing where operation using CNNL";
    }

#ifdef USE_BANG
    auto WhereCnnl::lower(Resources &res) const noexcept -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;
        using DT = DataType;

        struct Descriptors {
            cnnlTensorDescriptor_t cond, x, y, ans;
            bool f32;

            explicit Descriptors(decltype(f32) f32_)
                : cond(nullptr), x(nullptr), y(nullptr),
                  ans(nullptr), f32(f32_) {
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&cond));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&x));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&y));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&ans));
            }
            ~Descriptors() noexcept(false) {
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(cond));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(x));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(y));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(ans));
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>(info.dataType != DT::F64);

        std::vector<int> cDim(info.condDim.begin(), info.condDim.end()),
            xDim(info.thenDim.begin(), info.thenDim.end()),
            yDim(info.elseDim.begin(), info.elseDim.end()),
            ansDim(info.outputDim.begin(), info.outputDim.end());

        auto rightAlign = [](std::vector<int> &dim, uint32_t targetLength) {
            if (dim.size() < targetLength) {
                dim.insert(dim.begin(), targetLength - dim.size(), 1);
            }
        };
        if (ansDim.size() == 0) {
            ansDim.push_back(1);
        }
        rightAlign(cDim, ansDim.size());
        rightAlign(xDim, ansDim.size());
        rightAlign(yDim, ansDim.size());

        CNNL_ASSERT(cnnlSetTensorDescriptor(d->cond, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(DT::Bool), cDim.size(), cDim.data()));
        CNNL_ASSERT(cnnlSetTensorDescriptor(d->x, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(info.dataType), xDim.size(), xDim.data()));
        CNNL_ASSERT(cnnlSetTensorDescriptor(d->y, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(info.dataType), yDim.size(), yDim.data()));
        CNNL_ASSERT(cnnlSetTensorDescriptor(d->ans, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(info.dataType), ansDim.size(), ansDim.data()));

        auto handle = res.fetchOrStore<CnnlContext>()->handle;
        size_t workspaceSize;
        CNNL_ASSERT(cnnlGetSelectV2WorkspaceSize(handle, d->cond, d->x, d->y, &workspaceSize));

        res.fetchOrStore<CnnlContext>();
        auto routine = [d = std::move(d), workspaceSize](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            // fetch cnnl handle from resources
            auto handle = res.fetchOrStore<CnnlContext>()->handle;
            auto cond = inputs[0],
                 x = inputs[1],
                 y = inputs[2];
            auto ans = outputs[0];

            CNNL_ASSERT(cnnlSelectV2(
                handle, d->cond, cond, d->x, x,
                d->y, y, workspace, workspaceSize,
                d->ans, ans));

            cnrtQueueSync(res.fetchOrStore<CnnlContext>()->queue);
        };

        return {std::move(routine), workspaceSize};
    }
#endif

}// namespace refactor::kernel

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
        std::vector<int> cDim(inputs[0].get().shape.begin(), inputs[0].get().shape.end()),
            xDim(inputs[1].get().shape.begin(), inputs[1].get().shape.end()),
            yDim(inputs[2].get().shape.begin(), inputs[2].get().shape.end()),
            ansDim(outputs[0].get().shape.begin(), outputs[0].get().shape.end());
        if (ansDim.size() == 0) {
            ansDim.push_back(1);
        }
        if (xDim.size() == 0) {
            xDim.push_back(1);
        }
        if (yDim.size() == 0) {
            yDim.push_back(1);
        }
        if (cDim.size() == 0) {
            cDim.push_back(1);
        }
        return std::make_unique<K>(decltype(info){
            inputs[1].get().dataType, cDim, xDim, yDim, ansDim});
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

            explicit Descriptors()
                : cond(nullptr), x(nullptr), y(nullptr),
                  ans(nullptr) {
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
        auto d = std::make_shared<Descriptors>();

        CNNL_ASSERT(cnnlSetTensorDescriptor(
            d->cond, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(DT::Bool),
            info.condDim.size(), info.condDim.data()));
        CNNL_ASSERT(cnnlSetTensorDescriptor(
            d->x, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(info.dataType),
            info.thenDim.size(), info.thenDim.data()));
        CNNL_ASSERT(cnnlSetTensorDescriptor(
            d->y, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(info.dataType),
            info.elseDim.size(), info.elseDim.data()));
        CNNL_ASSERT(cnnlSetTensorDescriptor(
            d->ans, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(info.dataType),
            info.outputDim.size(), info.outputDim.data()));

        auto handle = res.fetchOrStore<CnnlContext>()->handle;
        size_t workspaceSize;
        CNNL_ASSERT(cnnlGetSelectV2WorkspaceSize(handle, d->cond, d->x, d->y, &workspaceSize));

        res.fetchOrStore<CnnlContext>();
        auto routine = [d = std::move(d), workspaceSize](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {

            CNNL_ASSERT(cnnlSelectV2(
                res.fetchOrStore<CnnlContext>()->handle,
                d->cond, inputs[0], d->x, inputs[1],
                d->y, inputs[2], workspace, workspaceSize,
                d->ans, outputs[0]));

            res.fetchOrStore<CnnlContext>()->queueSync();
        };

        return {std::move(routine), workspaceSize};
    }
#endif

}// namespace refactor::kernel

#include "cnnl_kernel.hh"

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#endif

namespace refactor::kernel {
    using K = ScatterNDCnnl;

    K::ScatterNDCnnl(decltype(info) info_)
        : Kernel(), info(std::move(info_)) {}

    auto K::build(TensorRefs inputs, TensorRefs outputs) noexcept -> KernelBox {
#ifndef USE_BANG
        return nullptr;
#endif
        return std::make_unique<ScatterNDCnnl>(decltype(info){
            inputs[0].get().dataType,
            inputs[1].get().dataType,
            inputs[2].get().dataType,
            std::vector<int>(inputs[0].get().shape.begin(), inputs[0].get().shape.end()),
            std::vector<int>(inputs[1].get().shape.begin(), inputs[1].get().shape.end()),
            std::vector<int>(inputs[2].get().shape.begin(), inputs[2].get().shape.end()),
            std::vector<int>(outputs[0].get().shape.begin(), outputs[0].get().shape.end()),
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
        return "Performing scatterNd operation using CNNL";
    }

#ifdef USE_BANG
    auto K::lower(Resources &res) const noexcept -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;

        struct Descriptors {
            cnnlTensorDescriptor_t inDesc, indexDesc, updateDesc, outDesc;

            Descriptors() : inDesc(nullptr), indexDesc(nullptr),
                            updateDesc(nullptr), outDesc(nullptr) {
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&inDesc));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&indexDesc));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&updateDesc));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&outDesc));
            }
            ~Descriptors() noexcept(false) {
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(inDesc));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(indexDesc));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(updateDesc));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(outDesc));
            }
        };
        auto d = std::make_shared<Descriptors>();
        CNNL_ASSERT(cnnlSetTensorDescriptor(
            d->inDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(info.dataType),
            info.inDim.size(), info.inDim.data()));
        CNNL_ASSERT(cnnlSetTensorDescriptor(
            d->indexDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(info.indexDataType),
            info.indexDim.size(), info.indexDim.data()));
        CNNL_ASSERT(cnnlSetTensorDescriptor(
            d->updateDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(info.updateDataType),
            info.updateDim.size(), info.updateDim.data()));
        CNNL_ASSERT(cnnlSetTensorDescriptor(
            d->outDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(info.updateDataType),
            info.outDim.size(), info.outDim.data()));
    
        res.fetchOrStore<CnnlContext>();
        return [d = std::move(d)](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            CNNL_ASSERT(cnnlScatterNd_v2(res.fetchOrStore<CnnlContext>()->handle, CNNL_SCATTERND_UPDATE,
                                         d->indexDesc, inputs[1], d->updateDesc, inputs[2],
                                         d->inDesc, inputs[0], d->outDesc, outputs[0]));
            BANG_ASSERT(cnrtQueueSync(res.fetchOrStore<CnnlContext>()->queue));
        };
    }
#endif

}// namespace refactor::kernel

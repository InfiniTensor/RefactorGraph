#include "cnnl_kernel.hh"

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#endif

namespace refactor::kernel {
    using K = GatherCnnl;

    K::GatherCnnl(decltype(info) info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(int axis, Tensor input, Tensor index, Tensor output) noexcept -> KernelBox {
#ifndef USE_BANG
        return nullptr;
#endif
        return std::make_unique<K>(decltype(info){
            input.dataType,
            index.dataType,
            axis,
            std::vector<int>(input.shape.begin(), input.shape.end()),
            std::vector<int>(index.shape.begin(), index.shape.end()),
            std::vector<int>(output.shape.begin(), output.shape.end()),
        });
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing gather using CNNL";
    }

#ifdef USE_BANG
    auto K::lower(Resources &res) const noexcept -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;

        struct Descriptors {
            cnnlTensorDescriptor_t inDesc, indexDesc, outDesc;

            Descriptors() : inDesc(nullptr), indexDesc(nullptr), outDesc(nullptr) {
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&inDesc));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&indexDesc));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&outDesc));
            }
            ~Descriptors() noexcept(false) {
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(inDesc));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(indexDesc));
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
            d->outDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(info.dataType),
            info.outDim.size(), info.outDim.data()));

        size_t workspaceSize = info.inDim.size() * sizeof(int);

        res.fetchOrStore<CnnlContext>();
        auto routine = [d = std::move(d),
                        shape = info.inDim.data(), workspaceSize,
                        dim = info.axis](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            BANG_ASSERT(cnrtMemcpy(workspace, (void*) shape, workspaceSize, CNRT_MEM_TRANS_DIR_HOST2DEV));
            CNNL_ASSERT(cnnlGatherV2(res.fetchOrStore<CnnlContext>()->handle, dim,
                                     d->inDesc, inputs[0], reinterpret_cast<const int *>(workspace),
                                      d->indexDesc, reinterpret_cast<const int *>(inputs[1]),
                                     d->outDesc, outputs[0]));
            BANG_ASSERT(cnrtQueueSync(res.fetchOrStore<CnnlContext>()->queue));
        };

        return {std::move(routine), workspaceSize};
    }
#endif

}// namespace refactor::kernel

#include "cnnl_kernel.hh"

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#endif

namespace refactor::kernel {
    using K = ExpandCnnl;

    K::ExpandCnnl(ExpandInfoCnnl info_) noexcept
        : Kernel(), info(info_) {}

    auto K::build(Tensor const &input, Tensor const &output) noexcept -> KernelBox {
#ifndef USE_BANG
        return nullptr;
#endif
        return std::make_unique<K>(ExpandInfoCnnl(
            input.dataType,
            slice(input.shape.data(), input.rank()),
            slice(output.shape.data(), output.rank())
        ));
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing expand operation using CNNL";
    }

#ifdef USE_BANG
    auto ExpandCnnl::lower(Resources &res) const noexcept -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;

        struct Descriptors {
            cnnlTensorDescriptor_t inDesc, outDesc;

            Descriptors() : inDesc(nullptr), outDesc(nullptr) {
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&inDesc));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&outDesc));
            }
            ~Descriptors() noexcept(false) {
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(inDesc));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(outDesc));
            }
        };
        auto d = std::make_shared<Descriptors>();
        std::vector<int> in(info.inDims.begin(), info.inDims.end()),
            out(info.outDims.begin(), info.outDims.end());
        setCnnlTensor(d->inDesc, info.dataType, slice(in.data(), in.size()));
        setCnnlTensor(d->outDesc, info.dataType, slice(out.data(), out.size()));

        res.fetchOrStore<CnnlContext>();
        return [d = std::move(d)](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            CNNL_ASSERT(cnnlExpand(res.fetchOrStore<CnnlContext>()->handle,
                                   d->inDesc, inputs[0], d->outDesc, outputs[0]));
        };
    }
#endif

}// namespace refactor::kernel

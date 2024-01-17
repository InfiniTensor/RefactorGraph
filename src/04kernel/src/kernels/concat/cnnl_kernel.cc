#include "cnnl_kernel.hh"

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#include <cnnl.h>
#endif

namespace refactor::kernel {
    using K = ConcatCnnl;

    K::ConcatCnnl(SplitInfoCnnl info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(int axis, TensorRefs inputs, Tensor const &output) noexcept -> KernelBox {
#ifndef USE_BANG
        return nullptr;
#endif
        return std::make_unique<K>(SplitInfoCnnl(axis, output, inputs));
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing split operation using CNNL";
    }

#ifdef USE_BANG
    auto ConcatCnnl::lower(Resources &res) const -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;
        using DT = DataType;

        struct Descriptors {
            cnnlTensorDescriptor_t in;
            std::vector<cnnlTensorDescriptor_t> out;
            bool f32;

            explicit Descriptors(int n, decltype(f32) f32_)
                : in(nullptr),
                  out(std::vector<cnnlTensorDescriptor_t>(n, nullptr)),
                  f32(f32_) {
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&in));
                for (auto i = 0; i < n; i++) {
                    CNNL_ASSERT(cnnlCreateTensorDescriptor(&out[i]));
                }
            }
            ~Descriptors() noexcept(false) {
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(in));
                for (auto i = 0; i < out.size(); i++) {
                    CNNL_ASSERT(cnnlDestroyTensorDescriptor(out[i]));
                }
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>(info.num, info.dataType != DT::F64);
        setCnnlTensor(d->in, info.dataType, slice(info.inDim.data(), info.inDim.size()));
        for (auto i = 0; i < info.outDims.size(); i++) {
            setCnnlTensor(d->out[i], info.dataType, slice(info.outDims[i].data(), info.outDims[i].size()));
        }

        auto handle = res.fetchOrStore<CnnlContext>()->handle;
        size_t workspaceSize;
        CNNL_ASSERT(cnnlGetSplitWorkspaceSize(handle, info.num, &workspaceSize));

        res.fetchOrStore<CnnlContext>();
        auto routine = [d = std::move(d), n = info.num, axis = info.axis, workspaceSize](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            // fetch cnnl handle from resources
            auto handle = res.fetchOrStore<CnnlContext>()->handle;

            const void *argv[n];
            for (auto i = 0; i < n; i++) {
                argv[i] = inputs[i];
            }

            CNNL_ASSERT(cnnlConcat(
                handle, n, axis, d->out.data(), argv,
                workspace, workspaceSize, d->in, outputs[0]));
        };

        return {std::move(routine), workspaceSize};
    }

#endif

}// namespace refactor::kernel

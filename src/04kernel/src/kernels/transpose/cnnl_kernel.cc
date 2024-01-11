#include "cnnl_kernel.hh"

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#include <cnnl.h>
#endif

namespace refactor::kernel {
    using K = TransposeCnnl;
    using Info = TransposeInfoCnnl;

    Info::TransposeInfoCnnl(DataType dataType_, std::vector<int> input_, std::vector<int> perm_) 
        : dataType(dataType_), inDim(input_), perm(perm_) {
        ASSERT(input_.size() == perm_.size(), "Unreachable");
        for (uint32_t i = 0; i < input_.size(); i++) {
            outDim.push_back(input_[perm_[i]]);
        }
    }

    Info::TransposeInfoCnnl(DataType dataType, Shape shape, Permutation perm) 
        : TransposeInfoCnnl(dataType, 
                            std::move(std::vector<int>(shape.begin(), shape.end())),
                            std::move(std::vector<int>(perm.begin(), perm.end()))) { }

    K::TransposeCnnl(Info info_) noexcept
        : Kernel(), info(std::move(info_)) { }

    auto K::build(DataType dataType, Shape shape, Permutation perm) noexcept -> KernelBox {
#ifndef USE_BANG
        return nullptr;
#endif
        return std::make_unique<K>(TransposeInfoCnnl(dataType, shape, perm));
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

        auto d = std::make_shared<Descriptors>(info.dataType != DT::F64);
        setCnnlTensor(d->x, info.dataType, slice(info.inDim.data(), info.inDim.size()));
        setCnnlTensor(d->y, info.dataType, slice(info.outDim.data(), info.outDim.size()));
        CNNL_ASSERT(cnnlSetTransposeDescriptor(d->trans, info.perm.size(), info.perm.data()));

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

#ifndef KERNEL_SCATTER_ND_CNNL_KERNEL_HH
#define KERNEL_SCATTER_ND_CNNL_KERNEL_HH

#include "kernel/attributes/scatter_nd_info.h"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct ScatterNDCnnl final : public Kernel {
        struct {
            DataType dataType, indexDataType, updateDataType;
            std::vector<int> inDim, indexDim, updateDim, outDim;
        } info;

        explicit ScatterNDCnnl(decltype(info));

        static KernelBox build(TensorRefs, TensorRefs) noexcept;

        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_SCATTER_ND_CNNL_KERNEL_HH

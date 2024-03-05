#ifndef KERNEL_SPLIT_CNNL_KERNEL_HH
#define KERNEL_SPLIT_CNNL_KERNEL_HH

#include "kernel/collectors/split.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {
    struct SplitInfoCnnl {
        DataType dataType;
        int axis;
        int num;
        std::vector<int> inDim;
        std::vector<std::vector<int>> outDims;

        SplitInfoCnnl(DataType, int, int, std::vector<int>, std::vector<std::vector<int>>);
        SplitInfoCnnl(int, Tensor const &, TensorRefs);
    };

    struct SplitCnnl final : public Kernel {
        SplitInfoCnnl info;

        explicit SplitCnnl(SplitInfoCnnl) noexcept;

        static KernelBox build(int, Tensor const &, TensorRefs) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_SPLIT_CNNL_KERNEL_HH

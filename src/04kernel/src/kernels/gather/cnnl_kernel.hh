#ifndef KERNEL_GATHER_CNNL_KERNEL_HH
#define KERNEL_GATHER_CNNL_KERNEL_HH

#include "kernel/attributes/gather_info.h"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct GatherCnnl final : public Kernel {
        struct {
            DataType dataType, indexDataType;
            int axis;
            std::vector<int> inDim, indexDim, outDim;
        } info;

        explicit GatherCnnl(decltype(info)) noexcept;

        static KernelBox build(int, Tensor const &, Tensor const &, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_TRANSPOSE_CNNL_KERNEL_HH

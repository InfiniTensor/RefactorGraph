#ifndef KERNEL_WHERE_CNNL_HH
#define KERNEL_WHERE_CNNL_HH

#include "kernel/collectors/where.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct WhereCnnl final : public Kernel {
        struct {
            DataType dataType;
            std::vector<int> condDim, thenDim, elseDim, outputDim;
        } info;

        WhereCnnl(decltype(info)) noexcept;

        static KernelBox build(TensorRefs const &, TensorRefs const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_WHERE_CNNL_HH

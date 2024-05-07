#ifndef KERNEL_SLICE_CNNL_KERNEL_HH
#define KERNEL_SLICE_CNNL_KERNEL_HH

#include "kernel/attributes/slice_info.h"
#include "kernel/collectors/slice.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct SliceCnnl final : public Kernel {
        struct {
            DataType dataType;
            Dimensions dims;
            std::vector<int> inDim, outDim;
        } info;

        explicit SliceCnnl(decltype(info)) noexcept;

        static KernelBox build(DataType, Dimensions, Shape, Shape) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_SLICE_CNNL_KERNEL_HH

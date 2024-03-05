#ifndef KERNEL_CLIP_CNNL_KERNEL_HH
#define KERNEL_CLIP_CNNL_KERNEL_HH

#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct ClipCnnl final : public Kernel {
        DataType dataType;
        std::vector<int> shape;
        bool hasMax;

        ClipCnnl(decltype(dataType), decltype(shape), decltype(hasMax)) noexcept;

        static KernelBox build(Tensor const &, bool hasMax) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_CLIP_CNNL_KERNEL_HH

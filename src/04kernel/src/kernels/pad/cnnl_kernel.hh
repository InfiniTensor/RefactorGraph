#ifndef KERNEL_PAD_CNNL_HH
#define KERNEL_PAD_CNNL_HH

#include "kernel/attributes/pad_info.h"
#include "kernel/collectors/pad.h"

namespace refactor::kernel {

    struct PadCnnl final : public Kernel {
        DataType dataType;
        PadType mode;
        std::vector<int> inDim, outDim, padDim;
        size_t valueLength;

        PadCnnl(DataType, PadType, std::vector<int>, std::vector<int>, std::vector<int>, size_t) noexcept;
        static KernelBox build(PadDimension, DataType, PadType, std::optional<std::reference_wrapper<Tensor const>>) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif//KERNEL_PAD_CNNL_HH

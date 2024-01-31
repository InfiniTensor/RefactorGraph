#ifndef KERNEL_SELECT_CUDA_KERNEL_HH
#define KERNEL_SELECT_CUDA_KERNEL_HH

#include "kernel/attributes/broadcaster.h"
#include "kernel/collectors/select.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct SelectCuda final : public Kernel {
        DataType dataType;
        SelectType selectType;
        Broadcaster broadcaster;
        size_t inputsNum;

        SelectCuda(decltype(dataType), decltype(selectType), decltype(broadcaster), decltype(inputsNum)) noexcept;

        static KernelBox build(SelectType, TensorRefs) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_SELECT_CUDA_KERNEL_HH

#ifndef KERNEL_SELECT_CPU_KERNEL_HH
#define KERNEL_SELECT_CPU_KERNEL_HH

#include "kernel/attributes/broadcaster.h"
#include "kernel/collectors/select.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct SelectCpu final : public Kernel {
        DataType dataType;
        SelectType selectType;
        Broadcaster broadcaster;
        size_t inputsNum;

        SelectCpu(decltype(dataType), decltype(selectType), decltype(broadcaster), decltype(inputsNum)) noexcept;

        static KernelBox build(SelectType, TensorRefs) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        RoutineWorkspace lower(Resources &) const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_Select_CPU_KERNEL_HH

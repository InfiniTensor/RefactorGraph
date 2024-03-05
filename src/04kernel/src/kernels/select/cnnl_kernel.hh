#ifndef KERNEL_SELECT_CNNL_KERNEL_HH
#define KERNEL_SELECT_CNNL_KERNEL_HH

#include "kernel/attributes/broadcaster.h"
#include "kernel/collectors/select.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct SelectCnnl final : public Kernel {
        DataType dataType;
        SelectType selectType;
        std::vector<std::vector<int>> inputDims;
        std::vector<std::vector<int>> outputDims;
        size_t inputsNum;

        SelectCnnl(decltype(dataType), decltype(selectType), decltype(inputDims),
                   decltype(outputDims), decltype(inputsNum)) noexcept;

        static KernelBox build(SelectType, TensorRefs) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_SELECT_CNNL_KERNEL_HH

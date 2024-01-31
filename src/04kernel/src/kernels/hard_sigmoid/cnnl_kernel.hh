#ifndef KERNEL_HARD_SIGMOID_CNNL_KERNEL_HH
#define KERNEL_HARD_SIGMOID_CNNL_KERNEL_HH

#include "kernel/collectors/simple_unary.h"

namespace refactor::kernel {

    struct HardSigmoidCnnl final : public Kernel {
        float alpha, beta;
        DataType dataType;
        int size;

        HardSigmoidCnnl(float, float, DataType, int) noexcept;

        static KernelBox build(float, float, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_HARD_SIGMOID_CNNL_KERNEL_HH

#ifndef KERNEL_HARD_SIGMOID_CUDA_KERNEL_HH
#define KERNEL_HARD_SIGMOID_CUDA_KERNEL_HH

#include "kernel/collectors/hard_sigmoid.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct HardSigmoidCuda final : public Kernel {
        float alpha, beta;
        DataType dataType;
        size_t size;

        explicit HardSigmoidCuda(float, float, DataType, size_t) noexcept;

        static KernelBox build(float, float, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_HARD_SIGMOID_CUDA_KERNEL_HH

#ifndef KERNEL_ALLREDUCE_NCCL_KERNEL_HH
#define KERNEL_ALLREDUCE_NCCL_KERNEL_HH

#include "kernel/collectors/all_reduce.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct AllReduceNccl final : public Kernel {
        AllReduceType opType;
        DataType dataType;
        size_t size;

        AllReduceNccl(AllReduceType, DataType, size_t) noexcept;

        static KernelBox build(AllReduceType, Tensor const &, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_ALLREDUCE_NCCL_KERNEL_HH

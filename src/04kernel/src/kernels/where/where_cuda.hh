#ifndef KERNEL_WHERE_CUDA_HH
#define KERNEL_WHERE_CUDA_HH

#include "kernel/attributes/broadcaster.h"
#include "kernel/collectors/where.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct WhereCuda final : public Kernel {
        DataType dataType;
        Broadcaster broadcaster;

        WhereCuda(DataType, Broadcaster) noexcept;

        static KernelBox build(TensorRefs const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_WHERE_CUDA_HH

#ifndef KERNEL_SOFTMAX_CPU_KERNEL_HH
#define KERNEL_SOFTMAX_CPU_KERNEL_HH

#include "kernel/collectors/softmax.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"
#include <numeric>

namespace refactor::kernel {

    struct AxisInfo {
        uint_lv2 pre, mid, post, size;
        DataType type;

        AxisInfo(Tensor const &data, uint_lv2 axis) noexcept;
        void locate(uint_lv2 k, uint_lv2 ans[]) const noexcept;
    };

    struct SoftmaxCpu final : public Kernel {
        AxisInfo info;

        explicit SoftmaxCpu(AxisInfo) noexcept;

        static KernelBox build(AxisInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;

        Routine lower() const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SOFTMAX_CPU_KERNEL_HH
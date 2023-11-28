#ifndef KERNEL_EXPAND_INFO_H
#define KERNEL_EXPAND_INFO_H

#include "../tensor.h"

namespace refactor::kernel {

    /// @brief 优化用于计算的单向广播描述。
    struct ExpandInfo {
        struct Dim {
            dim_t i, o;

            bool operator==(Dim const &) const noexcept;
            bool operator!=(Dim const &) const noexcept;
        };

        /// @brief 所有输入输出的各维度步长。
        std::vector<Dim> strides;
        dim_t blockCount, blockSize;

        ExpandInfo(DataType, slice_t<dim_t> input, slice_t<dim_t> output) noexcept;
        ExpandInfo(Tensor const &input, Tensor const &output) noexcept;
        ExpandInfo reform(dim_t maxblockSize) const noexcept;
        void reformAssign(dim_t maxblockSize) noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_EXPAND_INFO_H

#ifndef KERNEL_GATHER_INFO_H
#define KERNEL_GATHER_INFO_H

#include "../tensor.h"

namespace refactor::kernel {

    /// @brief 优化用于计算的 Gather 描述。
    struct GatherInfo {
        /// @brief Gather 的计算是 `prefix` 次将 `postfix` 大小的数据从输入拷贝到输出。
        ///        其中输入的总大小是 `prefix * midSizeI * postfix`，
        ///        输出的总大小是 `prefix * midSizeO * postfix`，
        ///        通过保存这两个值可以计算输入输出位置的偏移。
        uint_lv2 prefix, postfix, midSizeI, midSizeO;
        /// @brief `indices` 的数据类型，可以是 `I32` 或 `I64`。
        DataType idxType;

        GatherInfo(uint_lv2 axis, Tensor const &data, Tensor const &indices) noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_GATHER_INFO_H

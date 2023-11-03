#ifndef KERNEL_BROADCASTER_H
#define KERNEL_BROADCASTER_H

#include "../tensor.h"
#include <vector>

namespace refactor::kernel {

    /// @brief 优化用于计算的通用广播描述。
    struct Broadcaster {
        /// @brief 所有输入输出的各维度步长。
        std::vector<uint_lv2> strides;
        /// @brief 输出的总大小和输入的数量。
        uint_lv2 outputsCount, inputsCount;

        explicit Broadcaster(std::vector<slice_t<uint_lv2>>) noexcept;
        explicit Broadcaster(TensorRefs const &inputs) noexcept;
        void locate(uint_lv2 k, uint_lv2 ans[]) const noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_BROADCASTER_H

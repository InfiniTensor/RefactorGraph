#ifndef KERNEL_SLICE_INFO_H
#define KERNEL_SLICE_INFO_H

#include "../tensor.h"

namespace refactor::kernel {
    namespace slice {
        struct Dim {
            int64_t start, step, length;
        };
    }// namespace slice

    using Dimensions = std::vector<slice::Dim>;

    /// @brief 优化用于计算的 Slice 描述。
    struct SliceInfo {
        struct Dim {
            dim_t countStride, sizeStart;
            sdim_t sizeStride;

            bool operator==(Dim const &) const noexcept;
            bool operator!=(Dim const &) const noexcept;
        };
        std::vector<Dim> dims;
        dim_t blockSize;

        SliceInfo(Dimensions const &, Tensor const &) noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_SLICE_INFO_H

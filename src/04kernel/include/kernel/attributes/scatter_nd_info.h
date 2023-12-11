#ifndef KERNEL_SCATTER_ND_INFO_H
#define KERNEL_SCATTER_ND_INFO_H

#include "../tensor.h"

namespace refactor::kernel {

    /// @brief 优化用于计算的 ScatterND 描述。
    struct ScatterNDInfo {
        dim_t prefix;
        std::vector<dim_t> strides;
        size_t blockSize;

        ScatterNDInfo(Tensor const &data,
                      Tensor const &indices);
    };

}// namespace refactor::kernel

#endif// KERNEL_SCATTER_ND_INFO_H

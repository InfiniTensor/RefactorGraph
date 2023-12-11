#ifndef KERNEL_SCATTER_ND_H
#define KERNEL_SCATTER_ND_H

#include "../collector.h"

namespace refactor::kernel {

    struct ScatterNDCollector final : public InfoCollector {

        explicit ScatterNDCollector(decltype(_target)) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SCATTER_ND_H

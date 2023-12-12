#ifndef KERNEL_CAST_H
#define KERNEL_CAST_H

#include "../collector.h"

namespace refactor::kernel {

    struct CastCollector final : public InfoCollector {

        explicit CastCollector(decltype(_target)) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_CAST_H

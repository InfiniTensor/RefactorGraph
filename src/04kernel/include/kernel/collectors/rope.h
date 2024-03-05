#ifndef KERNEL_ROPE_COLLECTOR_H
#define KERNEL_ROPE_COLLECTOR_H

#include "../collector.h"

namespace refactor::kernel {
    struct RoPECollector final : public InfoCollector {
        float theta;
        constexpr RoPECollector(decltype(_target) target, float _theta) noexcept
            : InfoCollector(target), theta(_theta){}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif

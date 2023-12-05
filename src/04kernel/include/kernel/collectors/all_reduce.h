#ifndef KERNEL_COLLECTOR_ALL_REDUCE_H
#define KERNEL_COLLECTOR_ALL_REDUCE_H

#include "../collector.h"
#include "kernel/attributes/communication.h"

namespace refactor::kernel {

    struct AllReduceCollector final : public InfoCollector {

        AllReduceType type;

        constexpr AllReduceCollector(decltype(_target) target, AllReduceType type_) noexcept
            : InfoCollector(target), type(type_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };
}// namespace refactor::kernel

#endif
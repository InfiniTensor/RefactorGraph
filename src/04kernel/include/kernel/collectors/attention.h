#ifndef KERNEL_ATTENTION_H
#define KERNEL_ATTENTION_H

#include "../collector.h"

namespace refactor::kernel {

    struct AttentionCollector final : public InfoCollector {

        AttentionCollector(decltype(_target)) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_ATTENTION_H

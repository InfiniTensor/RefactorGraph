#ifndef KERNEL_ATTENTION_H
#define KERNEL_ATTENTION_H

#include "../collector.h"

namespace refactor::kernel {

    struct AttentionCollector final : public InfoCollector {
        dim_t maxSeqLen;

        AttentionCollector(decltype(_target), decltype(maxSeqLen)) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_ATTENTION_H
